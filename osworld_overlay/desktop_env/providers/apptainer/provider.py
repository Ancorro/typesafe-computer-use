"""OSWorld's Docker VM, run with Apptainer instead of Docker, for clusters that have no Docker.

OSWorld's Docker provider starts happysixd/osworld-docker, whose entrypoint runs QEMU with a tap
network that needs root. This provider runs the same QEMU from the same image (as an Apptainer
.sif), with the same machine, disk, firmware and devices, but with QEMU's user-mode network, which
forwards the four ports the harness uses to the guest without root. Every VM gets its own
copy-on-write overlay of the VM image and its own ports, so several run side by side on one node.

It replaces the Docker provider when OSWORLD_VM_RUNTIME=apptainer (see use_if_requested), so the
run keeps OSWorld's provider_name "docker" and everything DesktopEnv does for it. Settings:

  OSWORLD_SIF        the .sif built from the Docker provider's image (required)
  OSWORLD_VM_STAGE   a node-local directory; the VM image is copied there once per node, so the
                     VMs read it locally rather than over NFS (default: read it where it is)
  OSWORLD_VM_TMP     where each VM's overlay and log go (default $SLURM_TMPDIR, else /tmp; on
                     clusters where /tmp is RAM, point it at node-local disk)
"""
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import requests
from filelock import FileLock

from desktop_env.providers.docker.provider import DockerProvider

logger = logging.getLogger("desktopenv.providers.apptainer.ApptainerProvider")
logger.setLevel(logging.INFO)

# The firmware the Docker image boots (its /storage/uefi.rom is a copy of this file).
FIRMWARE = "/usr/share/OVMF/edk2-x86_64-code.fd"
BOOT_TIMEOUT = 600


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class ApptainerProvider(DockerProvider):
    def __init__(self, region: str):
        # DockerProvider's own __init__ connects to a Docker daemon; this keeps its settings only.
        self.region = region
        self.server_port = self.vnc_port = self.chromium_port = self.vlc_port = None
        self.container = None
        self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}
        self.sif = os.environ.get("OSWORLD_SIF")
        if not self.sif or not os.path.exists(self.sif):
            raise RuntimeError(f"OSWORLD_SIF must name the OSWorld .sif image, got {self.sif!r}")
        self.tmp = Path(os.environ.get("OSWORLD_VM_TMP") or os.environ.get("SLURM_TMPDIR") or "/tmp")
        self.tmp.mkdir(parents=True, exist_ok=True)
        # Ports are the node's, so the lock is too: /tmp is shared by every job on a node.
        self.lock_file = Path("/tmp") / f"osworld_apptainer_ports_{os.getuid()}.lck"
        self.vm_dir = None
        self.process = None

    def _exec(self, *args, **kwargs):
        binds = ",".join(sorted({str(self.tmp), str(Path(self.base).parent)}))
        return ["apptainer", "exec", "--bind", binds, self.sif, *args]

    def _port(self, start: int, taken: set) -> int:
        port = start
        while port in taken or not _free(port):
            port += 1
        taken.add(port)
        return port

    def _stage(self, path_to_vm: str) -> str:
        """The VM image, copied once per node to OSWORLD_VM_STAGE when it is set."""
        src = os.path.abspath(path_to_vm)
        stage = os.environ.get("OSWORLD_VM_STAGE")
        if not stage:
            return src
        os.makedirs(stage, exist_ok=True)
        dst = os.path.join(stage, os.path.basename(src))
        with FileLock(dst + ".lck", timeout=3600):
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                logger.info("Copying the VM image to %s", dst)
                shutil.copyfile(src, dst + ".part")
                os.replace(dst + ".part", dst)
        return dst

    def _wait_for_vm_ready(self, timeout: int = BOOT_TIMEOUT):
        start = time.time()
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(f"QEMU exited with {self.process.returncode}; see {self.vm_dir}/qemu.log")
            try:
                if requests.get(f"http://localhost:{self.server_port}/screenshot", timeout=(10, 10)).status_code == 200:
                    logger.info("VM ready after %.0f s", time.time() - start)
                    return True
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError("VM failed to become ready within timeout period")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        self.base = self._stage(path_to_vm)
        try:
            with FileLock(str(self.lock_file), timeout=60):
                taken = set()
                self.server_port = self._port(5000, taken)
                self.chromium_port = self._port(9222, taken)
                self.vlc_port = self._port(8080, taken)
                self.vnc_port = self._port(5900, taken)  # QEMU's own VNC server, not noVNC
                self.vm_dir = self.tmp / f"osworld-vm-{self.server_port}"
                shutil.rmtree(self.vm_dir, ignore_errors=True)
                self.vm_dir.mkdir(parents=True)
                disk = self.vm_dir / "boot.qcow2"
                subprocess.run(self._exec("qemu-img", "create", "-q", "-f", "qcow2", "-b", self.base,
                                          "-F", "qcow2", str(disk)), check=True)
                firmware = self.vm_dir / "uefi.rom"
                subprocess.run(self._exec("cp", FIRMWARE, str(firmware)), check=True)
                os.chmod(firmware, 0o644)
                kvm = os.access("/dev/kvm", os.R_OK | os.W_OK)
                if not kvm:
                    logger.warning("No usable /dev/kvm: the VM runs without hardware acceleration")
                cores, ram = self.environment["CPU_CORES"], self.environment["RAM_SIZE"]
                forwards = ",".join(f"hostfwd=tcp:127.0.0.1:{host}-:{guest}" for host, guest in
                                    ((self.server_port, 5000), (self.chromium_port, 9222), (self.vlc_port, 8080)))
                mac = "02:%02X:%02X:%02X:%02X:%02X" % tuple(os.urandom(5))
                # The Docker image's command line, with user-mode networking and no monitor.
                qemu = [
                    "qemu-system-x86_64",
                    "-cpu", "host,kvm=on,l3-cache=on,+hypervisor,migratable=no,+invtsc" if kvm else "max",
                    "-smp", f"{cores},sockets=1,dies=1,cores={cores},threads=1",
                    "-m", ram,
                    "-machine", "type=q35,smm=off,graphics=off,vmport=off,dump-guest-core=off,hpet=off,"
                                + ("accel=kvm" if kvm else "accel=tcg"),
                    *(["-enable-kvm", "-global", "kvm-pit.lost_tick_policy=discard"] if kvm else []),
                    "-display", f"vnc=127.0.0.1:{self.vnc_port - 5900}",
                    "-vga", "virtio",
                    "-name", "qemu,process=qemu,debug-threads=on",
                    "-serial", f"file:{self.vm_dir}/serial.log",
                    "-device", "qemu-xhci,id=xhci",
                    "-device", "usb-tablet",
                    "-netdev", f"user,id=hostnet0,{forwards}",
                    "-device", f"virtio-net-pci,romfile=,netdev=hostnet0,mac={mac},id=net0",
                    "-hda", str(disk),
                    "-pflash", str(firmware),
                    "-object", "rng-random,id=objrng0,filename=/dev/urandom",
                    "-device", "virtio-rng-pci,rng=objrng0,id=rng0,bus=pcie.0,addr=0x1c",
                    "-device", "virtio-balloon-pci,id=balloon0,bus=pcie.0,addr=0x4",
                ]
                log = open(self.vm_dir / "qemu.log", "w")
                self.process = subprocess.Popen(self._exec(*qemu), stdout=log, stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL, start_new_session=True)
                # Hold the lock until QEMU has bound its ports, so the next VM cannot pick them.
                deadline = time.time() + 60
                while _free(self.server_port) and self.process.poll() is None and time.time() < deadline:
                    time.sleep(0.5)
            logger.info(f"Started QEMU (apptainer, pid {self.process.pid}) with ports - VNC: {self.vnc_port}, "
                        f"Server: {self.server_port}, Chrome: {self.chromium_port}, VLC: {self.vlc_port}")
            self._wait_for_vm_ready()
        except Exception:
            self.stop_emulator(path_to_vm)
            raise

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        if self.process:
            logger.info("Stopping VM...")
            try:
                self.process.terminate()
                self.process.wait(timeout=20)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=20)
        if self.vm_dir and not os.environ.get("OSWORLD_KEEP_VM_DIR"):
            shutil.rmtree(self.vm_dir, ignore_errors=True)
        self.process = None
        self.vm_dir = None
        self.server_port = self.vnc_port = self.chromium_port = self.vlc_port = None


def use_if_requested():
    """With OSWORLD_VM_RUNTIME=apptainer, OSWorld's "docker" provider builds ApptainerProvider."""
    if os.environ.get("OSWORLD_VM_RUNTIME") != "apptainer":
        return
    import desktop_env.providers.docker.provider as docker_provider
    docker_provider.DockerProvider = ApptainerProvider
    logger.info("VM runtime: apptainer (%s)", os.environ.get("OSWORLD_SIF"))
