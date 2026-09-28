"""scripts/osworld and scripts/osworld-gcp take several tasks per run, and never reach a machine here.

scripts/osworld runs only where OSWorld is set up, so its task list is checked by running that one
function alone. scripts/osworld-gcp is checked through --dry-run, which prints each command and runs
none.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OSWORLD = ROOT / "scripts" / "osworld"
OSWORLD_GCP = ROOT / "scripts" / "osworld-gcp"

POPEN = subprocess.Popen  # the real one, before tests/conftest.py refuses it for every test

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="the scripts are bash")


@pytest.fixture(autouse=True)
def processes(monkeypatch):
    """These tests run bash on files only; tests/conftest.py refuses every process by default."""
    monkeypatch.setattr(subprocess, "Popen", POPEN)


def function(script: Path, name: str) -> str:
    """One shell function's source, from its `name() {` line to the closing brace at column one."""
    match = re.search(rf"^{name}\(\) {{\n.*?^}}\n", script.read_text(), re.S | re.M)
    assert match, f"{script.name} has no {name}()"
    return match.group(0)


def task_list(tmp_path: Path, *tasks: str) -> tuple[Path, dict]:
    source = function(OSWORLD, "task_meta")
    out = subprocess.run(
        ["bash", "-c", f'set -euo pipefail; REPO="$1"; shift; {source}\ntask_meta "$@"', "bash", str(tmp_path), *tasks],
        capture_output=True,
        text=True,
        check=True,
    )
    meta = Path(out.stdout.strip())
    return meta, json.loads(meta.read_text())


def test_one_task_is_its_own_list(tmp_path):
    meta, tasks = task_list(tmp_path, "chrome/bb5e4c0d")
    assert meta == tmp_path / ".osworld" / "tasks" / "chrome__bb5e4c0d.json"
    assert tasks == {"chrome": ["bb5e4c0d"]}


def test_several_tasks_group_by_domain_in_the_order_given(tmp_path):
    meta, tasks = task_list(tmp_path, "chrome/b", "os/x", "chrome/a")
    assert meta.name.startswith("list-")
    assert list(tasks.items()) == [("chrome", ["b", "a"]), ("os", ["x"])]


def test_a_different_list_gets_a_different_file(tmp_path):
    first, _ = task_list(tmp_path, "chrome/a", "chrome/b")
    second, _ = task_list(tmp_path, "chrome/a", "chrome/c")
    assert first != second


def dry_run(tmp_path: Path, *args: str) -> str:
    """scripts/osworld-gcp --dry-run in a bare copy of the repo, so no cache or result of this checkout is read.

    A dry run runs nothing, and should it ever try, gcloud, terraform, and rsync here only fail.
    """
    tmp_path.mkdir(exist_ok=True)
    shims = tmp_path / "bin"
    shims.mkdir()
    for tool in ("gcloud", "terraform", "rsync"):
        (shims / tool).write_text(f"#!/bin/sh\necho 'a dry run ran {tool}' >&2\nexit 99\n")
        (shims / tool).chmod(0o755)
    (tmp_path / "scripts").mkdir()
    shutil.copy(OSWORLD_GCP, tmp_path / "scripts" / "osworld-gcp")
    (tmp_path / ".osworld").mkdir()
    (tmp_path / ".osworld" / "gcp.json").write_text(
        '{\n  "project": "p",\n  "zone": "z",\n  "instance": "osworld",\n  "ssh": "iap"\n}\n'
    )
    out = subprocess.run(
        ["bash", str(tmp_path / "scripts" / "osworld-gcp"), "--dry-run", *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env={"PATH": f"{shims}:/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    return out.stderr


def test_the_cloud_run_passes_every_task_to_the_machine(tmp_path):
    printed = dry_run(tmp_path, "run-jev", "chrome/a", "chrome/b", "--ocr", "rapidocr")
    assert "osworld-exec scripts/osworld run-jev chrome/a chrome/b --ocr rapidocr" in printed


def test_the_cloud_run_lives_on_the_machine_not_in_the_connection(tmp_path):
    printed = dry_run(tmp_path, "run-jev", "chrome/a", "--ocr", "rapidocr")
    assert "setsid nohup bash -c" in printed, "the run is its own session on the machine"
    assert ".osworld/runs/" in printed and ".status" in printed, "its log and exit status stay on the machine"
    assert "tail -n +1 -F --pid=4242" in printed, "and this end follows its log"
    assert "ServerAliveInterval=15" in printed, "noticing a stalled stream instead of hanging on it"


def test_attach_follows_the_newest_run_and_cancel_stops_its_whole_session(tmp_path):
    assert "tail -n +1 -F --pid=4242" in dry_run(tmp_path / "a", "attach")
    assert "kill -TERM -- -4242" in dry_run(tmp_path / "b", "cancel")


def test_the_cloud_run_moves_aside_the_local_copy_of_each_task(tmp_path):
    for task in ("chrome/a", "chrome/b"):
        (tmp_path / "results" / "pyautogui" / "screenshot" / "jev" / task).mkdir(parents=True)
    printed = dry_run(tmp_path, "run-jev", "chrome/a", "chrome/b", "--ocr", "rapidocr")
    moved = [line for line in printed.splitlines() if line.startswith("+ mv ")]
    assert len(moved) == 2 and all("results/archive/" in line for line in moved)


def test_the_cloud_run_needs_a_task(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as failed:
        dry_run(tmp_path, "run-jev", "--ocr", "rapidocr")
    assert "usage: scripts/osworld-gcp run-jev DOMAIN/ID... --ocr OCR" in failed.value.stderr
