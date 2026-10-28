"""Windows adapter: synthetic input, app control, screen capture, OCR, and the focused UI Automation element.

Experimental. This is the only module that touches pywin32, UI Automation, Win32 SendInput, or
Windows.Media.Ocr; macos.py provides the same functions for macOS, and platform_adapter.py picks one.
The bounded tree walk itself lives in ax_walk.py, shared by both.

Once the process is per-monitor DPI aware (set below), Windows has no Retina-style point and pixel
split, so a screen point here already is a capture pixel. The pure rules (the role names, what counts
as pressable, which window belongs to an app, the input events) are plain functions, tested off Windows.
"""

from __future__ import annotations

import ctypes
import functools
import os
import shutil
import subprocess
import time
import webbrowser
from contextlib import suppress
from pathlib import Path

import psutil
import uiautomation as auto
import win32api
import win32con
import win32gui
import pywintypes
import win32process
import winocr
from PIL import Image, ImageGrab

from .ax_walk import AX_PRESS, AxAttrs, Frame, walk_actionable
from .config import ABORT_CORNER_PX
from .models import Abort, AxNode, Field, Missed

with suppress(AttributeError, OSError):  # pre-8.1 Windows without shcore, or awareness set by the host process
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE

MIN_WINDOW_SIDE_PT = 50.0  # anything smaller is a palette or a shadow, not the window being worked in
EVENT_GAP = 0.04  # seconds between synthetic events, as on macOS
CLICK_TOLERANCE_PX = 1  # how far the cursor may land from its target before a click is refused
WHEEL_DELTA = 120  # one notch of the wheel
LINES_PER_NOTCH = 3  # the Windows default; the actions ask for lines, as on macOS

# CU_MONITOR picks the one monitor captured and acted on, by a piece of its device ID or name ("BNQ", "BenQ",
# "DISPLAY6"); unset, it is the primary. Every coordinate the rest of the code sees is relative to that monitor's
# top-left corner, so captures, items and clicks line up as they do on the primary. Read on every call, so a value
# from .env counts.


def monitor_rect() -> tuple[int, int, int, int]:
    """left, top, right, bottom of the chosen monitor in virtual-screen pixels; a ValueError when none matches."""
    wanted = os.environ.get("CU_MONITOR", "").strip().lower()
    found = []
    for handle, _, _ in win32api.EnumDisplayMonitors():
        info = win32api.GetMonitorInfo(handle)
        device = win32api.EnumDisplayDevices(info["Device"], 0)
        names = f"{info['Device']} {device.DeviceID} {device.DeviceString}".lower()
        if (wanted in names) if wanted else info["Flags"] & win32con.MONITORINFOF_PRIMARY:
            found.append(tuple(info["Monitor"]))
    if len(found) != 1:
        raise ValueError(f"CU_MONITOR={wanted!r} matches {len(found)} monitors, not one")
    return found[0]


# With CU_MONITOR set, the person keeps working on the other monitors: their windows are never moved, only windows on
# the chosen monitor are switched to, and keys go nowhere unless the window in front is on it. A window this process
# started itself is its own and may be moved there.
_OWNED: set[int] = set()


def _chosen() -> bool:
    return bool(os.environ.get("CU_MONITOR", "").strip())


def _on_monitor(hwnd: int) -> bool:
    """Whether a window sits mostly on the chosen monitor (Windows' own nearest-monitor rule)."""
    info = win32api.GetMonitorInfo(win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST))
    return tuple(info["Monitor"]) == monitor_rect()


def _may_use(hwnd: int) -> bool:
    return not _chosen() or hwnd in _OWNED or (not win32gui.IsIconic(hwnd) and _on_monitor(hwnd))


def _keys_allowed() -> None:
    """Keys go to the window in front; with CU_MONITOR set it must be on that monitor, or the keys would land in
    whatever the person is working in elsewhere."""
    if _chosen() and not _may_use(win32gui.GetForegroundWindow()):
        raise Missed("the window in front is on another monitor; click a window on this screen first")


def _origin() -> tuple[int, int]:
    left, top, _, _ = monitor_rect()
    return left, top

# Windows.Media.Ocr reads one language per engine and reports no confidence, so every line it returns
# counts as certain and passes MIN_OCR_CONFIDENCE. The language pack must be installed (see docs/windows.md).
OCR_LANGUAGE = "en"
OCR_CONFIDENCE = 1.0

# UI Automation's ControlTypeName mapped onto the "AX*" role vocabulary ax_walk.py, models.ROLE_WORDS,
# and models.TEXT_ROLES already speak, so none of that shared logic needs a Windows-specific branch.
CONTROL_TYPE_TO_ROLE = {
    "ButtonControl": "AXButton",
    "CheckBoxControl": "AXCheckBox",
    "ComboBoxControl": "AXComboBox",
    "DataItemControl": "AXCell",
    "EditControl": "AXTextField",
    "GroupControl": "AXGroup",
    "HyperlinkControl": "AXLink",
    "ImageControl": "AXImage",
    "ListItemControl": "AXRow",
    "MenuItemControl": "AXMenuBarItem",
    "RadioButtonControl": "AXRadioButton",
    "SliderControl": "AXSlider",
    "SpinnerControl": "AXIncrementor",
    "TabItemControl": "AXTab",
    "TextControl": "AXStaticText",
}

# The browsers the site catalog can open, by the name CLICKER_BROWSER gives and their executable.
# The frontmost app is reported by the same name, so the classifier sees one browser, not two.
BROWSER_EXES = {"Google Chrome": "chrome", "Microsoft Edge": "msedge", "Firefox": "firefox", "Brave Browser": "brave"}

# Virtual keys for the keys the actions press. macOS's Command becomes Control, except Command-[,
# a browser's Back, which is Alt-Left on Windows.
VK = {"return": 0x0D, "tab": 0x09, "escape": 0x1B, "a": 0x41, "delete": 0x2E, "[": 0xDB, "left": 0x25}
VK_CONTROL = 0x11
VK_ALT = 0x12
VK_SHIFT = 0x10
VK_WIN = 0x5B
# Every key a model may name (the loop's key action, xdotool/pyautogui-style names), letters and digits included.
NAMED_KEYS = {
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B, "space": 0x20, "backspace": 0x08,
    "delete": 0x2E, "del": 0x2E, "insert": 0x2D, "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "pgup": 0x21, "pgdn": 0x22, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "ctrl": VK_CONTROL, "control": VK_CONTROL, "alt": VK_ALT, "shift": VK_SHIFT,
    "win": VK_WIN, "super": VK_WIN, "windows": VK_WIN, "meta": VK_WIN, "cmd": VK_WIN, "command": VK_WIN,
    "capslock": 0x14, "printscreen": 0x2C, "menu": 0x5D, "apps": 0x5D,
    "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, ";": 0xBA, "'": 0xDE, ",": 0xBC, ".": 0xBE, "/": 0xBF, "\\": 0xDC, "`": 0xC0,
    **{f"f{n}": 0x6F + n for n in range(1, 13)},
    **{chr(c): c - 32 for c in range(ord("a"), ord("z") + 1)},
    **{str(d): 0x30 + d for d in range(10)},
}
# Delete, Insert, Home, End, Page Up/Down, the arrows and the Windows keys live on the extended keypad
EXTENDED_KEYS = {0x2E, 0x2D, 0x24, 0x23, 0x21, 0x22, 0x25, 0x26, 0x27, 0x28, VK_WIN, 0x5D}
COMMAND_CHORDS = {"[": (VK_ALT, VK["left"])}

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800


# ------------------------------------------------------------------ pure rules


def role_for(control_type_name: str) -> str:
    """The AX role a UI Automation control type stands for, or "" for one the walk ignores."""
    return CONTROL_TYPE_TO_ROLE.get(control_type_name, "")


def pressable(has_invoke: bool, default_action: str) -> bool:
    """Whether an element accepts a press. Almost every element carries the LegacyIAccessible
    pattern, so that pattern alone says nothing; only a named default action makes it a control."""
    return has_invoke or bool(default_action.strip())


def exe_stem(name: str) -> str:
    """`Chrome.exe`, `chrome` and ` CHROME ` are one process name."""
    stem = name.strip().lower()
    return stem.removesuffix(".exe")


def app_matches(app: str, process_name: str) -> bool:
    """Whether a process is the app a caller named: the executable itself, or a known browser by
    its product name. Exact, never a substring, and never by window title, so a Chrome tab named
    after an app is not that app."""
    wanted = exe_stem(app)
    browsers = {name.lower(): exe for name, exe in BROWSER_EXES.items()}
    return exe_stem(process_name) in {wanted, browsers.get(wanted, wanted)}


def display_name(process_name: str) -> str:
    """The name to report for a process: a known browser by its product name, anything else as is."""
    stem = exe_stem(process_name)
    return next((name for name, exe in BROWSER_EXES.items() if exe == stem), process_name)


def vk_for(key: str) -> int:
    """The virtual key a key name stands for (any case); a ValueError names one it does not know."""
    vk = NAMED_KEYS.get(key.strip().lower()) if key.strip() else None
    if vk is None:
        raise ValueError(f"no Windows key named {key!r}")
    return vk


def key_events(key: str, command: bool = False) -> list[tuple[int, int]]:
    """(virtual key, flags) for pressing a key, with its modifiers held around it."""
    keys = COMMAND_CHORDS.get(key, (VK_CONTROL, vk_for(key))) if command else (vk_for(key),)
    return chord_events(keys)


def chord_events(keys) -> list[tuple[int, int]]:
    """(virtual key, flags) for a chord: each key down in order, then up in reverse."""
    down = [(vk, KEYEVENTF_EXTENDEDKEY if vk in EXTENDED_KEYS else 0) for vk in keys]
    up = [(vk, flags | KEYEVENTF_KEYUP) for vk, flags in reversed(down)]
    return down + up


def unicode_events(text: str) -> list[tuple[int, int]]:
    """(UTF-16 code unit, flags) for typing text as characters, not keys, so any script and any
    emoji arrive whatever the keyboard layout. A character outside the BMP is two units."""
    data = text.encode("utf-16-le")
    units = [int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2)]
    return [(unit, flags) for unit in units for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)]


def wheel_delta(lines: int) -> int:
    """The wheel delta that scrolls `lines` lines; positive scrolls up, as on macOS."""
    return round(lines * WHEEL_DELTA / LINES_PER_NOTCH)


def landed(target: tuple[int, int], actual: tuple[float, float]) -> bool:
    return abs(actual[0] - target[0]) <= CLICK_TOLERANCE_PX and abs(actual[1] - target[1]) <= CLICK_TOLERANCE_PX


def ocr_lines(result: dict) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """winocr's plain-dict result as text, confidence, and a box around the line's words."""
    out = []
    for line in result.get("lines", []):
        rects = [word["bounding_rect"] for word in line.get("words", [])]
        if not rects:
            continue
        x1 = min(r["x"] for r in rects)
        y1 = min(r["y"] for r in rects)
        x2 = max(r["x"] + r["width"] for r in rects)
        y2 = max(r["y"] + r["height"] for r in rects)
        out.append((line["text"], OCR_CONFIDENCE, (float(x1), float(y1), float(x2), float(y2))))
    return out


# ------------------------------------------------------------------ SendInput


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_int32),
        ("dy", ctypes.c_int32),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_uint16),
        ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput), ("ki", _KeyboardInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("u", _InputUnion)]


@functools.cache
def _user32():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(_Input), ctypes.c_int)
    user32.SendInput.restype = ctypes.c_uint
    return user32


def _send(event: _Input) -> None:
    """One synthetic event, then the same pause macOS leaves. A blocked event (a UAC prompt or the
    lock screen has the input desktop) raises rather than going missing."""
    if _user32().SendInput(1, ctypes.byref(event), ctypes.sizeof(_Input)) != 1:
        raise ctypes.WinError(ctypes.get_last_error())
    time.sleep(EVENT_GAP)


def _key(vk: int = 0, scan: int = 0, flags: int = 0) -> _Input:
    return _Input(type=INPUT_KEYBOARD, u=_InputUnion(ki=_KeyboardInput(wVk=vk, wScan=scan, dwFlags=flags)))


def _mouse(flags: int, data: int = 0) -> _Input:
    return _Input(type=INPUT_MOUSE, u=_InputUnion(mi=_MouseInput(mouseData=data & 0xFFFFFFFF, dwFlags=flags)))


# ------------------------------------------------------------------ escape hatch


def mouse_location() -> tuple[float, float]:
    x, y = win32api.GetCursorPos()
    left, top = _origin()
    return float(x - left), float(y - top)


def check_abort() -> None:
    x, y = mouse_location()
    if x <= ABORT_CORNER_PX and y <= ABORT_CORNER_PX:
        raise Abort("mouse in top-left corner")


def abort_hint() -> str:
    return "Ctrl-C, or slam the mouse into the top-left corner"


def sleep_watching(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        check_abort()
        time.sleep(0.1)


def accessibility_trusted() -> bool:
    """Windows has no Accessibility permission gate; confirm UI Automation can reach the desktop instead."""
    try:
        return auto.GetRootControl() is not None
    except Exception:
        return False


# ------------------------------------------------------------------ input


def _move(point: tuple[int, int]) -> None:
    """Windows refuses to move the cursor while an elevated window (Task Manager, an installer) or the secure desktop
    (a UAC prompt, the lock screen) has the input: that is a Missed action, not a crash."""
    try:
        left, top = _origin()
        win32api.SetCursorPos((point[0] + left, point[1] + top))
    except pywintypes.error as error:
        raise Missed(f"Windows would not move the cursor ({error.strerror or 'no reason given'}): an elevated window"
                     " or a locked screen has the input; close it or press a key instead") from error
    time.sleep(EVENT_GAP)


def click_at(point: tuple[float, float]) -> None:
    """Move, read back where the cursor landed, then press and release there.

    A cursor that did not reach the target (another desktop has the input, or the point is off
    every monitor) means the click would land somewhere unknown, so nothing is pressed.
    """
    target = (round(point[0]), round(point[1]))
    _move(target)
    actual = mouse_location()
    if not landed(target, actual):
        raise Missed(f"the cursor went to {actual}, not {target}")
    _send(_mouse(MOUSEEVENTF_LEFTDOWN))
    _send(_mouse(MOUSEEVENTF_LEFTUP))


def press(key: str, command: bool = False) -> None:
    _keys_allowed()
    for vk, flags in key_events(key, command):
        _send(_key(vk=vk, flags=flags))


def hotkey(*keys: str) -> None:
    """A key combination such as ctrl+shift+t or win+r: modifiers held in order, the last key pressed, all let go."""
    _keys_allowed()
    for vk, flags in chord_events([vk_for(k) for k in keys]):
        _send(_key(vk=vk, flags=flags))


def type_text(text: str) -> None:
    _keys_allowed()
    for unit, flags in unicode_events(text):
        _send(_key(scan=unit, flags=flags))


def clear_field() -> None:
    press("a", command=True)
    press("delete")


def scroll(lines: int) -> None:
    """Scroll events go to the view under the cursor, so park it over the frontmost window first."""
    _keys_allowed()
    center = frontmost_window_center()
    if center is not None:
        _move((round(center[0]), round(center[1])))
    _send(_mouse(MOUSEEVENTF_WHEEL, wheel_delta(lines)))


# ------------------------------------------------------------------ apps and windows


def _process_name(pid: int) -> str:
    try:
        return psutil.Process(pid).name()
    except psutil.Error:
        return ""


def _window_pid(hwnd: int) -> int:
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    return pid


def frontmost_app() -> str:
    name, _ = frontmost_app_and_pid()
    return name


def frontmost_app_and_pid() -> tuple[str, int]:
    """Name and pid of the foreground window's process."""
    pid = frontmost_pid()
    return display_name(_process_name(pid)), pid


def frontmost_pid() -> int:
    return _window_pid(win32gui.GetForegroundWindow())


def _find_window(app: str) -> int | None:
    """The topmost visible, titled top-level window whose process is `app` (see app_matches)."""
    found: list[int] = []

    def visit(hwnd, _):  # returns None, so the enumeration runs to the end without raising
        titled = not found and win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd)
        if titled and app_matches(app, _process_name(_window_pid(hwnd))):
            found.append(hwnd)

    win32gui.EnumWindows(visit, None)  # top of the z-order first
    return found[0] if found else None


def activate(app: str, timeout: float = 3.0) -> bool:
    """Bring an app to the front and confirm it got there."""
    hwnd = _find_window(app)
    if hwnd is None:
        return False
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    with suppress(Exception):  # Windows may refuse focus theft; the check below reports it
        win32gui.SetForegroundWindow(hwnd)
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if win32gui.GetForegroundWindow() == hwnd:
            return True
        time.sleep(0.1)
    return win32gui.GetForegroundWindow() == hwnd


WINDOW_KEY = "window:"
LAUNCH_KEY = "launch:"
MAX_WINDOWS = 20  # the classifier reads every one as a criterion
LAUNCH_SECONDS = 10.0  # a cold Store app can take this long to show its window
DWMWA_CLOAKED = 14  # a suspended Store app keeps a visible window that is cloaked off screen
# Apps switch_app may start when no window of theirs is open: window title -> what to run.
LAUNCHABLE = {"Calculator": "calc.exe", "Notepad": "notepad.exe", "File Explorer": "explorer.exe", "Paint": "mspaint.exe"}


def _cloaked(hwnd: int) -> bool:
    value = ctypes.c_int(0)
    with suppress(OSError):
        ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(value), ctypes.sizeof(value))
    return bool(value.value)


def _switchable() -> list[tuple[int, str, str]]:
    """The windows of `_all_windows` this process may use: with CU_MONITOR set, those on that monitor or its own."""
    return [w for w in _all_windows() if _may_use(w[0])]


def _all_windows() -> list[tuple[int, str, str]]:
    """(hwnd, app, title) for every window a person could Alt-Tab to, top of the z-order first.

    Matched by window, not by process: a Store app such as Calculator runs inside
    ApplicationFrameHost, so its process name says nothing and only its title names it.
    """
    found: list[tuple[int, str, str]] = []

    def visit(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        if not (title and win32gui.IsWindowVisible(hwnd)) or _cloaked(hwnd):
            return
        if win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) & win32con.WS_EX_TOOLWINDOW:
            return
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        # A minimized window reports a stub rect far off screen; it is still one to switch to.
        if not win32gui.IsIconic(hwnd) and min(right - left, bottom - top) < MIN_WINDOW_SIDE_PT:
            return
        found.append((hwnd, display_name(_process_name(_window_pid(hwnd))), title))

    win32gui.EnumWindows(visit, None)
    return found


def windows() -> list[tuple[str, str]]:
    """(key, description) for switch_app: the open windows behind the frontmost one, then the
    LAUNCHABLE apps that have no window yet. The key is what switch_to takes."""
    front = win32gui.GetForegroundWindow()
    found = [(h, app, title) for h, app, title in _switchable() if h != front][:MAX_WINDOWS]
    titles = {title for _, _, title in found} | {win32gui.GetWindowText(front)}
    return [(f"{WINDOW_KEY}{h}", f"the {title!r} window ({app})") for h, app, title in found] + [
        (f"{LAUNCH_KEY}{name}", f"start {name}") for name in LAUNCHABLE if name not in titles
    ]


def _bring_forward(hwnd: int, timeout: float) -> bool:
    """Put a window in front and confirm it got there.

    Windows ignores SetForegroundWindow from a process that is not in front, which is every
    process an agent runs from. Two standard workarounds, in order: share input state with the
    thread that owns the foreground, then an Alt tap, which counts as this process's own input.
    """
    if not _may_use(hwnd):
        return False
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    _onto_primary(hwnd)
    user32 = ctypes.windll.user32
    for attempt in ("attach", "alt"):
        check_abort()
        mine = win32api.GetCurrentThreadId()
        theirs = user32.GetWindowThreadProcessId(win32gui.GetForegroundWindow(), None)
        attached = attempt == "attach" and theirs != mine and user32.AttachThreadInput(mine, theirs, True)
        if attempt == "alt":
            _send(_key(VK_ALT))
            _send(_key(VK_ALT, flags=KEYEVENTF_KEYUP))
        try:
            with suppress(Exception):  # refused focus is what the check below reports
                win32gui.BringWindowToTop(hwnd)
                win32gui.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(mine, theirs, False)
        end = time.monotonic() + timeout / 2
        while time.monotonic() < end:
            if win32gui.GetForegroundWindow() == hwnd:
                return True
            time.sleep(0.1)
    return False


def _onto_primary(hwnd: int) -> None:
    """Slide a window onto the chosen monitor (the primary unless CU_MONITOR says), the only one captured,
    keeping its size. A window larger than the monitor lands at its top-left corner. With CU_MONITOR set, only a
    window this process started moves; the person's own stay where they are."""
    if _chosen() and hwnd not in _OWNED:
        return
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    m_left, m_top, m_right, m_bottom = monitor_rect()
    x = max(m_left, min(left, m_right - (right - left)))
    y = max(m_top, min(top, m_bottom - (bottom - top)))
    if (x, y) != (left, top):
        win32gui.SetWindowPos(hwnd, 0, x, y, 0, 0, win32con.SWP_NOSIZE | win32con.SWP_NOZORDER)


def switch_to(key: str, timeout: float = 3.0) -> bool:
    """Bring forward the window a `windows()` key names, starting its app first for a launch key,
    and move it onto the primary monitor when it is elsewhere."""
    check_abort()
    if key.startswith(LAUNCH_KEY):
        name = key.removeprefix(LAUNCH_KEY)
        if name not in LAUNCHABLE:
            return False
        before = {h for h, _, _ in _all_windows()}
        subprocess.Popen([LAUNCHABLE[name]])
        end = time.monotonic() + LAUNCH_SECONDS
        hwnd = None
        while hwnd is None and time.monotonic() < end:
            sleep_watching(0.25)
            hwnd = next((h for h, _, title in _all_windows() if title == name and h not in before), None)
        if hwnd is not None:
            _OWNED.add(hwnd)
        return hwnd is not None and _bring_forward(hwnd, timeout)
    if key.startswith(WINDOW_KEY) and key.removeprefix(WINDOW_KEY).isdigit():
        hwnd = int(key.removeprefix(WINDOW_KEY))
        return bool(win32gui.IsWindow(hwnd)) and _bring_forward(hwnd, timeout)
    return False


def open_url(browser: str, url: str) -> bool:
    exe = next((exe for name, exe in BROWSER_EXES.items() if name.lower() == browser.strip().lower()), None)
    if exe and shutil.which(exe):
        subprocess.Popen([exe, url])
    else:
        webbrowser.open(url)
    return activate(browser)


def browser_url(browser: str) -> str | None:
    """The address bar's text, read through UI Automation. No AppleScript-style API exists for this
    on Windows, so it depends on the browser exposing an Edit control named for the address bar,
    which Chrome, Edge, and Firefox do. Chrome and Edge show it without the scheme."""
    hwnd = _find_window(browser)
    if hwnd is None:
        return None
    try:
        window = auto.ControlFromHandle(hwnd)
        edit = window.EditControl(searchDepth=20, RegexName="(?i)address")
        if not edit.Exists(0, 0):
            return None
        return _ui_value(edit) or None
    except Exception:
        return None


def open_path(path: Path, as_text: bool = False) -> None:
    """Show a file to the user in its default app; a .txt already opens in the text editor."""
    os.startfile(path)


def frontmost_window_bounds(pid: int | None = None) -> tuple[float, float, float, float] | None:
    """The foreground window's bounds as x, y, w, h in screen pixels.

    Pass the pid when the caller already has it, to confirm the foreground window still belongs
    to that process rather than one that grabbed focus since.
    """
    hwnd = win32gui.GetForegroundWindow()
    if pid is not None and _window_pid(hwnd) != pid:
        return None
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w, h = float(right - left), float(bottom - top)
    m_left, m_top = _origin()
    if w > MIN_WINDOW_SIDE_PT and h > MIN_WINDOW_SIDE_PT:
        return float(left - m_left), float(top - m_top), w, h
    return None


def frontmost_window_center(pid: int | None = None) -> tuple[float, float] | None:
    """Center of the frontmost app's topmost on-screen window, in screen pixels."""
    bounds = frontmost_window_bounds(pid)
    if bounds is None:
        return None
    x, y, w, h = bounds
    return x + w / 2, y + h / 2


# ------------------------------------------------------------------ capture, OCR, and accessibility


def screenshot() -> Image.Image:
    """The primary monitor, the one macOS calls the main display. The window in front comes onto it first: one that
    opened on another monitor (Task Manager, a dialog) would otherwise be out of every capture, its screen "unchanged"."""
    with suppress(Exception):  # no window in front (the desktop), or one this process may not move
        hwnd = win32gui.GetForegroundWindow()
        if hwnd and not win32gui.IsIconic(hwnd):
            _onto_primary(hwnd)
    return ImageGrab.grab(bbox=monitor_rect(), all_screens=True).convert("RGB")


def display_scale(image: Image.Image) -> float:
    """Per-monitor DPI awareness keeps every coordinate in physical pixels, so a screen point
    already is a capture pixel; unlike macOS, there is no separate points-vs-pixels scale."""
    return 1.0


def recognize_text(image: Image.Image) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """Windows.Media.Ocr lines as text, confidence, and a box in the image's own pixels."""
    return ocr_lines(winocr.recognize_pil_sync(image, OCR_LANGUAGE))


def _pattern(element, pattern_id):
    """A UI Automation pattern, or None when the element does not support it or has died."""
    try:
        return element.GetPattern(pattern_id)
    except Exception:
        return None


def _ui_value(element) -> str | None:
    pattern = _pattern(element, auto.PatternId.ValuePattern)
    try:
        return pattern.Value if pattern is not None else None
    except Exception:
        return None


def _ui_placeholder(element) -> str:
    legacy = _pattern(element, auto.PatternId.LegacyIAccessiblePattern)
    try:
        return (legacy.Description or "") if legacy is not None else ""
    except Exception:
        return ""


def _ui_default_action(element) -> str:
    legacy = _pattern(element, auto.PatternId.LegacyIAccessiblePattern)
    try:
        return (legacy.DefaultAction or "") if legacy is not None else ""
    except Exception:
        return ""


def focused_field() -> Field | None:
    try:
        element = auto.GetFocusedControl()
    except Exception:
        return None
    if element is None:
        return None
    frame = _ui_frame(element)
    x, y, w, h = frame if frame is not None else (0.0, 0.0, 0.0, 0.0)
    return Field(
        role=role_for(getattr(element, "ControlTypeName", "")),
        label=(getattr(element, "Name", "") or "").strip(),
        placeholder=_ui_placeholder(element),
        value=_ui_value(element) or "",
        x=x,
        y=y,
        w=w,
        h=h,
        ref=element,
    )


# ------------------------------------------------------------------ acting on an element

# An element accepts these directly, so a press lands on the control the app declared rather than
# on whatever pixel happens to sit at its center. Every one of them is best effort: the element may
# be dead, the app may refuse, and the bridge raises on both. False means "use synthetic input".


def ax_press(ref) -> bool:
    """Invoke an element, or run its named legacy default action: the same rule as `pressable`."""
    invoke = _pattern(ref, auto.PatternId.InvokePattern)
    try:
        if invoke is not None:
            invoke.Invoke()
            return True
        if _ui_default_action(ref).strip():
            _pattern(ref, auto.PatternId.LegacyIAccessiblePattern).DoDefaultAction()
            return True
    except Exception:
        pass
    return False


def ax_focus(ref) -> bool:
    """Give an element the keyboard focus."""
    try:
        return bool(ref.SetFocus())
    except Exception:
        return False


def ax_set_value(ref, text: str) -> bool:
    """Write an element's value. A read-only or unwilling element reports an error."""
    pattern = _pattern(ref, auto.PatternId.ValuePattern)
    try:
        return pattern is not None and bool(pattern.SetValue(text))
    except Exception:
        return False


def ax_value(ref) -> str | None:
    """An element's value, when it has a textual one."""
    return _ui_value(ref)


# ------------------------------------------------------------------ actionable elements


def _ui_children(element) -> list:
    try:
        return list(element.GetChildren())
    except Exception:
        return []


def _ui_frame(element) -> Frame | None:
    try:
        rect = element.BoundingRectangle
    except Exception:
        return None
    if rect is None:
        return None
    left, top = _origin()
    return float(rect.left - left), float(rect.top - top), float(rect.width()), float(rect.height())


def _ui_attrs(element) -> AxAttrs:
    label = (getattr(element, "Name", "") or "").strip()
    return AxAttrs(role_for(getattr(element, "ControlTypeName", "")), label, _ui_frame(element))


def _ui_actions(element) -> list[str]:
    has_invoke = _pattern(element, auto.PatternId.InvokePattern) is not None
    return [AX_PRESS] if pressable(has_invoke, "" if has_invoke else _ui_default_action(element)) else []


def actionable_elements(pid: int, display_w_pt: float, display_h_pt: float) -> tuple[list[AxNode], list[AxNode], bool]:
    """Labelled controls of the foreground window, in screen pixels, and the pressable off-screen
    ones. Best effort: a window that refuses UI Automation, or belongs to a different process, or
    has none, yields nothing."""
    hwnd = win32gui.GetForegroundWindow()
    if _window_pid(hwnd) != pid:
        return [], [], False
    try:
        root = auto.ControlFromHandle(hwnd)
    except Exception:
        return [], [], False
    return walk_actionable(root, _ui_children, _ui_attrs, _ui_actions, display_w_pt, display_h_pt)
