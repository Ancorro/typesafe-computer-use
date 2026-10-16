from types import SimpleNamespace

from typesafe_computer_use.stack import GNOME_KEYS, Stack, StackConfig


def stack(tmp_path, **kw):
    cfg = StackConfig(goal="g", out=tmp_path, writer=None, context=lambda goal: None, **kw)
    return Stack(cfg)


def test_a_benchmark_step_with_no_app_goes_to_whatever_is_in_front(tmp_path):
    # The GNOME overview after "key super" reports no app; a real desktop still never types blind.
    screen = SimpleNamespace(app="")
    assert stack(tmp_path, ask=None, fallback_steps=50).front_is(None, screen)
    assert not stack(tmp_path).front_is(None, screen)


def test_the_replan_sees_every_failed_step_and_the_desktops_keyboard_routes(tmp_path, monkeypatch):
    s = stack(tmp_path, ask=None, fallback_steps=50, keyboard_routes=GNOME_KEYS)
    seen = []
    monkeypatch.setattr(s, "structured", lambda system, packet, image=None: seen.append((system, packet)) or
                        {"status": "ok", "note": "", "plan": "key | super | ", "complete": False})
    s.failed_here = [{"kind": "do", "what": "Click the Files icon in the dock", "why": "the screen did not change"},
                     {"kind": "do", "what": "Click Files in the dock", "why": "the screen did not change"}]
    screen = SimpleNamespace(app="gnome-shell", image=None)
    s.plan("g", screen, [], done=["do | x | y"], failed={"kind": "do", "what": "Click Files in the dock"}, why="no change")
    system, packet = seen[-1]
    assert "key super" in system and "never plan another \"do\" step with the same aim" in system
    assert [f["what"] for f in packet["failed_steps"]] == ["Click the Files icon in the dock", "Click Files in the dock"]


def test_no_keyboard_routes_unless_the_desktop_gives_them(tmp_path, monkeypatch):
    s = stack(tmp_path)
    seen = []
    monkeypatch.setattr(s, "structured", lambda system, packet, image=None: seen.append(system) or
                        {"status": "ok", "note": "", "plan": "do | open Calculator | app:Calculator", "complete": True})
    s.plan("g", SimpleNamespace(app="explorer.exe", image=None), [])
    assert "Keyboard routes" not in seen[-1]
