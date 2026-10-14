"""The stack: a planner splits the goal into checked steps, jev carries out each one, code checks the screen after it.

  goal -> planner (the writer's model) -> ordered steps, each with what the screen should show after it
       -> each step: "do"   jev's own loop (`runner.run`) on that one subgoal, with a small step cap
                     "type" text typed in one go (jev pressing keys one at a time loses track of progress the screen
                            does not show, such as a sum pending in Calculator)
                     "key"  one key or a ctrl+<key> shortcut
       -> check: code reads the next capture for the step's expected text (or app). Text that was on screen when the
          run started counts only if the step changed the screen. The plan's last expected text ends the run.

When the screen is wrong: a type/key step whose app is not in front types nothing; a failed check re-plans from what
the screen shows (at most `verify_tries` times), then asks. When the instruction is wrong: the planner answers
"cannot" (impossible, or the screen contradicts it) or "ask" (ambiguous, or a person, amount or file only the user
knows). Nothing is guessed. `ask` puts a question to the user and returns the reply, or None when nobody can answer;
then the run ends with the question.

`run_stack` reads and acts through `platform_adapter.desktop`, so it drives whichever desktop is in use: this machine
(`python -m typesafe_computer_use.stack`, below), or an OSWorld VM (`osworld/stack_agent.py`).

  python -m typesafe_computer_use.stack "<goal>" [--act] [--steps=8] [--tag=name] [--no-ask]
  python -m typesafe_computer_use.stack --resume=<run> --reply="<answer>" [--act]
Without --act: plan, one capture, and a one-step dry run of the first step; nothing is clicked or typed. A run that
needs the user prints "QUESTION: ..." last and exits 3; resume it with the answer. Log: runs/stack/<run>.jsonl.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from . import config, runner
from .config import MAX_OPTIONS
from .perception import OcrCache, capture, perceive
from .platform_adapter import desktop
from .runner import RunConfig
from .writer import _structured, make_writer

WINDOWS_KEYS = frozenset({"return", "escape", "tab", "delete"})  # what the Windows adapter can press

PLAN_SYSTEM = """You plan how to reach a goal on a computer ({desktop}). A fast classifier carries out "do" steps, one
action at a time, with no memory: give it one short concrete subgoal per step whose progress is visible on screen. Its
only actions are: left-click a visible item, open a website in the browser, type into the focused text field, press
Enter, Escape or Back, scroll, and bring an open app to the front. It cannot right-click, double-click, drag, hover or
use shortcuts: plan those as "key" steps (a shortcut) or reach the same thing through visible menus and buttons.
Text or a key sequence whose progress the screen does not show (a sum in Calculator, a form value, a cell's formula)
goes in a "type" step, typed in one go into the app in front. A "key" step presses {keys}.

"status": "ok" with a plan; "cannot" when the goal is impossible or contradicts what the screen shows (say why in
"note"); "ask" when the goal is ambiguous or misses information only the user has (put the one question in "note").
If "front_window_not_responding" is set, that window is hung: close it and open the app again before anything else.
{asking}
Steps that send, buy, delete or post something the goal did not ask for are never planned.

"plan" (status ok): one step per line, written as  kind | what | expect
  kind    do, type or key
  what    the subgoal (do), the exact characters (type), or the key (key)
  expect  exact words that will be visible on screen after the step, copied as the app shows them (a button's label,
          a menu item, a title, a number): code searches the screen's text for them, so never a description such as
          "menu appears". Or app:<Name> for the app that should be in front (named as "front_app" names apps), or
          blank if no words can be named for sure
A type or key step goes to the app the last app:<Name> expect named{typing_default}.
"complete": true when these steps finish the whole goal (then the last step's expect proves it is met); false when they
are the next stage only, and you will plan the rest after seeing the screen again. A long goal is best planned in
stages of a few steps."""

ASK_USER = "Never guess a person, account, amount, address or file the goal does not name: answer \"ask\"."
NO_USER = """Nobody can answer questions here: never answer "ask". When a detail is missing, take the most reasonable
reading of the goal and of the files and apps on screen, and say which in "note"."""

CONTINUE_NOTE = """The steps in "done_and_checked" are done. Plan the next stage from the screen as it is now."""

REPLAN_NOTE = """The screen is not what the plan expected ("failed_step" and "why" say how). Steps in "done_and_checked"
stay done. From the screen as it is now (its text is given): plan only what remains, or answer "cannot" if the screen
shows the goal cannot be done (say what you see), or "ask" if only the user can resolve it. Never plan the failed
step again as it was: it did not work, so reach its result another way (another control, a menu, a shortcut)."""

PLAN_REPLY = {"status": {"type": "string"}, "note": {"type": "string"}, "plan": {"type": "string"},
              "complete": {"type": "boolean"}}
MAX_PLANS = 20  # planner calls per run: past this, the run is stuck however its checks went

# What the planner may call an app, against the names the desktop reports ("front_app"). 2026-10-13 dev: the planner
# wrote app:Calc and app:Files for soffice and nautilus, and the check failed on the name alone.
APP_ALIASES = {
    "soffice": ("libreoffice", "calc", "writer", "impress", "office"),
    "nautilus": ("files", "file manager", "filemanager"),
    "code": ("vs code", "vscode", "visual studio code"),
    "gnome-terminal": ("terminal",),
    "evince": ("document viewer", "pdf viewer"),
    "eog": ("image viewer",),
    "gimp": ("gnu image",),
    "thunderbird": ("mail", "email"),
    "vlc": ("media player",),
}


@dataclass
class StackConfig:
    """One run. `context(goal)` returns the `ctx_factory` jev's `runner.run` takes, for a subgoal. `keys` is the set of
    keys a key step may press, None for any key and ctrl+<key>. `window()` returns the front window's (title, hung)."""

    goal: str
    out: Path
    writer: object
    context: Callable[[str], Callable]
    desktop_name: str = "Windows"
    act: bool = False
    step_cap: int = 8
    verify_tries: int = 2
    asks: int = 2
    keys: frozenset[str] | None = WINDOWS_KEYS
    ask: Callable[[str], str | None] | None = None
    fallback_steps: int = 0  # with no `ask`: when stuck, jev alone runs the whole goal for up to this many actions
    max_plans: int = MAX_PLANS
    window: Callable[[], tuple[str, bool]] = lambda: ("", False)
    run_config: dict = field(default_factory=dict)  # extra RunConfig fields for "do" steps (OSWorld: delay, read_ahead)
    resume: dict | None = None  # {"goal", "question", "done", "reply"}: the user answered the question a run stopped on


@dataclass
class StackResult:
    result: str  # done, dry run, cannot, needs user, failed check
    question: str | None = None
    steps_run: int = 0
    replans: int = 0
    questions: int = 0
    reason: str = ""  # the planner's note for cannot


class Stack:
    def __init__(self, cfg: StackConfig) -> None:
        self.cfg = cfg
        self.log_path = cfg.out / "stack.jsonl"
        self.plans = 0

    # ----- log, look, check --------------------------------------------------------------------------------------

    def log(self, rec: dict) -> None:
        self.cfg.out.mkdir(parents=True, exist_ok=True)
        rec = {"t": round(time.time(), 1), **rec}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def look(self):
        """One capture: the app in front and every text the screen shows (goal echoes kept: the check needs them).

        Items alone are not enough: perception leaves out what OCR reads inside the focused field, whose text is its
        value, and in a terminal the focused field is the whole terminal (2026-10-14 dev: every typed command read as
        "no change", 4 texts, the window's buttons). So the check also reads OCR's raw lines, kept by a fresh cache
        (one OCR pass, as before), and the field's value."""
        screen = capture(browser=config.browser())
        cache = OcrCache()
        texts = [it.text for it in perceive(screen, MAX_OPTIONS, "", cache=cache)]
        seen = set(texts)
        texts += [t for t, _, _ in cache.lines if t not in seen]
        if screen.field is not None and screen.field.value:
            texts.append(screen.field.value)
        return screen, texts

    def front_is(self, app, screen) -> bool:
        norm = lambda s: (s or "").lower().replace(" ", "").removesuffix(".exe")  # noqa: E731
        want, front, title = norm(app), norm(screen.app), norm(self.cfg.window()[0])
        if not want:
            return False
        if want in front or want in title or (front and front in want):
            return True
        return any(name in front and any(want in norm(a) or norm(a) in want for a in aliases)
                   for name, aliases in APP_ALIASES.items())

    def shows(self, expect, screen, texts) -> bool:
        """The check: code decides. Digits compare without separators (Calculator shows 86,400)."""
        if not expect:
            return True
        if expect.lower().startswith("app:"):
            return self.front_is(expect[4:].strip(), screen)
        norm = lambda s: re.sub(r"[\s,  ]", "", s).lower()  # noqa: E731
        want = norm(expect)
        return any(want in norm(t) for t in texts)

    # ----- plan --------------------------------------------------------------------------------------------------

    def key_ok(self, key: str) -> bool:
        key = key.lower()
        if self.cfg.keys is None:
            return bool(re.fullmatch(r"((ctrl|alt|shift)\+)*([a-z0-9]+|[^\s+])", key))
        return key in self.cfg.keys

    def parse_plan(self, text: str, front: str | None = None) -> list[dict]:
        steps = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.strip().strip("-*").split("|")]
            if len(parts) < 2 or parts[0].lower() not in ("do", "type", "key"):
                continue
            kind, what, expect = parts[0].lower(), parts[1], (parts[2] if len(parts) > 2 else "")
            if kind == "key" and not self.key_ok(what):
                continue
            steps.append(dict(kind=kind, what=what, expect=expect))
        # type/key steps go to the app the last app:<Name> expect named. In a benchmark VM, else the app in front at
        # planning; on a real desktop no default: typing into whatever is in front could land in the user's file
        # (2026-10-14 dry run: type "Calculator" planned with an unsaved Notepad file in front).
        app = front if self.no_user else None
        for s in steps:
            if s["kind"] != "do":
                s["app"] = app
            if s["expect"].lower().startswith("app:"):
                app = s["expect"][4:].strip()
        return steps

    def structured(self, system: str, packet: dict, image=None) -> dict:
        """One JSON reply from the writer's answering model, which sees the screenshot (Claude Code: sonnet). The
        2026-10-13 OSWorld check planned steps for a window that was not open: from the screen's text alone the
        planner could not tell what was showing."""
        return _structured(self.cfg.writer, system, packet, PLAN_REPLY, max_tokens=2000, answering=True, image=image)

    @property
    def no_user(self) -> bool:
        return self.cfg.ask is None and self.cfg.fallback_steps > 0  # a benchmark: nobody answers

    def plan(self, goal, screen, texts, done=(), failed=None, why=""):
        """-> (status, note, steps, complete): ok + steps, or cannot / ask with the reason or question in note.
        `failed` None with steps in `done` asks for the next stage of a plan that ran out (complete false)."""
        title, hung = self.cfg.window()
        packet = {"goal": goal, "front_app": screen.app, "front_window_title": title, "screen_text": texts[:120]}
        if hung:
            packet["front_window_not_responding"] = True
        keys = ("any key (enter, esc, tab, backspace, f2, up, ...) or a ctrl+<key> shortcut" if self.cfg.keys is None
                else "one of: " + ", ".join(sorted(self.cfg.keys)))
        system = PLAN_SYSTEM.format(desktop=self.cfg.desktop_name, keys=keys, asking=NO_USER if self.no_user else ASK_USER,
                                    typing_default=", or else to the app in front now" if self.no_user else
                                    ": name it in an earlier step, or the step is not typed")
        if failed:
            system += "\n\n" + REPLAN_NOTE
            packet |= {"done_and_checked": list(done), "failed_step": failed, "why": why}
        elif done:
            system += "\n\n" + CONTINUE_NOTE
            packet |= {"done_and_checked": list(done)}
        t0 = time.time()
        reply = self.structured(system, packet, screen.image)
        status = reply["status"].strip().lower()
        status = status if status in ("ok", "cannot", "ask") else "ok"
        steps = self.parse_plan(reply["plan"], screen.app) if status == "ok" else []
        complete = bool(reply.get("complete"))
        if status == "ok" and not steps:
            status, reply["note"] = "ask", "I could not make a plan for this goal. Can you say it another way?"
        self.plans += 1
        self.log({"event": "plan", "replan": bool(failed), "next_stage": bool(done and not failed), "status": status,
                  "complete": complete, "note": reply["note"], "seconds": round(time.time() - t0, 1), "steps": steps,
                  "raw": reply["plan"][-600:]})
        return status, reply["note"], steps, complete

    # ----- act ---------------------------------------------------------------------------------------------------

    def do_step(self, step: dict, i: int) -> str:
        """Carry out one step. "do" is a jev run on the subgoal alone; its outcome is returned for the log."""
        if step["kind"] == "do":
            cfg = RunConfig(goal=step["what"], out=self.cfg.out / f"step-{i:02d}", act=self.cfg.act,
                            steps=self.cfg.step_cap, handoffs=0, **self.cfg.run_config)
            with no_hand_off():
                state = runner.run(cfg, self.cfg.context(step["what"]))
            return state.outcome
        if not self.cfg.act:
            return "dry run"
        desktop.check_abort()
        if step["kind"] == "type":
            desktop.type_text(step["what"])
        else:
            keys = step["what"].lower().split("+")
            if len(keys) > 2 or (len(keys) == 2 and keys[0] != "ctrl"):
                desktop.hotkey(*keys)  # the OSWorld adapter; parse_plan allows combinations only with keys=None
            else:
                desktop.press(keys[-1], command=len(keys) == 2)
        desktop.sleep_watching(0.8)
        return "typed" if step["kind"] == "type" else "pressed"

    def ask_user(self, question, goal, done):
        self.log({"event": "ask", "question": question, "goal": goal, "done": list(done)})
        return self.cfg.ask(question) if self.cfg.ask else None

    # ----- the loop ----------------------------------------------------------------------------------------------

    def run(self) -> StackResult:
        cfg = self.cfg
        screen, texts = self.look()
        start = (screen, texts)  # what was already there: expected text on it proves nothing (the "Recycle Bin" icon)
        done, replans, asks, i, result, question, reason = [], 0, 0, 0, None, None, ""
        if cfg.resume:  # the user answered the question the run stopped on: carry its goal and checked steps
            r = cfg.resume
            goal, done = f"{r['goal']}\n(User, when asked {r['question']!r}: {r['reply']})", list(r["done"])
            self.log({"event": "resume", "reply": r["reply"], "act": cfg.act})
            status, note, steps, complete = self.plan(goal, screen, texts, done, failed={"question": r["question"]},
                                                      why="the user answered")
        else:
            goal = cfg.goal
            self.log({"event": "start", "goal": goal, "act": cfg.act, "step_cap": cfg.step_cap})
            status, note, steps, complete = self.plan(goal, screen, texts)
        final = steps[-1]["expect"] if steps and complete else ""  # the plan's proof that the goal is met
        while result is None:
            if status in ("cannot", "ask"):  # instruction wrong: the user decides, the run never guesses
                if asks >= cfg.asks:
                    result, reason = ("cannot", note) if status == "cannot" else ("needs user", "")
                    break
                asks += 1
                q = note if status == "ask" else f"I can't do this as asked: {note} What should I do instead?"
                reply = self.ask_user(q, goal, done)
                if reply is None and cfg.fallback_steps and status == "ask":
                    # Nobody to answer (a benchmark): jev alone takes the whole goal for the steps left, as it would
                    # have without the stack (2026-10-13: a stuck stack stopped after 5 of OSWorld's 50 steps).
                    self.log({"event": "fallback", "why": q})
                    rc = RunConfig(goal=cfg.goal, out=cfg.out / "fallback", act=cfg.act, steps=cfg.fallback_steps,
                                   **cfg.run_config)
                    result = f"fallback: {runner.run(rc, cfg.context(cfg.goal)).outcome}"
                    break
                if reply is None:
                    result, question = ("cannot" if status == "cannot" else "needs user"), q
                    reason = note if status == "cannot" else ""
                    break
                goal = f"{goal}\n(User, when asked {q!r}: {reply})"
                replans = 0  # the user's answer is new information: the re-plan budget starts again
                screen, texts = self.look()
                status, note, steps, complete = self.plan(goal, screen, texts, done, failed={"question": q},
                                                          why="the user answered")
                final = steps[-1]["expect"] if steps and complete else ""
                continue
            if not steps:
                if complete or not cfg.act:
                    result = "done"
                    break
                if self.plans >= cfg.max_plans:
                    status, note = "ask", f"I'm stuck: {self.plans} plans and the goal is not met. Try another way, or stop?"
                    continue
                status, note, steps, complete = self.plan(goal, screen, texts, done)  # the next stage
                final = steps[-1]["expect"] if steps and complete else ""
                continue
            step = steps.pop(0)
            i += 1
            if step["kind"] != "do" and cfg.act and not self.front_is(step.get("app"), screen):  # never type blind
                why = (f"{step['app'] or 'no app was named'} should be in front for this {step['kind']} step, "
                       f"but {screen.app!r} is")
                outcome, ok, changed = "not run", False, False
            else:
                before = (screen.app, texts)
                outcome = self.do_step(step, i)
                screen, texts = self.look()
                # Text on screen when the run started proves nothing unless the step changed the screen. A step that
                # only reads (2026-10-13: "read the display", already showing 160) changes nothing and still passes.
                stale = self.shows(step["expect"], *start) and (screen.app, texts) == before
                ok = self.shows(step["expect"], screen, texts) and not stale
                # Soft pass: jev finished the subgoal and the screen changed, but the planner's words are not in what
                # OCR and the tree read (2026-10-13: "New folder" after jev opened the dialog). The next plan sees the
                # screenshot, so a wrong soft pass is caught there rather than spending a re-plan here.
                changed = (screen.app, texts) != before
                soft = not ok and step["kind"] == "do" and outcome == "done" and changed
                ok = ok or soft
                why = ("" if ok else f"the screen did not change; jev said {outcome!r}" if stale or not changed
                       else f"expected {step['expect']!r} on screen after the step; jev said {outcome!r}")
            title, hung = cfg.window()
            if hung and not ok:
                why += f"; the window in front ({title!r}) is not responding"
            if ok and why:
                why = f"soft pass: {why}"
            self.log({"event": "step", "i": i, **step, "outcome": outcome, "check": ok, "why": why, "app": screen.app,
                      "title": title, "hung": hung, "screen_text": texts[:40]})
            uncertain = not ok and outcome != "not run" and changed
            if not cfg.act:
                result = "dry run"  # nothing acted: later checks would only test the unchanged screen
            elif ok:
                replans = 0  # the budget is for failures in a row: a long task re-plans many times between them
                done.append(f"{step['kind']} | {step['what']} | {step['expect']}")
                if final and self.shows(final, screen, texts) and not self.shows(final, *start):
                    result = "done"  # the goal's proof is on screen: steps left would only re-check it
            elif self.plans < cfg.max_plans and (replans < cfg.verify_tries or uncertain):
                # Screen wrong: re-plan from what it shows (the planner sees the screenshot; it may also say cannot /
                # ask). A step that changed the screen but missed its words is uncertain, not a failure: OCR misses
                # words (2026-10-13 dev: VS Code's open "snake.py" tab), and only no change at all counts toward the
                # failures in a row.
                replans += 0 if uncertain else 1
                status, note, steps, complete = self.plan(goal, screen, texts, done, failed=step, why=why)
                final = steps[-1]["expect"] if steps and complete else ""
            else:
                status, note = "ask", (f"I'm stuck: {why}. The screen shows {screen.app!r}. "
                                       "Should I try another way, or stop?")
        self.log({"event": "end", "result": result, "steps_run": i, "replans": replans, "questions": asks})
        return StackResult(result, question, i, replans, asks, reason)


@contextmanager
def no_hand_off():
    """A "do" step is one subgoal, and the check after it reads the screen. jev's hand-off to the writer when it stops
    (a review of the screen, 11-12 s per step on the 2026-10-13 live runs) would only repeat that check."""
    saved = runner.hand_off
    runner.hand_off = lambda cfg, ctx, state, step, log: False
    try:
        yield
    finally:
        runner.hand_off = saved


# ----- this machine --------------------------------------------------------------------------------------------------


def windows_front() -> tuple[str, bool]:
    """(title, not responding) of the window in front. Store apps such as Calculator run under ApplicationFrameHost.exe,
    so the process name alone does not say which app it is; a hung window is a wrong screen no click fixes (2026-10-13:
    Calculator stuck on its splash). Windows only; elsewhere ("", False)."""
    try:
        import ctypes

        import win32gui

        h = win32gui.GetForegroundWindow()
        return win32gui.GetWindowText(h), bool(ctypes.windll.user32.IsHungAppWindow(h))
    except Exception:
        return "", False


def main(argv: list[str] | None = None) -> None:
    from .actions import Context

    argv = sys.argv[1:] if argv is None else argv
    args = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in argv if a.startswith("--"))
    goal = " ".join(a for a in argv if not a.startswith("--"))
    resume_tag = args.get("resume")
    if not (goal or resume_tag) or (resume_tag and not args.get("reply")):
        sys.exit(__doc__)
    config.load_dotenv(Path.cwd() / ".env")
    writer = make_writer()
    if writer is None:
        sys.exit("no writer: set CLICKER_WRITER_API (claude-code / codex / opencode) or a key in .env")
    tag = resume_tag or args.get("tag", time.strftime("%Y%m%d-%H%M%S"))
    out = Path("runs/stack") / tag
    resume = None
    if resume_tag:
        last = [r for r in map(json.loads, (out / "stack.jsonl").read_text(encoding="utf-8").splitlines())
                if r["event"] == "ask"][-1]
        resume = {"goal": last["goal"], "question": last["question"], "done": last["done"], "reply": args["reply"]}

    def context(subgoal):
        def factory(typesafe, history):
            return Context(goal=subgoal, browser=config.browser(), email=config.email(), typesafe=typesafe,
                           writer=writer, history=history)
        return factory

    def ask(question):
        if "no-ask" in args or not sys.stdin.isatty():  # an agent's stdin may never answer
            return None
        try:
            return input(f"\a{question}\n  > ").strip() or None
        except EOFError:
            return None

    print(f"RUN: {tag}", flush=True)
    windows = sys.platform == "win32"
    res = Stack(StackConfig(goal=goal, out=out, writer=writer, context=context,
                            desktop_name="Windows" if windows else sys.platform, act="act" in args,
                            step_cap=int(args.get("steps", 8)), keys=WINDOWS_KEYS if windows else None, ask=ask,
                            window=windows_front, resume=resume)).run()
    if res.question:
        print(f"QUESTION: {res.question}", flush=True)
        sys.exit(3)
    print(f"RESULT: {res.result}", flush=True)
    sys.exit(0 if res.result in ("done", "dry run") else 1)


if __name__ == "__main__":
    main()
