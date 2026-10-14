"""The stack (`stack.py`) as an OSWorld agent: the planner splits the task, jev carries out each step, code checks it.

Everything `JevAgent` does stays: the worker thread, the queues to `predict`, the VM as the desktop through
`OSWorldDesktop`, the light tree fetch and the run record. Only the worker's body differs: it runs `Stack.run` instead
of one `runner.run`, so each "do" step is a `runner.run` of its own on a subgoal, and "type" / "key" steps go straight to
the VM. Every action list still counts against OSWorld's step budget.

Nobody answers questions in a benchmark. A task the planner calls impossible ends with `FAIL`, which OSWorld scores as
the claim that the task is infeasible (right on its infeasible tasks); every other end, a question among them, ends with
`DONE` and leaves the score to OSWorld's check. Results go to `<task result folder>/stack/`.
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import partial

from ..actions import Context
from ..models import Abort
from ..platform_adapter import using
from ..stack import Stack, StackConfig
from . import tree
from .agent import BROWSER, DONE, JevAgent, _Failed, _Run, _Step, _Watching
from .desktop import OSWorldDesktop

FAIL = "FAIL"
RUN_FOLDER = "stack"
STEP_CAP = 8  # jev actions per "do" step


class StackAgent(JevAgent):
    """`JevAgent` with the stack's loop in its worker. Same arguments, plus `step_cap` for each "do" step."""

    def __init__(self, *args, step_cap: int = STEP_CAP, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step_cap = step_cap

    def predict(self, instruction: str, obs: dict) -> tuple[str, list[str]]:
        if self._run is None:
            self._run = _StackRun(self, instruction, self._stale)
            self._stale = None
        return self._run.predict(obs)


class _StackRun(_Run):
    def __init__(self, agent: StackAgent, instruction: str, after) -> None:
        super().__init__(agent, instruction, after)
        self.cfg = replace(self.cfg, out=agent.out_root / RUN_FOLDER)

    def _work(self, obs: dict) -> None:
        message: _Step | _Failed
        try:
            if self.after is not None:
                self.after.join()
            fetch = None
            if self.agent.controller is not None:
                check = tree.checking()
                fetch = partial(tree.fetch, self.agent.controller(), check=check)
                self.tree = tree.LIGHT_CHECKED if check else tree.LIGHT
            save = self.cfg.out if os.environ.get("JEV_OSWORLD_SAVE_A11Y") == "1" else None
            adapter = self.adapter = OSWorldDesktop(
                obs, self.agent.recognize_text, self._next_obs, fetch_tree=fetch, save_a11y=save
            )
            end = DONE
            with using(adapter):
                stack = Stack(StackConfig(
                    goal=self.cfg.goal, out=self.cfg.out, writer=self.agent.writer, context=self._subgoal_context,
                    desktop_name="Ubuntu, an OSWorld virtual machine", act=True, step_cap=self.agent.step_cap,
                    keys=None, ask=None, run_config={"delay": 0, "read_ahead": True},
                ))
                try:
                    result = stack.run()
                    ended = f"stack ended: {result.result}"
                    if result.result == "cannot":
                        end, ended = FAIL, f"{ended}: {result.reason}"
                    elif result.question:
                        ended = f"{ended}; would ask: {result.question}"
                except Abort as e:  # OSWorld's step limit, or a reset, outside a "do" step
                    stack.log({"event": "end", "result": f"aborted ({e})"})
                    ended = f"stack ended: aborted ({e})"
            final = [*self.final, *adapter.actions]
            response = f"{self.decision}; {ended}" if final else ended
            message = _Step(response, [*final, end], last=True)
        except BaseException as error:
            message = _Failed(error)
        self.to_predict.put(message)

    def _subgoal_context(self, subgoal: str):
        def factory(typesafe, history):
            return Context(goal=subgoal, browser=BROWSER, email=self.agent.email,
                           typesafe=_Watching(typesafe, self._decided), writer=self.agent.writer, history=history,
                           ask=None)
        return factory


__all__ = ["StackAgent"]
