"""
challenge_engine.py — UPGRADE #4
Randomised challenge-response: random ordered sequence of actions.
Pre-recorded replays cannot satisfy unseen challenge sequences.
"""
import random
import time
from dataclasses import dataclass


class Action:
    BLINK = "BLINK"
    LOOK_LEFT = "LOOK_LEFT"
    LOOK_RIGHT = "LOOK_RIGHT"
    LOOK_UP = "LOOK_UP"
    LOOK_DOWN = "LOOK_DOWN"


ACTION_POOL = [
    Action.BLINK, Action.LOOK_LEFT, Action.LOOK_RIGHT,
    Action.LOOK_UP, Action.LOOK_DOWN
]
ACTION_LABELS = {
    Action.BLINK:       "Blink",
    Action.LOOK_LEFT:   "Look LEFT",
    Action.LOOK_RIGHT:  "Look RIGHT",
    Action.LOOK_UP:     "Look UP",
    Action.LOOK_DOWN:   "Look DOWN",
    "DONE":             "Done",
}
DIRECTION_TO_ACTION = {
    "LEFT":  Action.LOOK_LEFT,
    "RIGHT": Action.LOOK_RIGHT,
    "UP":    Action.LOOK_UP,
    "DOWN":  Action.LOOK_DOWN,
}
HOLD_DURATION = 0.4


@dataclass
class ChallengeStep:
    action: str
    completed: bool = False


class ChallengeEngine:
    def __init__(self, num_steps=2, seed=None):
        self.num_steps = num_steps
        self._rng = random.Random(seed)
        self.steps = []
        self._hold_start = {}
        self._blink_base = 0
        self.generate()

    def generate(self):
        actions = []
        last = None
        for _ in range(self.num_steps):
            cands = [a for a in ACTION_POOL if a != last]
            chosen = self._rng.choice(cands)
            actions.append(chosen)
            last = chosen
        self.steps = [ChallengeStep(a) for a in actions]
        self._hold_start = {}
        self._blink_base = 0

    def update(self, blink_result, head_result):
        if self.all_done:
            return self._result()

        idx = self.current_step_idx
        step = self.steps[idx]
        now = time.time()

        if step.action == Action.BLINK:
            if blink_result.get("blink_count", 0) > self._blink_base:
                step.completed = True
                self._blink_base = blink_result.get("blink_count", 0)
        else:
            direction = head_result.get("direction", "CENTER")
            required = DIRECTION_TO_ACTION.get(direction)
            if required == step.action:
                if step.action not in self._hold_start:
                    self._hold_start[step.action] = now
                elif now - self._hold_start[step.action] >= HOLD_DURATION:
                    step.completed = True
                    self._hold_start.pop(step.action, None)
            else:
                self._hold_start.pop(step.action, None)

        return self._result()

    @property
    def current_step_idx(self):
        for i, s in enumerate(self.steps):
            if not s.completed:
                return i
        # all done — return last valid index, not out-of-bounds
        return len(self.steps) - 1

    @property
    def all_done(self):
        return all(s.completed for s in self.steps)

    def _result(self):
        idx = self.current_step_idx
        all_done = self.all_done
        # show "Done" label only when all steps complete
        cur_action = "DONE" if all_done else self.steps[idx].action
        steps_done = sum(1 for s in self.steps if s.completed)
        return {
            "current_step_idx": idx,
            "current_action":   cur_action,
            "current_label":    ACTION_LABELS.get(cur_action, "Done"),
            "steps_done":       steps_done,
            "total_steps":      len(self.steps),
            "all_done":         all_done,
            "steps": [
                {
                    "action":    s.action,
                    "label":     ACTION_LABELS[s.action],
                    "completed": s.completed,
                }
                for s in self.steps
            ],
        }

    def reset(self):
        self.generate()
