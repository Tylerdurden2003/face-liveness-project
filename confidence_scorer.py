"""
confidence_scorer.py
Session-level weighted confidence scorer with:
  - Hard countdown timer (SPOOF if time runs out)
  - Texture as a hard gate
  - Motion coherence gate with cross-axis stability check
  - Spoof reason tracking
  - Coherence and cross-axis lock immediately without waiting for all_done
  - Supports periodic re-verification mode
"""

import time
import numpy as np

WEIGHTS = {
    "challenge": 0.60,
    "texture":   0.30,
    "motion":    0.05,
    "pupil":     0.05,
}

LIVE_THRESHOLD = 0.48
HISTORY_LEN = 25
SESSION_TIMEOUT = 20.0
TEXTURE_HARD_MIN = 0.36
COHERENCE_SPOOF_THRESHOLD = 0.82

HEAD_MOVEMENT_ACTIONS = {"LOOK_LEFT", "LOOK_RIGHT", "LOOK_UP", "LOOK_DOWN"}

CROSS_AXIS_PITCH_MAX = 0.06
CROSS_AXIS_YAW_MAX = 0.07


class ConfidenceScorer:
    def __init__(self, weights=None, live_threshold=LIVE_THRESHOLD,
                 history_len=HISTORY_LEN, timeout=SESSION_TIMEOUT):
        self.weights = weights or WEIGHTS
        self.live_threshold = live_threshold
        self.timeout = timeout
        self._maxlen = history_len
        self._history = []
        self.confidence = 0.0
        self.verdict = "CHECKING"
        self.spoof_reason = None
        self._locked = False
        self._component_scores = {}
        self._texture_fail_count = 0
        self._coherence_fail_count = 0
        self._cross_axis_fail_count = 0
        self._start_time = time.time()
        self.time_remaining = timeout

    def update(self, challenge_result, texture_result, motion_result,
               pupil_result, head_result=None):
        if self._locked:
            self.time_remaining = max(
                0.0, self.timeout - (time.time() - self._start_time))
            return self._result()

        elapsed = time.time() - self._start_time
        self.time_remaining = max(0.0, self.timeout - elapsed)

        # ── Hard timeout ──────────────────────────────────────────────
        if elapsed >= self.timeout:
            self.verdict = "SPOOF"
            self.spoof_reason = "Session timed out — no response"
            self._locked = True
            return self._result()

        ch_score = 1.0 if challenge_result.get("all_done") else (
            challenge_result.get("steps_done", 0) /
            max(challenge_result.get("total_steps", 1), 1)
        )
        tx_score = float(texture_result.get("texture_score", 0.0))
        mo_score = float(motion_result.get("score",          0.0))
        pu_score = float(pupil_result.get("pupil_score",     0.0))
        coherence_val = float(motion_result.get("coherence",      0.0))

        self._component_scores = {
            "challenge": round(ch_score,      3),
            "texture":   round(tx_score,      3),
            "motion":    round(mo_score,       3),
            "pupil":     round(pu_score,       3),
            "coherence": round(coherence_val,  3),
        }

        frame_conf = (
            self.weights["challenge"] * ch_score +
            self.weights["texture"] * tx_score +
            self.weights["motion"] * mo_score +
            self.weights["pupil"] * pu_score
        )

        self._history.append(frame_conf)
        if len(self._history) > self._maxlen:
            self._history.pop(0)

        self.confidence = float(np.mean(self._history))

        # ── Texture hard gate ─────────────────────────────────────────
        if tx_score < TEXTURE_HARD_MIN and elapsed > 2.0:
            self._texture_fail_count += 1
        else:
            self._texture_fail_count = max(0, self._texture_fail_count - 1)
        texture_spoof = self._texture_fail_count > 20 and elapsed > 2.5

        # ── Coherence gate ────────────────────────────────────────────
        current_action = challenge_result.get("current_action", "")
        all_done = challenge_result.get("all_done", False)
        head_challenge_active = (current_action in HEAD_MOVEMENT_ACTIONS
                                 and not all_done)

        cross_axis_fail = False
        if head_challenge_active and head_result is not None and elapsed > 2.0:
            pitch_delta = abs(float(head_result.get("pitch_delta", 0.0)))
            yaw_val = abs(float(head_result.get("yaw",         0.0)))
            if current_action in ("LOOK_LEFT", "LOOK_RIGHT"):
                if pitch_delta > CROSS_AXIS_PITCH_MAX:
                    cross_axis_fail = True
            elif current_action in ("LOOK_UP", "LOOK_DOWN"):
                if yaw_val > CROSS_AXIS_YAW_MAX:
                    cross_axis_fail = True

        if cross_axis_fail:
            self._cross_axis_fail_count += 1
        else:
            self._cross_axis_fail_count = max(
                0, self._cross_axis_fail_count - 1)
        cross_axis_spoof = self._cross_axis_fail_count > 12 and elapsed > 2.5

        if (coherence_val > COHERENCE_SPOOF_THRESHOLD
                and elapsed > 2.0
                and not head_challenge_active):
            self._coherence_fail_count += 1
        else:
            self._coherence_fail_count = max(
                0, self._coherence_fail_count - 2)
        coherence_spoof = self._coherence_fail_count > 15 and elapsed > 2.5

        # ── Glare check ───────────────────────────────────────────────
        glare_score = float(texture_result.get("glare_score", 1.0))
        glare_spoof = glare_score < 0.35 and elapsed > 3.0

        # ── Build spoof reason ────────────────────────────────────────
        def _build_reason():
            if texture_spoof and glare_spoof:
                return "Printed photo or screen — texture + glare failed"
            if glare_spoof:
                return "Phone screen detected — screen glare signature"
            if texture_spoof:
                return "Printed photo detected — unnatural skin texture"
            if cross_axis_spoof:
                return "Phone tilt detected — off-axis movement during challenge"
            if coherence_spoof:
                return "Rigid object detected — all face points moving together"
            return None

        # ── Verdict ───────────────────────────────────────────────────
        if coherence_spoof or cross_axis_spoof:
            self.verdict = "SPOOF"
            self.spoof_reason = _build_reason()
            self._locked = True

        elif (texture_spoof or glare_spoof) and all_done:
            self.verdict = "SPOOF"
            self.spoof_reason = _build_reason()
            self._locked = True

        elif all_done and self.confidence >= self.live_threshold \
                and not texture_spoof and not glare_spoof:
            self.verdict = "LIVE"
            self.spoof_reason = None
            self._locked = True

        else:
            self.verdict = "CHECKING"

        return self._result()

    def _result(self):
        return {
            "verdict":          self.verdict,
            "confidence":       round(self.confidence, 4),
            "live_threshold":   self.live_threshold,
            "component_scores": self._component_scores,
            "progress":         min(self.confidence / self.live_threshold, 1.0),
            "time_remaining":   round(self.time_remaining, 1),
            "timeout":          self.timeout,
            "spoof_reason":     self.spoof_reason,
        }

    def reset(self):
        self._history.clear()
        self.confidence = 0.0
        self.verdict = "CHECKING"
        self.spoof_reason = None
        self._locked = False
        self._component_scores = {}
        self._texture_fail_count = 0
        self._coherence_fail_count = 0
        self._cross_axis_fail_count = 0
        self._start_time = time.time()
        self.time_remaining = self.timeout

    @property
    def is_locked(self):
        return self._locked
