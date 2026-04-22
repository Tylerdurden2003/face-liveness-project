"""
head_pose_detector.py
Reliable head direction detection using facial geometry ratios.

For UP/DOWN: uses the ratio of nose-to-forehead vs nose-to-chin distance.
  - Look DOWN → chin moves away from camera → nose-to-chin increases
  - Look UP   → forehead moves away → nose-to-forehead increases

For LEFT/RIGHT: nose horizontal offset from ear midpoint, normalised by face width.
"""
import numpy as np

NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_EAR = 234
RIGHT_EAR = 454
NOSE_BASE = 168
UPPER_LIP = 0

YAW_THR = 0.08    # lowered from 0.10 — easier to trigger left/right
PITCH_THR = 0.055   # lowered from 0.08 — easier to trigger up/down
HYSTERESIS = 2       # lowered from 3 — faster response


class HeadPoseDetector:
    def __init__(self, yaw_thr=YAW_THR, pitch_thr=PITCH_THR):
        self.yaw_thr = yaw_thr
        self.pitch_thr = pitch_thr
        self.movements_detected = set()
        self.current_direction = "CENTER"
        self.yaw = 0.0
        self.pitch = 0.0
        self.pitch_delta = 0.0   # exposed for coherence cross-check
        self._pending_dir = "CENTER"
        self._pending_count = 0
        self._baseline_pitch = None
        self._baseline_frames = 0
        self._pitch_samples = []

    def update(self, face_landmarks, w, h):
        lm = face_landmarks.landmark

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        nose = pt(NOSE_TIP)
        chin = pt(CHIN)
        forehead = pt(FOREHEAD)
        left_ear = pt(LEFT_EAR)
        right_ear = pt(RIGHT_EAR)

        # ── Yaw ───────────────────────────────────────────────────────
        ear_mid_x = (left_ear[0] + right_ear[0]) / 2.0
        face_w = max(abs(right_ear[0] - left_ear[0]), 1.0)
        yaw_offset = (nose[0] - ear_mid_x) / face_w
        self.yaw = float(yaw_offset)

        # ── Pitch ─────────────────────────────────────────────────────
        nose_to_forehead = abs(nose[1] - forehead[1])
        nose_to_chin = abs(nose[1] - chin[1])
        total_h = nose_to_forehead + nose_to_chin
        if total_h < 1.0:
            return self._result()

        pitch_ratio = nose_to_forehead / total_h
        self.pitch = float(pitch_ratio)

        # ── Baseline — first 10 frames (reduced from 15) ──────────────
        if self._baseline_pitch is None or self._baseline_frames < 10:
            self._baseline_frames += 1
            if self._baseline_pitch is None:
                self._baseline_pitch = pitch_ratio
            else:
                self._baseline_pitch = (0.85 * self._baseline_pitch
                                        + 0.15 * pitch_ratio)
            return self._result()

        baseline = self._baseline_pitch
        pitch_delta = pitch_ratio - baseline
        self.pitch_delta = float(pitch_delta)   # exposed for scorer

        # ── Classify ──────────────────────────────────────────────────
        yaw_dominant = (abs(yaw_offset) > self.yaw_thr
                        and abs(yaw_offset) > abs(pitch_delta) * 1.5)

        if yaw_dominant:
            raw_dir = "LEFT" if yaw_offset > 0 else "RIGHT"
        elif pitch_delta < -self.pitch_thr:
            raw_dir = "UP"
        elif pitch_delta > self.pitch_thr:
            raw_dir = "DOWN"
        elif abs(yaw_offset) > self.yaw_thr:
            raw_dir = "LEFT" if yaw_offset > 0 else "RIGHT"
        else:
            raw_dir = "CENTER"

        # ── Hysteresis ────────────────────────────────────────────────
        if raw_dir == self._pending_dir:
            self._pending_count += 1
        else:
            self._pending_dir = raw_dir
            self._pending_count = 1

        if self._pending_count >= HYSTERESIS:
            self.current_direction = self._pending_dir

        if self.current_direction != "CENTER":
            self.movements_detected.add(self.current_direction)

        return self._result()

    def _result(self):
        return {
            "yaw":                round(self.yaw,         4),
            "pitch":              round(self.pitch,        4),
            "pitch_delta":        round(self.pitch_delta,  4),   # NEW
            "roll":               0.0,
            "direction":          self.current_direction,
            "movements_detected": list(self.movements_detected),
            "movement_count":     len(self.movements_detected),
        }

    def reset(self):
        self.movements_detected = set()
        self.current_direction = "CENTER"
        self.yaw = 0.0
        self.pitch = 0.0
        self.pitch_delta = 0.0
        self._pending_dir = "CENTER"
        self._pending_count = 0
        self._baseline_pitch = None
        self._baseline_frames = 0
        self._pitch_samples = []
