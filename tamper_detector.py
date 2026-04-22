"""
tamper_detector.py
Detects evasion attempts during liveness verification:
  1. Camera cover     — sudden drop to near-black frame
  2. Lighting attack  — sudden dramatic brightness change
  3. Out of frame     — face disappears repeatedly
  4. Camera block     — sustained darkness over multiple frames
"""

import time
import numpy as np
import cv2

DARKNESS_THRESHOLD = 18     # mean pixel value below this = dark frame
BRIGHTNESS_JUMP = 55     # sudden mean brightness change = lighting attack
COVER_FRAMES = 8      # sustained dark frames to trigger camera cover
OUT_OF_FRAME_LIMIT = 45     # frames without face = out of frame tamper
LIGHTING_HISTORY = 6      # frames to compare for brightness jump


class TamperDetector:
    def __init__(self):
        self._dark_count = 0
        self._no_face_count = 0
        self._brightness_hist = []
        self._events = []   # list of tamper events with timestamps
        self._start_time = time.time()
        self.tamper_flag = False
        self.tamper_reason = None
        self.current_warning = None

    def update(self, frame_bgr, face_detected: bool) -> dict:
        elapsed = time.time() - self._start_time

        # ── Brightness metrics ────────────────────────────────────────
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_bright = float(np.mean(gray))

        self._brightness_hist.append(mean_bright)
        if len(self._brightness_hist) > LIGHTING_HISTORY:
            self._brightness_hist.pop(0)

        # ── Check 1: Camera cover / sustained darkness ─────────────────
        if mean_bright < DARKNESS_THRESHOLD:
            self._dark_count += 1
        else:
            self._dark_count = max(0, self._dark_count - 1)

        if self._dark_count >= COVER_FRAMES:
            self._log_event("CAMERA_COVER", elapsed,
                            "Camera covered or blocked")
            self.current_warning = "Camera covered — remove obstruction"
            self.tamper_flag = True
            self.tamper_reason = "Camera covered during session"

        # ── Check 2: Sudden lighting change ───────────────────────────
        elif len(self._brightness_hist) >= LIGHTING_HISTORY:
            brightness_delta = abs(
                mean_bright - np.mean(self._brightness_hist[:-1]))
            if brightness_delta > BRIGHTNESS_JUMP and elapsed > 2.0:
                self._log_event("LIGHTING_ATTACK", elapsed,
                                f"Brightness jumped {brightness_delta:.0f} units")
                self.current_warning = "Sudden lighting change detected"
                self.tamper_flag = True
                self.tamper_reason = "Lighting tampered during session"

        # ── Check 3: Face out of frame ────────────────────────────────
        if not face_detected:
            self._no_face_count += 1
        else:
            self._no_face_count = max(0, self._no_face_count - 2)

        if self._no_face_count >= OUT_OF_FRAME_LIMIT:
            self._log_event("OUT_OF_FRAME", elapsed,
                            "Face absent for extended period")
            self.current_warning = "Face not detected — stay in frame"
            self.tamper_flag = True
            self.tamper_reason = "Subject left frame repeatedly"

        # Clear warning if conditions normalise
        if (self._dark_count == 0
                and self._no_face_count < OUT_OF_FRAME_LIMIT // 2
                and not self.tamper_flag):
            self.current_warning = None

        return self._result(mean_bright, elapsed)

    def _log_event(self, event_type, elapsed, detail):
        """Only log each event type once per 3 seconds to avoid spam."""
        now = time.time()
        for e in self._events:
            if e["type"] == event_type and (now - e["ts_unix"]) < 3.0:
                return
        self._events.append({
            "type":    event_type,
            "detail":  detail,
            "elapsed": round(elapsed, 2),
            "ts_unix": now,
        })

    def _result(self, mean_bright, elapsed):
        return {
            "tamper_flag":     self.tamper_flag,
            "tamper_reason":   self.tamper_reason,
            "current_warning": self.current_warning,
            "dark_count":      self._dark_count,
            "no_face_count":   self._no_face_count,
            "mean_brightness": round(mean_bright, 1),
            "events":          list(self._events),
        }

    def get_events(self):
        return list(self._events)

    def reset(self):
        self._dark_count = 0
        self._no_face_count = 0
        self._brightness_hist = []
        self._events = []
        self._start_time = time.time()
        self.tamper_flag = False
        self.tamper_reason = None
        self.current_warning = None
