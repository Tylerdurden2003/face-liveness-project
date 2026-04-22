"""
pupil_analyzer.py  ── UPGRADE #5
===================================
Pupil response detection.

A live eye's pupil will micro-constrict when a bright stimulus is
flashed on screen.  A printed photo or replay has no pupil response.

Method
------
1. Extract the iris/pupil region using MediaPipe's refined landmarks
   (indices 468-477 are the refined iris landmarks when
   refine_landmarks=True is set in FaceMesh).
2. Measure average pixel intensity inside the iris bounding box.
3. When a "flash" is triggered (bright white overlay shown for
   FLASH_FRAMES frames), record pre-flash vs post-flash iris intensity.
4. If intensity *increases* (darker pupil → less surface area of dark
   pupil = constriction) or the variance changes in the expected
   direction → pupil_response = True.

Note: This is a subtle signal and works best in controlled lighting.
The engine is conservative: it only marks a positive response if the
intensity delta exceeds a meaningful threshold across multiple frames.

MediaPipe iris landmark indices (refined landmarks):
  Left iris:  [473, 474, 475, 476, 477]  (centre = 473)
  Right iris: [468, 469, 470, 471, 472]  (centre = 468)
"""

import cv2
import numpy as np

LEFT_IRIS_IDX  = [473, 474, 475, 476, 477]
RIGHT_IRIS_IDX = [468, 469, 470, 471, 472]

FLASH_FRAMES     = 4    # frames to show the white flash overlay
RESPONSE_FRAMES  = 8    # frames to wait post-flash for pupil response
DELTA_THRESHOLD  = 3.5  # intensity increase units (0-255) required
FLASH_ALPHA      = 0.55 # overlay opacity


class PupilAnalyzer:
    """
    Detects pupil constriction response to a white flash.

    The flash is triggered automatically after MIN_STABLE_FRAMES of
    stable face tracking.  It fires at most once per session.
    """

    def __init__(self, min_stable_frames: int = 30):
        self.min_stable_frames = min_stable_frames
        self._stable_count     = 0
        self._flash_countdown  = 0     # >0 while flash is active
        self._response_wait    = 0     # >0 while waiting for response
        self._pre_flash_intensity: list[float] = []
        self._post_flash_intensity: list[float] = []
        self._fired            = False
        self.pupil_response    = False
        self.show_flash        = False
        self.delta             = 0.0
        self.pupil_score       = 0.0

    # ------------------------------------------------------------------

    def _iris_intensity(self, gray: np.ndarray, lm, indices: list, w: int, h: int) -> float:
        pts = np.array(
            [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
        )
        x1, y1 = pts.min(axis=0) - 3
        x2, y2 = pts.max(axis=0) + 3
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(np.mean(roi))

    # ------------------------------------------------------------------

    def update(self, frame_bgr: np.ndarray, face_landmarks, w: int, h: int) -> dict:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lm   = face_landmarks.landmark

        left_int  = self._iris_intensity(gray, lm, LEFT_IRIS_IDX,  w, h)
        right_int = self._iris_intensity(gray, lm, RIGHT_IRIS_IDX, w, h)
        avg_int   = (left_int + right_int) / 2.0

        self.show_flash = False

        if not self._fired:
            self._stable_count += 1

            # Trigger flash after stable detection
            if self._stable_count == self.min_stable_frames:
                self._flash_countdown = FLASH_FRAMES
                self._pre_flash_intensity = []
                self._post_flash_intensity = []
                self._fired = True

        if self._flash_countdown > 0:
            self.show_flash = True
            self._pre_flash_intensity.append(avg_int)
            self._flash_countdown -= 1
            if self._flash_countdown == 0:
                self._response_wait = RESPONSE_FRAMES

        elif self._response_wait > 0:
            self._post_flash_intensity.append(avg_int)
            self._response_wait -= 1

            if self._response_wait == 0 and self._pre_flash_intensity:
                pre  = np.mean(self._pre_flash_intensity)
                post = np.mean(self._post_flash_intensity) if self._post_flash_intensity else pre
                # After flash, pupil constricts → iris region gets brighter
                # (pupil dark area shrinks, more iris/sclera visible)
                self.delta = float(post - pre)
                if self.delta >= DELTA_THRESHOLD:
                    self.pupil_response = True
                # Normalise score
                self.pupil_score = float(np.clip(self.delta / (DELTA_THRESHOLD * 2), 0.0, 1.0))

        return self._result()

    # ------------------------------------------------------------------

    def apply_flash_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Call every frame; overlays white flash when active."""
        if not self.show_flash:
            return frame_bgr
        overlay = np.full_like(frame_bgr, 255)
        return cv2.addWeighted(overlay, FLASH_ALPHA, frame_bgr, 1 - FLASH_ALPHA, 0)

    def _result(self) -> dict:
        return {
            "pupil_response": self.pupil_response,
            "pupil_score":    round(self.pupil_score, 4),
            "flash_active":   self.show_flash,
            "delta":          round(self.delta, 2),
            "fired":          self._fired,
        }

    def reset(self):
        self._stable_count     = 0
        self._flash_countdown  = 0
        self._response_wait    = 0
        self._pre_flash_intensity  = []
        self._post_flash_intensity = []
        self._fired            = False
        self.pupil_response    = False
        self.show_flash        = False
        self.delta             = 0.0
        self.pupil_score       = 0.0
