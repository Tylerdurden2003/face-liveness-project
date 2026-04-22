"""
motion_analyzer.py — UPGRADE #4
Micro-texture motion via Lucas-Kanade Optical Flow.
Now includes COHERENCE CHECK:
  - Real faces: vectors are random (low coherence) → live
  - Phone/photo tilted: all vectors move together uniformly (high coherence) → spoof
Score: 0.0 (spoof) → 1.0 (live)
"""
import cv2
import numpy as np

LK_PARAMS = dict(winSize=(15, 15), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
FEATURE_PARAMS = dict(maxCorners=60, qualityLevel=0.01,
                      minDistance=7, blockSize=7)
MOTION_FLOOR = 0.08
MOTION_CEIL = 1.80
COHERENCE_THRESHOLD = 0.82   # above this = rigid object moving = spoof


class MotionAnalyzer:
    def __init__(self, live_thresh=0.30, history=20):
        self.live_thresh = live_thresh
        self._history = history
        self._prev_gray = None
        self._prev_pts = None
        self._scores = []
        self.score = 0.5
        self.motion = 0.0
        self.coherence = 0.0   # NEW — exposed for UI/debug
        self.is_live = False
        self._coherence_buf = []   # rolling coherence history

    # ── public ────────────────────────────────────────────────────────
    def update(self, frame_bgr, face_box):
        x1, y1, x2, y2 = face_box
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return self._pack()

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_pts is None \
                or len(self._prev_pts) < 5:
            self._init_tracker(gray, face_box)
            return self._pack()

        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **LK_PARAMS)

        if nxt is None or st is None:
            self._init_tracker(gray, face_box)
            return self._pack()

        good_old = self._prev_pts[st.ravel() == 1]
        good_new = nxt[st.ravel() == 1]

        if len(good_old) < 4:
            self._init_tracker(gray, face_box)
            return self._pack()

        # ── motion magnitude ──────────────────────────────────────────
        displacements = good_new - good_old          # shape (N, 1, 2)
        displacements = displacements.reshape(-1, 2)
        mags = np.linalg.norm(displacements, axis=1)
        mag = float(np.median(mags))
        self.motion = mag

        # ── coherence check ───────────────────────────────────────────
        coherence = self._compute_coherence(displacements, mags)
        self._coherence_buf.append(coherence)
        if len(self._coherence_buf) > self._history:
            self._coherence_buf.pop(0)
        self.coherence = float(np.mean(self._coherence_buf))

        # ── raw motion score ──────────────────────────────────────────
        if mag < MOTION_FLOOR:
            raw = 0.0                                # static → spoof
        elif mag > MOTION_CEIL:
            raw = max(0.0, 1.0 - (mag - MOTION_CEIL) / MOTION_CEIL)
        else:
            raw = (mag - MOTION_FLOOR) / (MOTION_CEIL - MOTION_FLOOR)

        # ── coherence penalty ─────────────────────────────────────────
        # If vectors are too uniform the object is moving rigidly (phone tilt)
        # Smoothly penalise as coherence rises above threshold
        if self.coherence > COHERENCE_THRESHOLD:
            excess = (self.coherence - COHERENCE_THRESHOLD) / \
                (1.0 - COHERENCE_THRESHOLD + 1e-6)
            penalty = excess ** 1.5          # non-linear — hits hard near 1.0
            raw = raw * max(0.0, 1.0 - penalty)

        self._scores.append(raw)
        if len(self._scores) > self._history:
            self._scores.pop(0)
        self.score = float(np.mean(self._scores))
        self.is_live = self.score >= self.live_thresh

        # ── tracker upkeep ────────────────────────────────────────────
        self._prev_gray = gray.copy()
        self._prev_pts = good_new.reshape(-1, 1, 2)
        if len(self._prev_pts) < 10:
            self._init_tracker(gray, face_box)

        return self._pack()

    def reset(self):
        self._prev_gray = None
        self._prev_pts = None
        self._scores = []
        self._coherence_buf = []
        self.score = 0.5
        self.motion = 0.0
        self.coherence = 0.0
        self.is_live = False

    # ── private ───────────────────────────────────────────────────────
    def _compute_coherence(self, displacements, mags):
        """
        Returns 0..1.  High value = all vectors point the same way = rigid motion.

        Method: normalise each displacement vector, compute the length of
        their mean.  If all point the same direction the mean length → 1.
        If random the vectors cancel and mean length → 0.
        """
        moving_mask = mags > MOTION_FLOOR * 0.5
        if moving_mask.sum() < 4:
            return 0.0                           # not enough motion to judge

        d = displacements[moving_mask]
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1e-6, norms)
        unit_vecs = d / norms                  # normalise
        mean_vec = unit_vecs.mean(axis=0)
        coherence = float(np.linalg.norm(mean_vec))   # 0..1
        return coherence

    def _init_tracker(self, gray, face_box):
        x1, y1, x2, y2 = face_box
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)
        self._prev_gray = gray.copy()
        self._prev_pts = pts if pts is not None \
            else np.empty((0, 1, 2), dtype=np.float32)

    def _pack(self):
        return {
            "motion":     round(self.motion, 3),
            "coherence":  round(self.coherence, 3),   # NEW field
            "score":      round(self.score, 3),
            "is_live":    self.is_live,
        }
