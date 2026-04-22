"""
blink_detector.py — Eye blink detection via Eye Aspect Ratio (EAR).
MediaPipe FaceMesh indices:
  Left eye:  [362, 385, 387, 263, 373, 380]
  Right eye: [33,  160, 158, 133, 153, 144]
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
"""
import numpy as np
from scipy.spatial import distance as dist

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 2

def _ear(landmarks, indices, w, h):
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
        dtype=np.float64,
    )
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

class BlinkDetector:
    def __init__(self, ear_threshold=EAR_THRESHOLD, consec_frames=CONSEC_FRAMES):
        self.ear_threshold  = ear_threshold
        self.consec_frames  = consec_frames
        self._frame_counter = 0
        self.blink_count    = 0
        self.ear            = 1.0
        self.eye_closed     = False

    def update(self, face_landmarks, w, h):
        lm = face_landmarks.landmark
        left_ear  = _ear(lm, LEFT_EYE_IDX,  w, h)
        right_ear = _ear(lm, RIGHT_EYE_IDX, w, h)
        self.ear  = (left_ear + right_ear) / 2.0
        if self.ear < self.ear_threshold:
            self._frame_counter += 1
            self.eye_closed = True
        else:
            if self._frame_counter >= self.consec_frames:
                self.blink_count += 1
            self._frame_counter = 0
            self.eye_closed     = False
        return {"ear": round(self.ear, 4), "eye_closed": self.eye_closed,
                "blink_count": self.blink_count}

    def reset(self):
        self._frame_counter = 0; self.blink_count = 0
        self.ear = 1.0; self.eye_closed = False
