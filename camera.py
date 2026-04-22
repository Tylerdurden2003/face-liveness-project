"""camera.py — OpenCV VideoCapture wrapper."""
import cv2

class Camera:
    def __init__(self, source=0, width=800, height=600):
        self.source = source; self.width = width; self.height = height
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def read(self):
        return self._cap.read()

    def read_rgb(self):
        ret, frame = self.read()
        if not ret: return ret, frame
        import cv2 as _cv2
        return ret, _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)

    def release(self):
        if self._cap and self._cap.isOpened(): self._cap.release()
        self._cap = None

    def is_opened(self): return self._cap is not None and self._cap.isOpened()
    def __enter__(self): return self.open()
    def __exit__(self, *a): self.release()
