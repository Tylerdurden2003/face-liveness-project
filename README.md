# Face Liveness Detection v2

Production-grade anti-spoofing system using **6 independent signal layers**
fused into a single weighted confidence score.  No cloud, no deep-learning
inference server, no GPU required — pure OpenCV + MediaPipe + NumPy.

---

## What's New vs v1

| # | Upgrade | File | Why it Matters |
|---|---------|------|---------------|
| 1 | **LBP + FFT Texture Analysis** | `texture_analyzer.py` | Detects paper grain, moiré patterns, and screen pixel grids |
| 2 | **solvePnP Head Pose** | `head_pose_detector.py` | Proper Euler angles in degrees; robust across face sizes |
| 3 | **Optical Flow Micro-Motion** | `motion_analyzer.py` | A still photo has zero flow; live skin never does |
| 4 | **Randomised Challenge-Response** | `challenge_engine.py` | Prevents replay attacks; sequence differs every session |
| 5 | **Pupil Flash Response** | `pupil_analyzer.py` | Live eyes constrict to bright flash; photos don't |
| 6 | **Weighted Confidence Scoring** | `confidence_scorer.py` | Fuses all signals; no single-point-of-failure verdict |

---

## Project Structure

```
face_liveness_v2/
├── app.py                  # Standalone OpenCV demo
├── streamlit_app.py        # Streamlit web UI
├── blink_detector.py       # EAR blink counter
├── head_pose_detector.py   # solvePnP yaw/pitch/roll
├── texture_analyzer.py     # LBP entropy + FFT spread
├── motion_analyzer.py      # Lucas-Kanade optical flow
├── pupil_analyzer.py       # Pupil constriction flash test
├── challenge_engine.py     # Random ordered challenge sequence
├── confidence_scorer.py    # Weighted multi-signal fusion
├── camera.py               # OpenCV camera wrapper
├── drawing_utils.py        # Full HUD + overlay renderer
├── requirements.txt
└── README.md
```

---

## Installation

> Python **3.11** required.  Python 3.12+ not supported by mediapipe 0.10.x.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running

### OpenCV window

```bash
python app.py
# q = quit   r = reset
```

### Streamlit web UI

```bash
streamlit run streamlit_app.py
# Open http://localhost:8501
```

---

## How Each Signal Works

### 1. Texture Analysis (LBP + FFT)
- **LBP entropy**: Real skin has chaotic micro-texture → high entropy.
  A printed face has uniform paper grain → low entropy.
- **FFT spread**: Live skin energy is broadband.
  Screens/prints have moiré spikes at predictable frequencies → low spread score.
- Scores averaged over 15 frames to reduce noise.

### 2. Head Pose (solvePnP)
- 6 stable landmarks (nose, chin, eye corners, mouth corners) mapped to
  a canonical 3D face model.
- `cv2.solvePnP` → rotation vector → `cv2.Rodrigues` → rotation matrix →
  `cv2.decomposeProjectionMatrix` → Euler angles (pitch, yaw, roll) in degrees.
- Thresholds: yaw ±12°, pitch ±10°.

### 3. Micro-Motion (Lucas-Kanade)
- 10 distributed facial landmarks tracked frame-to-frame with `calcOpticalFlowPyrLK`.
- Mean flow magnitude maintained over 20-frame rolling window.
- Floor: 0.30 px/frame.  Below floor → motion_score = 0 → spoof signal.

### 4. Challenge-Response
- Each session randomly draws 2 challenges from: BLINK, LOOK_LEFT, LOOK_RIGHT,
  LOOK_UP, LOOK_DOWN.  Order is randomised.
- User must complete them in sequence within 8 s per step.
- Makes pre-recorded replay attacks impractical.

### 5. Pupil Flash Response
- After 30 frames of stable tracking, a white overlay (55% opacity) is
  flashed for 4 frames.
- Iris intensity is measured before and after.
- A live pupil constricts → the iris region gets brighter post-flash.
- Required delta: +3.5 intensity units (0-255 scale).

### 6. Confidence Scoring
```
confidence = 0.40 × challenge_score
           + 0.25 × texture_score
           + 0.20 × motion_score
           + 0.15 × pupil_score
```
Rolling 30-frame average must exceed **0.58** AND challenge must be complete
for a LIVE verdict.  Any challenge timeout → immediate SPOOF.

---

## Tuning

```python
# confidence_scorer.py
ConfidenceScorer(live_threshold=0.58, history_len=30)

# challenge_engine.py
ChallengeEngine(n_challenges=2, step_timeout=8.0)

# texture_analyzer.py
TextureAnalyzer(live_threshold=0.52, history_len=15)

# motion_analyzer.py
MotionAnalyzer(live_floor=0.30, history_len=20)

# head_pose_detector.py
HeadPoseDetector(yaw_thr=12.0, pitch_thr=10.0)
```

---

## Attack Resistance

| Attack | Blocked By |
|--------|-----------|
| Printed photo held still | Texture (LBP/FFT) + Motion (zero flow) |
| Printed photo tilted/moved | Challenge randomisation + Pupil response |
| Screen replay (video) | Texture (pixel grid FFT) + Pupil flash (no response) |
| Pre-recorded compliance clip | Challenge randomisation (unpredictable sequence) |
| 3D printed mask | Texture (no organic micro-texture) + Motion + Pupil |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Camera not found | `Camera(source=1)` or check permissions |
| Pupil test fails in dark room | Increase ambient lighting |
| High false SPOOF rate | Lower `live_threshold` to 0.50 |
| High false LIVE rate | Raise `live_threshold` to 0.65 |
| Low FPS | Reduce `Camera(width=640, height=480)` |
