"""
app.py - Standalone OpenCV demo (optional).
Run manually with: python app.py
This file does NOT run automatically.
"""

import cv2
import mediapipe as mp
from camera import Camera
from blink_detector import BlinkDetector
from head_pose_detector import HeadPoseDetector
from texture_analyzer import TextureAnalyzer
from motion_analyzer import MotionAnalyzer
from pupil_analyzer import PupilAnalyzer
from challenge_engine import ChallengeEngine
from confidence_scorer import ConfidenceScorer
from drawing_utils import draw_face_box, draw_mesh_dots, apply_flash


def make_detectors():
    return (
        BlinkDetector(),
        HeadPoseDetector(),
        TextureAnalyzer(),
        MotionAnalyzer(),
        PupilAnalyzer(),
        ChallengeEngine(num_steps=2),
        ConfidenceScorer(),
    )


def _put(frame, text, row, color):
    cv2.putText(frame, text, (10, 30 + row * 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)


def main():
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    cam = Camera(0, 800, 600)
    blink, head, texture, motion, pupil, challenge, scorer = make_detectors()
    print("Liveness v2 standalone — q:quit  r:reset")

    with cam:
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces = res.multi_face_landmarks

            # ── Multi-face rejection ──────────────────────────────────
            if faces and len(faces) > 1:
                cv2.putText(frame,
                            "MULTIPLE FACES DETECTED — SPOOF",
                            (20, 50), cv2.FONT_HERSHEY_DUPLEX,
                            0.85, (0, 61, 255), 2)
                cv2.imshow("Liveness v2", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == ord("r"):
                    blink, head, texture, motion, \
                        pupil, challenge, scorer = make_detectors()
                continue

            if faces:
                fl = faces[0]
                blr = blink.update(fl, w, h)
                hdr = head.update(fl, w, h)
                pur = pupil.update(frame, fl, w, h)
                xs = [lm.x * w for lm in fl.landmark]
                ys = [lm.y * h for lm in fl.landmark]
                bbox = (int(min(xs)) - 5, int(min(ys)) - 5,
                        int(max(xs)) + 5, int(max(ys)) + 5)
                txr = texture.update(frame, bbox)
                mor = motion.update(frame, bbox)
                cr = challenge.update(blr, hdr)
                sr = scorer.update(cr, txr, mor, pur, hdr)

                if pur.get("flash_active"):
                    apply_flash(frame)
                draw_mesh_dots(frame, fl, w, h)
                draw_face_box(frame, fl, w, h, sr["verdict"])

                verdict = sr["verdict"]
                conf = sr.get("confidence",     0.0)
                coherence = mor.get("coherence",     0.0)
                tx_score = txr.get("texture_score", 0.0)
                glare_s = txr.get("glare_score",   1.0)
                mo_score = mor.get("score",         0.0)
                time_left = sr.get("time_remaining", 0.0)
                challenge_label = cr.get("current_label", "---")

                verdict_color = (
                    (0, 230, 118) if verdict == "LIVE" else
                    (0,  61, 255) if verdict == "SPOOF" else
                    (0, 180, 255)
                )
                coh_color = (0, 61, 255) if coherence > 0.82 else (
                    200, 200, 200)
                glare_color = (0, 61, 255) if glare_s < 0.35 else (
                    200, 200, 200)

                _put(frame, f"VERDICT   : {verdict}",
                     0, verdict_color)
                _put(frame, f"CONFIDENCE: {conf:.3f}",
                     1, (200, 200, 200))
                _put(
                    frame, f"CHALLENGE : {challenge_label}",            2, (0, 229, 255))
                _put(frame, f"TEXTURE   : {tx_score:.3f}",
                     3, (100, 255, 218))
                _put(frame, f"GLARE     : {glare_s:.3f}"
                     + (" << SCREEN" if glare_s < 0.35 else ""),  4, glare_color)
                _put(frame, f"MOTION    : {mo_score:.3f}",
                     5, (200, 200, 200))
                _put(frame, f"COHERENCE : {coherence:.3f}"
                     + (" << RIGID" if coherence > 0.82 else ""), 6, coh_color)
                _put(
                    frame, f"TIME LEFT : {time_left:.1f}s",             7, (200, 200, 200))

            else:
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 100, 255), 2)

            cv2.imshow("Liveness v2", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("r"):
                blink, head, texture, motion, \
                    pupil, challenge, scorer = make_detectors()

    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
