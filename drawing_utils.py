"""
drawing_utils.py
Full overlay renderer for the v2 liveness system.
Shows: face box, landmark dots, HUD with all 6 signal scores,
challenge instruction banner, confidence gauge, flash overlay.
"""

import cv2
import numpy as np

C = {
    "green":  (0, 220, 90),
    "red":    (0, 60,  230),
    "orange": (0, 165, 255),
    "white":  (255, 255, 255),
    "black":  (0,   0,   0),
    "cyan":   (255, 220,  0),
    "yellow": (0,  220, 220),
    "purple": (220, 80, 200),
    "teal":   (200, 220,  50),
    "gray":   (130, 130, 130),
}

VERDICT_C = {"LIVE": C["green"], "SPOOF": C["red"], "CHECKING": C["orange"]}
FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_face_box(frame, face_landmarks, w, h, verdict="CHECKING"):
    xs = [lm.x * w for lm in face_landmarks.landmark]
    ys = [lm.y * h for lm in face_landmarks.landmark]
    x1, y1 = int(min(xs)) - 10, int(min(ys)) - 10
    x2, y2 = int(max(xs)) + 10, int(max(ys)) + 10
    col = VERDICT_C.get(verdict, C["white"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    lc, th = 22, 3
    for (px, py, sx, sy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px, py), (px + sx*lc, py),         col, th)
        cv2.line(frame, (px, py), (px,         py + sy*lc), col, th)
    return x1, y1, x2, y2


def draw_mesh_dots(frame, face_landmarks, w, h, colour=None, radius=1):
    colour = colour or C["green"]
    for lm in face_landmarks.landmark:
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), radius, colour, -1)


def _bar(frame, x, y, w, h_bar, value, colour, label):
    cv2.rectangle(frame, (x, y), (x + w, y + h_bar), C["gray"], 1)
    fill = int(w * max(0.0, min(1.0, value)))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h_bar), colour, -1)
    cv2.putText(frame, f"{label} {value:.2f}", (x, y - 3), FONT, 0.36, colour, 1, cv2.LINE_AA)


def draw_hud(frame, scorer_result, challenge_result, blink_result,
             head_result, texture_result, motion_result, pupil_result):
    h_f, w_f = frame.shape[:2]

    # Sidebar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, h_f), C["black"], -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

    verdict    = scorer_result.get("verdict", "CHECKING")
    confidence = scorer_result.get("confidence", 0.0)
    threshold  = scorer_result.get("live_threshold", 0.58)
    comp       = scorer_result.get("component_scores", {})
    verdict_c  = VERDICT_C.get(verdict, C["white"])

    y = 28
    cv2.putText(frame, "LIVENESS v2",   (8, y), FONT, 0.60, C["white"],  1, cv2.LINE_AA); y += 26
    cv2.putText(frame, verdict,         (8, y), FONT, 0.90, verdict_c,   2, cv2.LINE_AA); y += 30
    cv2.putText(frame, f"Conf: {confidence:.3f} / {threshold}", (8, y), FONT, 0.40, verdict_c, 1, cv2.LINE_AA); y += 22

    # Confidence bar
    _bar(frame, 8, y, 254, 8, confidence / threshold, verdict_c, ""); y += 18

    cv2.line(frame, (5, y), (265, y), C["gray"], 1); y += 10

    # Signal scores
    scores = [
        ("Challenge", comp.get("challenge", 0), C["cyan"]),
        ("Texture",   comp.get("texture",   0), C["yellow"]),
        ("Motion",    comp.get("motion",    0), C["teal"]),
        ("Pupil",     comp.get("pupil",     0), C["purple"]),
    ]
    for label, val, col in scores:
        _bar(frame, 8, y, 254, 7, val, col, label); y += 20

    cv2.line(frame, (5, y), (265, y), C["gray"], 1); y += 8

    # Blink / head
    cv2.putText(frame, f"Blinks : {blink_result.get('blink_count',0)}", (8, y), FONT, 0.45, C["cyan"],   1, cv2.LINE_AA); y += 18
    cv2.putText(frame, f"EAR    : {blink_result.get('ear',0):.3f}",    (8, y), FONT, 0.42, C["cyan"],   1, cv2.LINE_AA); y += 18
    cv2.putText(frame, f"Head   : {head_result.get('direction','—')}",  (8, y), FONT, 0.45, C["yellow"], 1, cv2.LINE_AA); y += 18
    cv2.putText(frame, f"Yaw    : {head_result.get('yaw_deg',0):+.1f}°", (8, y), FONT, 0.40, C["yellow"], 1, cv2.LINE_AA); y += 18
    cv2.putText(frame, f"Pitch  : {head_result.get('pitch_deg',0):+.1f}°",(8, y),FONT, 0.40, C["yellow"], 1, cv2.LINE_AA); y += 18
    cv2.putText(frame, f"Flow   : {motion_result.get('flow_mag',0):.3f}", (8, y),FONT, 0.42, C["teal"],   1, cv2.LINE_AA); y += 18
    cv2.putText(frame, f"LBP    : {texture_result.get('lbp_score',0):.3f}",(8,y),FONT, 0.40, C["orange"], 1, cv2.LINE_AA); y += 18
    cv2.putText(frame, f"FFT    : {texture_result.get('fft_score',0):.3f}",(8,y),FONT, 0.40, C["orange"], 1, cv2.LINE_AA); y += 18

    if pupil_result.get("fired"):
        pu_txt = f"Pupil Δ: {pupil_result.get('delta',0):+.1f}"
        pu_col = C["green"] if pupil_result.get("pupil_response") else C["red"]
        cv2.putText(frame, pu_txt, (8, y), FONT, 0.42, pu_col, 1, cv2.LINE_AA); y += 18

    # Challenge banner (bottom of frame)
    instr = challenge_result.get("current_instruction", "")
    icon  = challenge_result.get("current_icon", "")
    step  = challenge_result.get("current_step", 0)
    total = len(challenge_result.get("challenge_sequence", []))
    pct   = challenge_result.get("step_elapsed", 0) / max(challenge_result.get("step_timeout", 8), 1)

    banner_h = 46
    bx1, by1 = 280, h_f - banner_h - 4
    bx2, by2 = w_f - 4, h_f - 4
    ov2 = frame.copy()
    cv2.rectangle(ov2, (bx1, by1), (bx2, by2), (10, 20, 40), -1)
    cv2.addWeighted(ov2, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), C["cyan"], 1)

    txt = f"Step {step+1}/{total}: {instr}"
    cv2.putText(frame, txt, (bx1 + 10, by1 + 20), FONT, 0.52, C["cyan"], 1, cv2.LINE_AA)

    # Step timer bar
    timer_w = int((bx2 - bx1 - 4) * (1.0 - min(pct, 1.0)))
    timer_c = C["green"] if pct < 0.6 else (C["orange"] if pct < 0.85 else C["red"])
    cv2.rectangle(frame, (bx1+2, by2-8), (bx1+2+timer_w, by2-2), timer_c, -1)


def apply_flash(frame):
    overlay = np.full_like(frame, 240)
    return cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
