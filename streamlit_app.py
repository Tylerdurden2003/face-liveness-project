import cv2
import time
import streamlit as st
import mediapipe as mp

from blink_detector import BlinkDetector
from head_pose_detector import HeadPoseDetector
from texture_analyzer import TextureAnalyzer
from motion_analyzer import MotionAnalyzer
from pupil_analyzer import PupilAnalyzer
from challenge_engine import ChallengeEngine
from confidence_scorer import ConfidenceScorer
from session_logger import SessionLogger
from tamper_detector import TamperDetector
from report_generator import generate_report, REPORTLAB_AVAILABLE
from drawing_utils import draw_face_box, draw_mesh_dots, apply_flash

st.set_page_config(page_title="Liveness Detection v2", page_icon=":shield:",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap");
html,body,[class*="css"]{font-family:"Syne",sans-serif;background:#030810;color:#aac8e8;}
.mtitle{font-family:"Share Tech Mono",monospace;font-size:2rem;letter-spacing:.16em;color:#00e5ff;text-shadow:0 0 22px #00e5ff66;margin:0 0 4px 0;}
.mbadge{font-size:.7rem;letter-spacing:.3em;color:#1e5070;text-transform:uppercase;}
.sc{background:linear-gradient(135deg,#07131f,#05101a);border:1px solid #0e2a44;border-radius:10px;padding:14px 16px 10px;margin-bottom:10px;}
.sl{font-size:.62rem;letter-spacing:.22em;text-transform:uppercase;color:#305878;margin-bottom:4px;}
.sv{font-family:"Share Tech Mono",monospace;font-size:1.55rem;font-weight:700;line-height:1.15;}
.ss{font-size:.68rem;color:#305878;margin-top:2px;}
.live{color:#00e676;text-shadow:0 0 10px #00e67666;}
.spoof{color:#ff3d57;text-shadow:0 0 10px #ff3d5766;}
.checking{color:#ffab40;text-shadow:0 0 10px #ffab4066;}
.cyan{color:#00e5ff;}.yellow{color:#ffe066;}.teal{color:#64ffda;}.purple{color:#ce93d8;}.orange{color:#ffab40;}
.gt{background:#0a1e30;border:1px solid #1e3a5f;border-radius:6px;height:12px;overflow:hidden;margin:6px 0;}
.gf{height:100%;border-radius:6px;}
.cb{background:linear-gradient(90deg,#041828,#062035);border:1px solid #00e5ff44;border-left:3px solid #00e5ff;border-radius:0 10px 10px 0;padding:14px 20px;margin:6px 0;}
.cs{font-size:.62rem;letter-spacing:.25em;text-transform:uppercase;color:#00e5ff88;}
.ci{font-family:"Share Tech Mono",monospace;font-size:1.05rem;color:#00e5ff;margin:4px 0 0;}
.sp{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.7rem;letter-spacing:.1em;margin:3px 3px 0 0;border:1px solid #1e3a5f;color:#4a80a8;}
.sp.done{background:#003d20;border-color:#00e676;color:#00e676;}
.sp.active{background:#002b3d;border-color:#00e5ff;color:#00e5ff;}
.warn{background:linear-gradient(90deg,#1a0800,#1f0d00);border:1px solid #ff3d5744;border-left:3px solid #ff3d57;border-radius:0 10px 10px 0;padding:10px 16px;margin:6px 0;font-family:"Share Tech Mono",monospace;font-size:.85rem;color:#ff3d57;}
.live-banner{background:linear-gradient(90deg,#001a0d,#002b14);border:1px solid #00e67644;border-left:3px solid #00e676;border-radius:0 10px 10px 0;padding:10px 16px;margin:6px 0;font-family:"Share Tech Mono",monospace;font-size:.85rem;color:#00e676;}
.reason{background:linear-gradient(90deg,#1a0800,#1f0d00);border:1px solid #ff3d5744;border-left:3px solid #ff3d57;border-radius:0 10px 10px 0;padding:14px 20px;margin:6px 0;}
.reason-label{font-size:.62rem;letter-spacing:.25em;text-transform:uppercase;color:#ff3d5788;}
.reason-text{font-family:"Share Tech Mono",monospace;font-size:.95rem;color:#ff3d57;margin:4px 0 0;}
.reverify{background:linear-gradient(90deg,#1a1000,#1f1500);border:1px solid #ffab4044;border-left:3px solid #ffab40;border-radius:0 10px 10px 0;padding:10px 16px;margin:6px 0;font-family:"Share Tech Mono",monospace;font-size:.85rem;color:#ffab40;}
.session-complete-live{background:linear-gradient(90deg,#001a0d,#003320);border:2px solid #00e676;border-radius:10px;padding:18px 24px;margin:8px 0;text-align:center;font-family:"Share Tech Mono",monospace;}
.session-complete-spoof{background:linear-gradient(90deg,#1a0005,#330010);border:2px solid #ff3d57;border-radius:10px;padding:18px 24px;margin:8px 0;text-align:center;font-family:"Share Tech Mono",monospace;}
.sc-title-live{font-size:1.5rem;color:#00e676;text-shadow:0 0 20px #00e67688;font-weight:700;letter-spacing:.1em;}
.sc-title-spoof{font-size:1.5rem;color:#ff3d57;text-shadow:0 0 20px #ff3d5788;font-weight:700;letter-spacing:.1em;}
.sc-sub{font-size:.75rem;color:#aac8e8;margin-top:6px;letter-spacing:.05em;}
div[data-testid="stButton"] button{background:linear-gradient(135deg,#07131f,#05101a);border:1px solid #00e5ff33;color:#00e5ff;font-family:"Share Tech Mono",monospace;letter-spacing:.12em;border-radius:6px;}
div[data-testid="stButton"] button:hover{border-color:#00e5ff;box-shadow:0 0 14px #00e5ff33;}
div[data-testid="stDownloadButton"] button{background:linear-gradient(135deg,#003320,#004d30) !important;border:2px solid #00e676 !important;color:#00e676 !important;font-family:"Share Tech Mono",monospace;letter-spacing:.12em;border-radius:6px;font-size:1rem;padding:12px !important;font-weight:700;}
div[data-testid="stDownloadButton"] button:hover{box-shadow:0 0 20px #00e67644 !important;}
footer,#MainMenu{visibility:hidden;}.block-container{padding-top:1.2rem;}
</style>
""", unsafe_allow_html=True)

REVERIFY_INTERVAL = 300


def _detect_cameras():
    available = []
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available if available else [0]


def _make():
    return {
        "blink":     BlinkDetector(),
        "head":      HeadPoseDetector(),
        "texture":   TextureAnalyzer(),
        "motion":    MotionAnalyzer(),
        "pupil":     PupilAnalyzer(),
        "challenge": ChallengeEngine(num_steps=2),
        "scorer":    ConfidenceScorer(),
        "logger":    SessionLogger(),
        "tamper":    TamperDetector(),
    }


if "D" not in st.session_state:
    st.session_state.D = _make()
if "cam_idx" not in st.session_state:
    st.session_state.cam_idx = 0
if "cameras" not in st.session_state:
    st.session_state.cameras = _detect_cameras()
if "running" not in st.session_state:
    st.session_state.running = False
if "last" not in st.session_state:
    st.session_state.last = None
if "graph_path" not in st.session_state:
    st.session_state.graph_path = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "exam_mode" not in st.session_state:
    st.session_state.exam_mode = False
if "exam_verified" not in st.session_state:
    st.session_state.exam_verified = False
if "last_verify_time" not in st.session_state:
    st.session_state.last_verify_time = None
if "reverify_count" not in st.session_state:
    st.session_state.reverify_count = 0
if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = ""
if "candidate_id" not in st.session_state:
    st.session_state.candidate_id = ""

st.markdown('''<p class="mtitle">LIVENESS DETECTION v2</p>
<p class="mbadge">Anti-Spoofing — 6-Signal Fusion — MediaPipe FaceMesh</p>
<hr style="border-color:#0e2a44;margin:10px 0 18px">''', unsafe_allow_html=True)

vc, pc = st.columns([3, 2], gap="large")
with pc:
    vph = st.empty()
    sesph = st.empty()
    dlph = st.empty()
    cph = st.empty()
    bph = st.empty()
    rph = st.empty()
    tph = st.empty()
    dph = st.empty()
    r1a, r1b = st.columns(2)
    with r1a:
        blph = st.empty()
    with r1b:
        hdph = st.empty()
    r2a, r2b = st.columns(2)
    with r2a:
        txph = st.empty()
    with r2b:
        moph = st.empty()
    r3a, r3b = st.columns(2)
    with r3a:
        puph = st.empty()
    with r3b:
        cohph = st.empty()
    r4a, r4b = st.columns(2)
    with r4a:
        eph = st.empty()
    with r4b:
        warnph = st.empty()
    st.markdown('<hr style="border-color:#0e2a44;margin:8px 0">',
                unsafe_allow_html=True)
    st.markdown('<div class="sl" style="margin-bottom:4px">CANDIDATE INFO</div>',
                unsafe_allow_html=True)
    cname = st.text_input("Full Name", value=st.session_state.candidate_name,
                          placeholder="Enter candidate name",
                          label_visibility="collapsed")
    cid = st.text_input("Candidate ID", value=st.session_state.candidate_id,
                        placeholder="Enter candidate ID / roll number",
                        label_visibility="collapsed")
    st.session_state.candidate_name = cname
    st.session_state.candidate_id = cid
    exam_mode = st.toggle("Exam Mode (periodic re-verification)",
                          value=st.session_state.exam_mode)
    st.session_state.exam_mode = exam_mode
    cam_list = st.session_state.cameras
    sel = st.selectbox(
        "Select Camera", options=cam_list,
        format_func=lambda i:
            f"Camera {i} {'(likely USB webcam)' if i > 0 else '(built-in / default)'}",
        key="cam_select")
    st.session_state.cam_idx = sel
    if st.button("Refresh Camera List", use_container_width=True):
        st.session_state.cameras = _detect_cameras()
        st.rerun()
    b1, b2 = st.columns(2)
    with b1:
        stbtn = st.button("START / NEW", use_container_width=True)
    with b2:
        rsbtn = st.button("RESET", use_container_width=True)
    if stbtn:
        st.session_state.D = _make()
        st.session_state.running = True
        st.session_state.last = None
        st.session_state.graph_path = None
        st.session_state.pdf_path = None
        st.session_state.exam_verified = False
        st.session_state.last_verify_time = None
    if rsbtn:
        st.session_state.D = _make()
        st.session_state.running = False
        st.session_state.last = None
        st.session_state.graph_path = None
        st.session_state.pdf_path = None
        st.session_state.exam_verified = False
        st.session_state.last_verify_time = None
        st.session_state.reverify_count = 0

with vc:
    vidph = st.empty()
    capph = st.empty()
    mfph = st.empty()
    rvph = st.empty()
    grph = st.empty()
    pdfph = st.empty()


def _vcard(verdict, conf, threshold):
    cls = {"LIVE": "live", "SPOOF": "spoof",
           "CHECKING": "checking"}.get(verdict, "checking")
    col = {"LIVE": "#00e676", "SPOOF": "#ff3d57",
           "CHECKING": "#ffab40"}.get(verdict, "#ffab40")
    pct = min(conf / max(threshold, 0.01), 1.0) * 100
    return (
        '<div class="sc"><div class="sl">VERDICT</div>'
        f'<div class="sv {cls}">{verdict}</div>'
        '<div class="gt"><div class="gf" style="width:'
        f'{pct:.0f}%;background:{col}"></div></div>'
        f'<div class="ss">Confidence {conf:.3f} / threshold {threshold}</div></div>'
    )


def _rcard(reason):
    if not reason:
        return ""
    return (
        '<div class="reason"><div class="reason-label">Spoof type detected</div>'
        f'<div class="reason-text">&#9888; {reason}</div></div>'
    )


def _ccard(cr):
    steps = cr.get("steps", [])
    idx = cr.get("current_step_idx", 0)
    lbl = cr.get("current_label", "---")
    pills = "".join(
        f'<span class="sp {"done" if s["completed"] else ("active" if i == idx else "")}">'
        f'{s["label"]}</span>'
        for i, s in enumerate(steps)
    )
    done = cr.get("steps_done", 0)
    tot = cr.get("total_steps", 2)
    return (
        '<div class="cb"><div class="cs">Step ' +
        str(done + 1) + ' of ' + str(tot) + '</div>'
        f'<div class="ci">{lbl}</div>'
        f'<div style="margin-top:8px">{pills}</div></div>'
    )


def _sc(label, val, sub, cls):
    return (
        f'<div class="sc"><div class="sl">{label}</div>'
        f'<div class="sv {cls}">{val}</div>'
        f'<div class="ss">{sub}</div></div>'
    )


def _gc(label, score, cls, col):
    pct = min(score * 100, 100)
    return (
        '<div class="sc"><div class="sl">' + label + ' SCORE</div>'
        '<div class="gt"><div class="gf" style="width:'
        f'{pct:.0f}%;background:{col}"></div></div>'
        f'<div class="sv {cls}" style="font-size:1.1rem">{score:.3f}</div></div>'
    )


def _bcard(sr):
    comp = sr.get("component_scores", {})
    rows = "".join(
        f'<div style="display:inline-block;margin:4px 10px 4px 0">'
        f'<div class="sl">{k.upper()}</div>'
        f'<div style="font-family:monospace;font-size:.95rem;color:#aac8e8">{v:.3f}</div></div>'
        for k, v in comp.items()
    )
    return f'<div class="sc"><div class="sl">SIGNAL BREAKDOWN</div>{rows}</div>'


def _timercard(time_remaining, timeout):
    pct = max(0.0, time_remaining / timeout) * 100
    if time_remaining > timeout * 0.5:
        col = "#00e676"
    elif time_remaining > timeout * 0.25:
        col = "#ffab40"
    else:
        col = "#ff3d57"
    urgent = " HURRY UP!" if time_remaining < 6 else ""
    return (
        '<div class="sc"><div class="sl">TIME REMAINING</div>'
        f'<div class="sv" style="color:{col};font-size:2rem">{time_remaining:.0f}s{urgent}</div>'
        '<div class="gt"><div class="gf" style="width:'
        f'{pct:.0f}%;background:{col}"></div></div></div>'
    )


def _coherence_card(coherence_val):
    if coherence_val < 0.60:
        cls, label = "teal", "NORMAL"
    elif coherence_val < 0.82:
        cls, label = "yellow", "ELEVATED"
    else:
        cls, label = "spoof", "RIGID — SPOOF"
    pct = min(coherence_val * 100, 100)
    col = {"teal": "#64ffda", "yellow": "#ffe066",
           "spoof": "#ff3d57"}.get(cls, "#ff3d57")
    return (
        '<div class="sc"><div class="sl">COHERENCE</div>'
        '<div class="gt"><div class="gf" style="width:'
        f'{pct:.0f}%;background:{col}"></div></div>'
        f'<div class="sv {cls}" style="font-size:1.1rem">{coherence_val:.3f}</div>'
        f'<div class="ss">{label}</div></div>'
    )


def _warn_banner(text):
    return f'<div class="warn">&#9888; {text}</div>'


def _live_banner():
    return '<div class="live-banner">&#10003; Identity verified — live person confirmed</div>'


def _reverify_banner(seconds_left):
    return f'<div class="reverify">&#8635; Re-verification required in {seconds_left:.0f}s</div>'


def _session_complete_banner(verdict):
    if verdict == "LIVE":
        return (
            '<div class="session-complete-live">'
            '<div class="sc-title-live">&#10003; SESSION COMPLETE</div>'
            '<div class="sc-sub">Identity verified — Live person confirmed<br>'
            'Download the audit report below &#8595;</div></div>'
        )
    else:
        return (
            '<div class="session-complete-spoof">'
            '<div class="sc-title-spoof">&#9888; SESSION COMPLETE</div>'
            '<div class="sc-sub">Spoof attempt detected<br>'
            'Download the audit report below &#8595;</div></div>'
        )


def render(sr, cr, blr, hdr, txr, mor, pur, tamper_result=None):
    verdict = sr.get("verdict", "CHECKING")
    conf = sr.get("confidence", 0.0)
    thr = sr.get("live_threshold", 0.48)
    tr = sr.get("time_remaining", 20.0)
    tout = sr.get("timeout", 20.0)
    spoof_reason = sr.get("spoof_reason", None)
    coherence_val = sr.get("component_scores", {}).get("coherence", 0.0)

    vph.markdown(_vcard(verdict, conf, thr), unsafe_allow_html=True)
    cph.markdown(_timercard(tr, tout), unsafe_allow_html=True)
    bph.markdown(_ccard(cr), unsafe_allow_html=True)

    if verdict == "SPOOF" and spoof_reason:
        rph.markdown(_rcard(spoof_reason), unsafe_allow_html=True)
    elif verdict == "LIVE":
        rph.markdown(_live_banner(), unsafe_allow_html=True)
    else:
        rph.empty()

    if tamper_result and tamper_result.get("current_warning"):
        tph.markdown(_warn_banner(
            tamper_result["current_warning"]), unsafe_allow_html=True)
    else:
        tph.empty()

    dph.markdown(_bcard(sr), unsafe_allow_html=True)
    blph.markdown(_sc("BLINKS", str(blr.get("blink_count", 0)),
                      f"EAR {blr.get('ear', 0):.3f}", "cyan"), unsafe_allow_html=True)
    hdph.markdown(_sc("HEAD", hdr.get("direction", "---"),
                      f"Yaw {hdr.get('yaw', 0):+.1f}deg", "yellow"), unsafe_allow_html=True)
    txph.markdown(_gc("TEXTURE", txr.get("texture_score", 0), "teal", "#64ffda"),
                  unsafe_allow_html=True)
    moph.markdown(_gc("MOTION", mor.get("score", 0), "cyan", "#00e5ff"),
                  unsafe_allow_html=True)
    pur2 = pur.get("pupil_response", False)
    pus = pur.get("pupil_score", 0.0)
    puph.markdown(_sc("PUPIL", "YES" if pur2 else "....",
                      f"Score {pus:.3f}", "purple"), unsafe_allow_html=True)
    cohph.markdown(_coherence_card(coherence_val), unsafe_allow_html=True)
    eph.markdown(_sc("EAR", f"{blr.get('ear', 0):.3f}",
                     "closed" if blr.get("eye_closed") else "open", "cyan"),
                 unsafe_allow_html=True)
    glare_s = txr.get("glare_score", 1.0)
    if coherence_val > 0.82:
        warnph.markdown(_warn_banner("Rigid motion detected — phone or photo suspected"),
                        unsafe_allow_html=True)
    elif glare_s < 0.35:
        warnph.markdown(_warn_banner("Screen glare detected — phone spoof suspected"),
                        unsafe_allow_html=True)
    else:
        warnph.empty()


def _finish_session(D, verdict):
    graph_path = D["logger"].save()
    st.session_state.graph_path = graph_path
    if REPORTLAB_AVAILABLE:
        try:
            summary = D["logger"].get_summary()
            safe_id = (st.session_state.candidate_id or "NA").replace(
                "/", "-").replace("\\", "-").replace(" ", "_")
            pdf_path = generate_report(
                candidate_name=st.session_state.candidate_name or "Unknown",
                candidate_id=safe_id,
                session_summary=summary,
                graph_path=graph_path,
            )
            st.session_state.pdf_path = pdf_path
        except Exception as e:
            import traceback
            st.session_state.pdf_path = None
            print(f"PDF ERROR: {e}")
            traceback.print_exc()
    if verdict == "LIVE":
        st.session_state.exam_verified = True
        st.session_state.last_verify_time = time.time()
        st.session_state.reverify_count += 1


if st.session_state.running:
    fm = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    cam_index = st.session_state.get("cam_idx", 0)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error(
            f"Cannot access Camera {cam_index}. Try selecting a different camera.")
        st.session_state.running = False
    else:
        D = st.session_state.D
        capph.caption("Live feed active")

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            mpr = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces = mpr.multi_face_landmarks
            face_detected = bool(faces and len(faces) == 1)
            tamper_result = D["tamper"].update(frame, face_detected)

            if faces and len(faces) > 1:
                cv2.putText(frame, "MULTIPLE FACES — SPOOF REJECTED",
                            (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 61, 255), 2)
                mfph.markdown(_warn_banner(
                    "Multiple faces detected — session rejected as spoof"),
                    unsafe_allow_html=True)
                vidph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            channels="RGB", width=800)
                D["scorer"].verdict = "SPOOF"
                D["scorer"].spoof_reason = "Multiple faces detected in frame"
                D["scorer"]._locked = True
                _finish_session(D, "SPOOF")
                st.session_state.running = False
                cap.release()
                fm.close()
                st.rerun()

            mfph.empty()

            if faces:
                fl = faces[0]
                blr = D["blink"].update(fl, w, h)
                hdr = D["head"].update(fl, w, h)
                xs = [lm.x * w for lm in fl.landmark]
                ys = [lm.y * h for lm in fl.landmark]
                bbox = (int(min(xs)) - 5, int(min(ys)) - 5,
                        int(max(xs)) + 5, int(max(ys)) + 5)
                txr = D["texture"].update(frame, bbox)
                mor = D["motion"].update(frame, bbox)
                pur = D["pupil"].update(frame, fl, w, h)
                cr = D["challenge"].update(blr, hdr)
                sr = D["scorer"].update(cr, txr, mor, pur, hdr)
                D["logger"].record(sr, txr, mor, blr, cr, tamper_result)

                if pur.get("flash_active"):
                    apply_flash(frame)
                draw_mesh_dots(frame, fl, w, h)
                draw_face_box(frame, fl, w, h, sr["verdict"])

                st.session_state.last = (
                    sr, cr, blr, hdr, txr, mor, pur, tamper_result)
                render(sr, cr, blr, hdr, txr, mor, pur, tamper_result)

                if sr["verdict"] in ("LIVE", "SPOOF"):
                    _finish_session(D, sr["verdict"])
                    st.session_state.running = False
                    cap.release()
                    fm.close()
                    st.rerun()

            else:
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 100, 255), 2)

            if (st.session_state.exam_mode
                    and st.session_state.exam_verified
                    and st.session_state.last_verify_time):
                elapsed_since = time.time() - st.session_state.last_verify_time
                seconds_left = REVERIFY_INTERVAL - elapsed_since
                if seconds_left <= 0:
                    rvph.markdown(_warn_banner(
                        "Re-verification required — please complete liveness check"),
                        unsafe_allow_html=True)
                    st.session_state.exam_verified = False
                    st.session_state.running = False
                elif seconds_left <= 60:
                    rvph.markdown(_reverify_banner(seconds_left),
                                  unsafe_allow_html=True)
                else:
                    rvph.empty()

            vidph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB", width=800)

        cap.release()
        fm.close()

elif st.session_state.last:
    last = st.session_state.last
    if len(last) == 8:
        render(*last)
    else:
        render(*last, None)

    final_verdict = last[0].get("verdict", "UNKNOWN")

    sesph.markdown(_session_complete_banner(
        final_verdict), unsafe_allow_html=True)

    if st.session_state.pdf_path:
        with open(st.session_state.pdf_path, "rb") as f:
            dlph.download_button(
                label="⬇  Download Audit Report (PDF)",
                data=f,
                file_name=st.session_state.pdf_path.replace(
                    "\\", "/").split("/")[-1],
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        dlph.markdown(
            '<div class="warn">Audit report not available — check terminal for errors</div>',
            unsafe_allow_html=True)

    vidph.markdown(
        '<div style="background:#07131f;border:1px solid #0e2a44;border-radius:10px;'
        'padding:20px;text-align:center;margin-bottom:10px;">'
        '<div style="font-family:Share Tech Mono,monospace;font-size:1.1rem;'
        'color:#aac8e8;letter-spacing:.1em;">SESSION ENDED</div>'
        '<div style="font-size:.8rem;color:#305878;margin-top:6px;">'
        'Press START / NEW for another check</div></div>',
        unsafe_allow_html=True)

    if st.session_state.exam_mode and not st.session_state.exam_verified:
        rvph.markdown(_warn_banner(
            "Re-verification required — press START / NEW to continue exam"),
            unsafe_allow_html=True)

    if st.session_state.graph_path:
        grph.image(st.session_state.graph_path,
                   caption="Session signal timeline",
                   use_container_width=True)

else:
    vph.markdown(
        '<div class="sc"><div class="sl">VERDICT</div>'
        '<div class="sv checking">STANDBY</div></div>',
        unsafe_allow_html=True)
    cph.markdown(
        '<div class="cb"><div class="cs">CHALLENGE</div>'
        '<div class="ci">Press START to begin</div></div>',
        unsafe_allow_html=True)
    vidph.info("Press START / NEW to begin liveness verification.")
