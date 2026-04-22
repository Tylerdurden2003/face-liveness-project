"""
session_logger.py
Records per-frame signal data during a liveness session and
produces a matplotlib timeline graph after the session ends.
Saves logs as JSON and graphs as PNG in a /session_logs folder.
Now includes tamper events and confidence score history.
"""

import json
import os
import time
from datetime import datetime
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

LOGS_DIR = "session_logs"


class SessionLogger:
    def __init__(self):
        self._frames = []
        self._tamper_events = []
        self._start_time = time.time()
        self._verdict = "CHECKING"
        self._spoof_reason = None
        os.makedirs(LOGS_DIR, exist_ok=True)

    def record(self, sr, txr, mor, blr, cr, tamper_result=None):
        """Call once per frame with result dicts from each module."""
        elapsed = time.time() - self._start_time
        self._frames.append({
            "t":            round(elapsed, 2),
            "confidence":   sr.get("confidence",    0.0),
            "texture":      txr.get("texture_score", 0.0),
            "motion":       mor.get("score",         0.0),
            "coherence":    mor.get("coherence",     0.0),
            "glare":        txr.get("glare_score",   1.0),
            "challenge":    sr.get("component_scores", {}).get("challenge", 0.0),
            "steps_done":   cr.get("steps_done",     0),
            "verdict":      sr.get("verdict",        "CHECKING"),
            "brightness":   tamper_result.get("mean_brightness", 0.0)
                            if tamper_result else 0.0,
                            "tamper_flag":  tamper_result.get("tamper_flag", False)
                            if tamper_result else False,
                            })
        self._verdict = sr.get("verdict",      "CHECKING")
        self._spoof_reason = sr.get("spoof_reason", None)

        if tamper_result:
            for evt in tamper_result.get("events", []):
                if evt not in self._tamper_events:
                    self._tamper_events.append(evt)

    def save(self):
        """Save JSON log and generate PNG graph. Returns graph path or None."""
        if not self._frames:
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        verdict = self._verdict
        base = f"{LOGS_DIR}/session_{ts}_{verdict}"
        json_path = base + ".json"
        graph_path = base + ".png"

        # ── Save JSON ─────────────────────────────────────────────────
        payload = {
            "timestamp":     ts,
            "verdict":       verdict,
            "spoof_reason":  self._spoof_reason,
            "duration_s":    self._frames[-1]["t"] if self._frames else 0,
            "tamper_events": self._tamper_events,
            "frames":        self._frames,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        if not MATPLOTLIB_AVAILABLE:
            return None

        # ── Build graph ───────────────────────────────────────────────
        times = [f["t"] for f in self._frames]
        confidence = [f["confidence"] for f in self._frames]
        texture = [f["texture"] for f in self._frames]
        motion = [f["motion"] for f in self._frames]
        coherence = [f["coherence"] for f in self._frames]
        glare = [f["glare"] for f in self._frames]
        challenge = [f["challenge"] for f in self._frames]
        brightness = [f["brightness"] for f in self._frames]
        tamper_mask = [f["tamper_flag"] for f in self._frames]

        lock_time = None
        lock_color = "#00e676"
        for f in self._frames:
            if f["verdict"] in ("LIVE", "SPOOF"):
                lock_time = f["t"]
                lock_color = "#00e676" if f["verdict"] == "LIVE" else "#ff3d57"
                break

        fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
        fig.patch.set_facecolor("#030810")
        for ax in axes:
            ax.set_facecolor("#07131f")
            ax.tick_params(colors="#aac8e8", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#0e2a44")
            ax.yaxis.label.set_color("#aac8e8")
            ax.xaxis.label.set_color("#aac8e8")

        # ── Plot 1: Confidence vs threshold ───────────────────────────
        ax1 = axes[0]
        ax1.plot(times, confidence, color="#00e5ff",
                 linewidth=1.8, label="Confidence")
        ax1.axhline(y=0.48, color="#ffab40", linewidth=1,
                    linestyle="--", label="Threshold (0.48)")
        ax1.fill_between(times, confidence, 0,
                         alpha=0.12, color="#00e5ff")
        # Shade tamper regions
        for i, t in enumerate(tamper_mask):
            if t and i > 0:
                ax1.axvspan(times[i-1], times[i],
                            alpha=0.15, color="#ff3d57")
        if lock_time:
            ax1.axvline(x=lock_time, color=lock_color,
                        linewidth=2, linestyle="--",
                        label=f"{verdict} at {lock_time:.1f}s")
        ax1.set_ylabel("Confidence", fontsize=9)
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc="upper left", fontsize=8,
                   facecolor="#07131f", labelcolor="#aac8e8",
                   edgecolor="#0e2a44")
        title = f"Session — {verdict}"
        if self._spoof_reason:
            title += f"  |  {self._spoof_reason}"
        ax1.set_title(title, color="#00e5ff", fontsize=10, pad=8)

        # ── Plot 2: Texture + Glare + Challenge ───────────────────────
        ax2 = axes[1]
        ax2.plot(times, texture,   color="#64ffda",
                 linewidth=1.5, label="Texture")
        ax2.plot(times, glare,     color="#ce93d8",
                 linewidth=1.5, label="Glare")
        ax2.plot(times, challenge, color="#ffe066",
                 linewidth=1.5, label="Challenge")
        ax2.axhline(y=0.36, color="#ff3d57", linewidth=1,
                    linestyle=":", label="Texture gate (0.36)")
        if lock_time:
            ax2.axvline(x=lock_time, color=lock_color,
                        linewidth=2, linestyle="--")
        ax2.set_ylabel("Score", fontsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="upper left", fontsize=8,
                   facecolor="#07131f", labelcolor="#aac8e8",
                   edgecolor="#0e2a44")

        # ── Plot 3: Motion + Coherence ────────────────────────────────
        ax3 = axes[2]
        ax3.plot(times, motion,    color="#00e5ff",
                 linewidth=1.5, label="Motion")
        ax3.plot(times, coherence, color="#ff3d57",
                 linewidth=1.5, label="Coherence")
        ax3.axhline(y=0.82, color="#ff3d57", linewidth=1,
                    linestyle=":", label="Coherence gate (0.82)")
        if lock_time:
            ax3.axvline(x=lock_time, color=lock_color,
                        linewidth=2, linestyle="--")
        ax3.set_ylabel("Score", fontsize=9)
        ax3.set_ylim(0, 1.05)
        ax3.legend(loc="upper left", fontsize=8,
                   facecolor="#07131f", labelcolor="#aac8e8",
                   edgecolor="#0e2a44")

        # ── Plot 4: Brightness + tamper events ────────────────────────
        ax4 = axes[3]
        ax4.plot(times, brightness, color="#ffe066",
                 linewidth=1.5, label="Brightness")
        ax4.axhline(y=DARKNESS_THRESHOLD if True else 18,
                    color="#ff3d57", linewidth=1,
                    linestyle=":", label="Darkness gate (18)")
        # Mark tamper events as vertical lines
        for evt in self._tamper_events:
            ax4.axvline(x=evt["elapsed"], color="#ff3d57",
                        linewidth=1.5, linestyle="--", alpha=0.8)
            ax4.text(evt["elapsed"] + 0.1, 20,
                     evt["type"].replace("_", " "),
                     color="#ff3d57", fontsize=7, rotation=90)
        if lock_time:
            ax4.axvline(x=lock_time, color=lock_color,
                        linewidth=2, linestyle="--")
        ax4.set_ylabel("Brightness", fontsize=9)
        ax4.set_xlabel("Time (seconds)", fontsize=9,
                       color="#aac8e8")
        ax4.legend(loc="upper left", fontsize=8,
                   facecolor="#07131f", labelcolor="#aac8e8",
                   edgecolor="#0e2a44")

        plt.tight_layout()
        plt.savefig(graph_path, dpi=120,
                    facecolor="#030810", bbox_inches="tight")
        plt.close(fig)

        return graph_path

    def get_confidence_history(self):
        """Returns list of (time, confidence) tuples for audit report."""
        return [(f["t"], f["confidence"]) for f in self._frames]

    def get_summary(self):
        """Returns summary dict for audit report."""
        if not self._frames:
            return {}
        confs = [f["confidence"] for f in self._frames]
        return {
            "verdict":        self._verdict,
            "spoof_reason":   self._spoof_reason,
            "duration_s":     self._frames[-1]["t"],
            "total_frames":   len(self._frames),
            "avg_confidence": round(float(np.mean(confs)), 4),
            "min_confidence": round(float(np.min(confs)),  4),
            "max_confidence": round(float(np.max(confs)),  4),
            "tamper_events":  self._tamper_events,
        }

    def reset(self):
        self._frames = []
        self._tamper_events = []
        self._start_time = time.time()
        self._verdict = "CHECKING"
        self._spoof_reason = None


DARKNESS_THRESHOLD = 18
