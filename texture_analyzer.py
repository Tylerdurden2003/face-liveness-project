"""
texture_analyzer.py  — Anti-Spoof Texture Analysis
====================================================
Five complementary signals to distinguish live skin from printed photos:

  1. LBP Entropy       — skin has chaotic micro-texture; paper is uniform
  2. FFT Spike Ratio   — photos have moiré/halftone frequency spikes
  3. Chromatic Noise   — live skin has natural colour channel variation;
                         printed photos have flat, correlated RGB channels
  4. Specular Variance — live faces reflect light unevenly (highlights);
                         flat printed photos have uniform brightness
  5. Glare Detection   — phone screens produce hard bright hotspots that
                         real skin never produces; catches screen spoofs
                         that pass specular variance

Preprocessing: CLAHE applied before all signals so low-light conditions
don't tank scores.

All scores 0=spoof, 1=live. Weighted combination → texture_score.
"""

import cv2
import numpy as np


# ── Preprocessing ─────────────────────────────────────────────────────────────

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _preprocess(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to normalise low-light and over-exposed frames."""
    return _CLAHE.apply(gray)


# ── LBP ──────────────────────────────────────────────────────────────────────

def _lbp_entropy(gray: np.ndarray) -> float:
    roi = cv2.resize(gray, (64, 64))
    padded = cv2.copyMakeBorder(roi, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    h, w = roi.shape
    center = padded[1:h+1, 1:w+1].astype(np.float32)
    code = np.zeros((h, w), dtype=np.uint8)
    for bit, (dr, dc) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1),
                                    (1,  1), (1, 0), (1, -1), (0, -1)]):
        n = padded[1+dr:h+1+dr, 1+dc:w+1+dc].astype(np.float32)
        code |= ((n >= center).astype(np.uint8) << bit)
    hist, _ = np.histogram(code.ravel(), bins=256,
                           range=(0, 256), density=True)
    hist = hist + 1e-10
    entropy = -np.sum(hist * np.log2(hist))
    return float(np.clip(entropy / 8.0, 0.0, 1.0))


# ── FFT ──────────────────────────────────────────────────────────────────────

def _fft_score(gray: np.ndarray) -> float:
    roi = cv2.resize(gray, (64, 64)).astype(np.float32)
    fmag = np.abs(np.fft.fftshift(np.fft.fft2(roi)))
    h, w = fmag.shape
    fmag[h//2-2:h//2+3, w//2-2:w//2+3] = 0   # suppress DC
    total = fmag.sum() + 1e-10
    thresh = np.percentile(fmag.ravel(), 99)
    spike_ratio = fmag[fmag >= thresh].sum() / total
    return float(np.clip(1.0 - spike_ratio * 6.0, 0.0, 1.0))


# ── Chromatic Noise ───────────────────────────────────────────────────────────

def _chromatic_score(roi_bgr: np.ndarray) -> float:
    if roi_bgr.shape[0] < 8 or roi_bgr.shape[1] < 8:
        return 0.5
    small = cv2.resize(roi_bgr, (32, 32)).astype(np.float32)
    b = small[:, :, 0].ravel()
    g = small[:, :, 1].ravel()
    r = small[:, :, 2].ravel()

    def safe_corr(a, b):
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return 1.0
        return float(np.corrcoef(a, b)[0, 1])

    corr_rg = abs(safe_corr(r, g))
    corr_rb = abs(safe_corr(r, b))
    corr_gb = abs(safe_corr(g, b))
    avg_corr = (corr_rg + corr_rb + corr_gb) / 3.0
    return float(np.clip(1.0 - avg_corr, 0.0, 1.0))


# ── Specular Variance ─────────────────────────────────────────────────────────

def _specular_score(gray: np.ndarray) -> float:
    roi = cv2.resize(gray, (64, 64)).astype(np.float32)
    threshold = np.percentile(roi, 85)
    highlights = (roi > threshold).astype(np.float32)
    cell = 16
    densities = []
    for i in range(4):
        for j in range(4):
            patch = highlights[i*cell:(i+1)*cell, j*cell:(j+1)*cell]
            densities.append(patch.mean())
    variance = float(np.var(densities))
    return float(np.clip(variance * 25.0, 0.0, 1.0))


# ── Glare Detection ───────────────────────────────────────────────────────────

def _glare_score(gray: np.ndarray, roi_bgr: np.ndarray) -> float:
    """
    Phone screens produce small, very intense, sharply-bounded hotspots.
    Real skin: bright areas are broad and gradually transition.
    Real skin: RGB channels stay balanced even in bright areas.

    Two sub-checks:
      A) Hotspot compactness — screen glare is small and intense;
         skin highlights are diffuse. If tiny saturated blobs exist → spoof.
      B) Channel saturation imbalance — screen glare blows out one or two
         channels unevenly; skin highlights stay roughly balanced.

    Returns 1.0 (live) when no suspicious glare found, 0.0 (spoof) when
    strong screen-like glare is present.
    """
    roi_gray = cv2.resize(gray, (64, 64)).astype(np.float32)

    # ── Sub-check A: hotspot compactness ─────────────────────────────
    saturated = (roi_gray > 245).astype(np.uint8)
    total_pixels = roi_gray.size
    saturated_count = int(saturated.sum())

    # Find connected saturated blobs
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        saturated, connectivity=8)

    compactness_penalty = 0.0
    for i in range(1, num_labels):   # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        # Small intense blob = screen glare signature
        # Real skin highlights are large and diffuse
        if area < total_pixels * 0.03 and area > 2:
            compactness_penalty += 1.0

    # Normalise — more than 3 small hotspots is very suspicious
    compactness_spoof = float(np.clip(compactness_penalty / 3.0, 0.0, 1.0))

    # ── Sub-check B: channel saturation imbalance ─────────────────────
    small_bgr = cv2.resize(roi_bgr, (64, 64)).astype(np.float32)
    bright_mask = roi_gray > 200
    channel_imbalance = 0.0
    if bright_mask.sum() > 10:
        b_bright = small_bgr[:, :, 0][bright_mask]
        g_bright = small_bgr[:, :, 1][bright_mask]
        r_bright = small_bgr[:, :, 2][bright_mask]
        means = np.array([b_bright.mean(), g_bright.mean(), r_bright.mean()])
        # High std across channel means = one channel blown out = screen
        channel_imbalance = float(np.clip(np.std(means) / 60.0, 0.0, 1.0))

    # Combine — both sub-checks must agree to be confident
    glare_spoof_score = 0.6 * compactness_spoof + 0.4 * channel_imbalance

    # Return liveness score: high glare_spoof → low liveness
    return float(np.clip(1.0 - glare_spoof_score, 0.0, 1.0))


# ── Combined Analyzer ─────────────────────────────────────────────────────────

class TextureAnalyzer:
    def __init__(self, history_len=20, live_threshold=0.52):
        self.live_threshold = live_threshold
        self._history = []
        self._maxlen = history_len

    def update(self, frame_bgr: np.ndarray, face_bbox: tuple) -> dict:
        x1, y1, x2, y2 = [max(0, int(v)) for v in face_bbox]
        h_f, w_f = frame_bgr.shape[:2]
        x2 = min(w_f, x2)
        y2 = min(h_f, y2)
        roi_bgr = frame_bgr[y1:y2, x1:x2]

        if roi_bgr.size == 0 or roi_bgr.shape[0] < 8 or roi_bgr.shape[1] < 8:
            return self._result(0.0, 0.0, 0.5, 0.5, 0.5)

        gray_raw = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = _preprocess(gray_raw)   # CLAHE normalised

        lbp_s = _lbp_entropy(gray)
        fft_s = _fft_score(gray)
        chrom_s = _chromatic_score(roi_bgr)
        spec_s = _specular_score(gray)
        glare_s = _glare_score(gray_raw, roi_bgr)  # use raw gray for glare

        # Weighted combination
        # Chromatic still strongest; glare added as meaningful contributor
        combined = (0.28 * lbp_s +
                    0.22 * fft_s +
                    0.28 * chrom_s +
                    0.12 * spec_s +
                    0.10 * glare_s)

        self._history.append(combined)
        if len(self._history) > self._maxlen:
            self._history.pop(0)

        return self._result(lbp_s, fft_s, chrom_s, spec_s, glare_s)

    def _result(self, lbp_s, fft_s, chrom_s, spec_s, glare_s) -> dict:
        avg = float(np.mean(self._history)) if self._history else 0.0
        return {
            "lbp_score":     round(lbp_s,   4),
            "fft_score":     round(fft_s,   4),
            "chrom_score":   round(chrom_s, 4),
            "spec_score":    round(spec_s,  4),
            "glare_score":   round(glare_s, 4),
            "texture_score": round(avg,     4),
            "texture_live":  avg >= self.live_threshold,
        }

    def reset(self):
        self._history.clear()
