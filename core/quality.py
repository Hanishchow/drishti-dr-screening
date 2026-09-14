"""Image quality gate.

Runs before any analysis. An unusable fundus photo must be rejected at the
point of capture, while the patient is still in the chair -- a wrong grade on a
blurred image is worse than no grade at all.

MATLAB equivalents: var(imfilter(I,fspecial('laplacian'))), stdfilt, im2double.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

# Thresholds are in the units produced by the metric functions below. Sharpness
# is a mid-band structure ratio x1000: a well-focused fundus image scores ~225,
# a mildly soft one ~205, and visible blur falls below ~170.
SHARPNESS_MIN = 155.0
ILLUM_UNIFORMITY_MIN = 0.45
FOV_COVERAGE_MIN = 0.25
FOV_COVERAGE_MAX = 0.95
CLIPPED_HIGHLIGHT_MAX = 0.015
MEAN_LUMA_MIN, MEAN_LUMA_MAX = 28.0, 215.0


@dataclass
class QualityReport:
    passed: bool
    score: float                      # 0-100, for the ASHA-facing dial
    metrics: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    guidance: str = ""

    def to_dict(self):
        return {
            "passed": self.passed,
            "score": round(self.score, 1),
            "metrics": {k: round(float(v), 4) for k, v in self.metrics.items()},
            "failures": self.failures,
            "guidance": self.guidance,
        }


def field_of_view_mask(bgr):
    """Segment the illuminated retinal circle from the black surround."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Otsu on a blurred copy is robust to the vignette edge being soft.
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    thr = max(10, int(0.5 * np.percentile(blur[blur > 0], 60)) if (blur > 0).any() else 10)
    mask = (blur > thr).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:  # keep the largest blob, discarding reflections and text overlays
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    return mask


def _interior(mask, fov_radius):
    """Mask shrunk away from the vignette boundary.

    The FOV rim is the strongest edge in the frame by a wide margin. Left in, it
    dominates every focus statistic, and a badly blurred retina still scores as
    sharp because the rim is still a hard edge.
    """
    r = max(3, int(0.06 * fov_radius))
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))


def _sharpness(gray, interior, fov_radius):
    """Mid-band structure ratio -- an exposure-invariant focus measure.

    Two approaches fail here and are worth naming, because both are the obvious
    thing to reach for:

      * Laplacian variance (raw or contrast-normalised) is not monotonic in
        blur. Normalising by image variance divides out the very signal being
        measured, so an underexposed frame scores as extremely sharp.
      * A high-band / mid-band energy ratio inverts. Sensor noise is added
        AFTER the optical blur in a real camera, so a blurred frame keeps its
        full noise floor in the high band and scores as sharper than a crisp one.

    Measuring mid-scale structure (2-8 px, the scale of vessels and lesions)
    against overall retinal contrast avoids both. Blur destroys mid-band
    structure while leaving low-frequency contrast intact, and exposure scales
    numerator and denominator together so it cancels.
    """
    sel = interior > 0
    if sel.sum() < 64:
        return 0.0
    # Light pre-blur suppresses per-pixel sensor noise without touching the
    # 2-8 px band being measured.
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.8)
    scale = max(fov_radius, 1.0) / 256.0
    mid = (cv2.GaussianBlur(g, (0, 0), 2.0 * scale)
           - cv2.GaussianBlur(g, (0, 0), 8.0 * scale))
    contrast = float(g[sel].std())
    if contrast < 1e-3:
        return 0.0
    return float(mid[sel].std()) / contrast * 1000.0


def _illumination_uniformity(gray, mask):
    """1.0 = evenly lit; falls toward 0 as one side of the retina drops into
    shadow.

    Measures DIRECTIONAL imbalance only. Every fundus image has strong radial
    falloff toward the periphery -- that is normal optics, not a capture fault,
    and a plain min/max or decile ratio flags every healthy image because of it.
    Fitting a plane to the low-frequency luminance isolates the one-sided
    shadow that actually warrants a recapture.
    """
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST) > 0
    if m.sum() < 64:
        return 0.0
    lowfreq = cv2.GaussianBlur(small, (0, 0), 6)
    ys, xs = np.where(m)
    z = lowfreq[m]
    mean = float(z.mean())
    if mean < 1e-6:
        return 0.0
    # Least-squares plane z = a*x + b*y + c over the retinal area.
    A = np.stack([xs - xs.mean(), ys - ys.mean(), np.ones_like(xs)], axis=1).astype(np.float32)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    # Peak-to-peak of the fitted tilt across the retina, relative to brightness.
    span = float(abs(coef[0]) * (xs.max() - xs.min()) + abs(coef[1]) * (ys.max() - ys.min()))
    return float(np.clip(1.0 - span / mean, 0.0, 1.0))


def assess(bgr) -> QualityReport:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = field_of_view_mask(bgr)
    coverage = float(mask.sum()) / (h * w)
    fov_radius = float(np.sqrt(max(mask.sum(), 1) / np.pi))

    interior = _interior(mask, fov_radius)
    sharp = _sharpness(gray, interior, fov_radius)
    uniformity = _illumination_uniformity(gray, mask)
    inside = gray[interior > 0] if interior.any() else gray[mask > 0]
    mean_luma = float(inside.mean()) if inside.size else 0.0
    clipped = float((inside >= 250).mean()) if inside.size else 1.0

    metrics = {
        "sharpness": sharp,
        "illumination_uniformity": uniformity,
        "fov_coverage": coverage,
        "mean_luminance": mean_luma,
        "clipped_highlight_fraction": clipped,
    }

    failures, guidance = [], []
    if coverage < FOV_COVERAGE_MIN:
        failures.append("fov_too_small")
        guidance.append("Retina fills too little of the frame - move the camera closer.")
    elif coverage > FOV_COVERAGE_MAX:
        failures.append("fov_cropped")
        guidance.append("Frame is cropped - pull back so the whole circle is visible.")
    # Glare is checked before blur: a saturated blob destroys mid-band structure
    # and so also trips the blur test, where "refocus" would be useless advice.
    # Clipping with a normal overall exposure is a localised reflection; clipping
    # with high overall exposure is simply too much flash, and gets the
    # overexposure message below instead.
    localised_glare = clipped > CLIPPED_HIGHLIGHT_MAX and mean_luma <= MEAN_LUMA_MAX * 0.8
    if localised_glare:
        failures.append("glare")
        guidance.append("Lens glare or reflection detected - tilt the camera "
                        "slightly off-axis and recapture.")
    elif sharp < SHARPNESS_MIN:
        failures.append("blurred")
        guidance.append("Image is blurred - hold steady and refocus, then recapture.")
    if uniformity < ILLUM_UNIFORMITY_MIN:
        failures.append("uneven_illumination")
        guidance.append("One side is in shadow - centre the flash on the pupil.")
    if mean_luma < MEAN_LUMA_MIN:
        failures.append("underexposed")
        guidance.append("Too dark - increase flash or dilate the pupil further.")
    elif mean_luma > MEAN_LUMA_MAX or (clipped > CLIPPED_HIGHLIGHT_MAX
                                       and not localised_glare):
        failures.append("overexposed")
        guidance.append("Too bright - reduce flash intensity.")

    # Score blends the normalised sub-metrics; it drives the capture-screen dial.
    subscores = [
        min(1.0, sharp / (SHARPNESS_MIN * 1.45)),
        min(1.0, uniformity / 0.8),
        1.0 if FOV_COVERAGE_MIN <= coverage <= FOV_COVERAGE_MAX else 0.3,
        1.0 - min(1.0, clipped / 0.2),
        1.0 if MEAN_LUMA_MIN <= mean_luma <= MEAN_LUMA_MAX else 0.3,
    ]
    score = 100.0 * float(np.mean(subscores))

    return QualityReport(
        passed=not failures,
        score=score,
        metrics=metrics,
        failures=failures,
        guidance=" ".join(guidance) or "Image quality acceptable for grading.",
    )
