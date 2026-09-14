"""Multi-scale morphological lesion segmentation -- the 'where' channel.

This stage is deliberately NOT a neural network. It is deterministic,
inspectable classical image processing, so every lesion the system reports can
be traced back to a specific morphological response at a specific scale. That
is what lets a clinician audit the machine rather than trust it.

Lesion classes follow the ICDR grading vocabulary:
  MA  microaneurysm       small round dark, under 125 um
  HEM haemorrhage         larger irregular dark
  EX  hard exudate        bright, sharp-edged lipid deposit
  CWS cotton-wool spot    bright, fuzzy-edged nerve-fibre infarct

MATLAB equivalents: imtophat / imbothat with strel disk, imreconstruct,
fibermetric, and regionprops for Area, Circularity, Centroid, MajorAxisLength.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from skimage.filters import frangi

from .preprocess import flatten_illumination

# Radii in pixels at the 512px working resolution (~25 um/px).
DARK_SCALES = (2, 4, 7, 11)
BRIGHT_SCALES = (3, 6, 10)
MA_MAX_DIAMETER_UM = 125.0

# Anatomical constants as a fraction of the field-of-view width. The fovea sits
# about 2.5 disc diameters temporal to the optic disc; the band is widened to
# cover normal inter-patient variation. Expressing these against the FOV rather
# than against the ESTIMATED disc radius matters -- the radius estimate carries
# its own error, and compounding it moved the search annulus off the macula
# entirely.
DISC_RADIUS_FRAC = 0.09
MACULA_DIST_FRAC = (0.20, 0.42)
MACULA_TEMPORAL_COS = 0.77          # accept within ~40 degrees of temporal
# Mean edge-gradient above which a bright lesion is called a hard exudate.
EXUDATE_EDGE_SHARPNESS = 90.0

# Detection thresholds, expressed as multiples of the robust noise sigma (MAD)
# of the top-hat response, with an absolute floor in grey levels.
DARK_K, DARK_FLOOR, DARK_MIN_PX = 5.0, 10.0, 2
BRIGHT_K, BRIGHT_FLOOR, BRIGHT_MIN_PX = 13.0, 14.0, 6
VESSEL_DILATE = 2
DARK_OPEN_RADIUS = 0     # microaneurysms are only a few px across
BRIGHT_OPEN_RADIUS = 1   # exudates are larger; opening trims mottling

# A dark component is discarded as a vessel fragment only if it is BOTH mostly
# covered by the vessel map AND not compact. Deleting every vessel pixel
# outright removes ~40% of genuine microaneurysms, which sit on the capillary
# bed by definition; shape is what separates a lesion from a vessel segment.
# The circularity value is a minimum-enclosing-circle fill ratio (see
# _shape_stats), so it is tuned against that scale, not against 4*pi*A/P^2.
VESSEL_OVERLAP_REJECT = 0.70
VESSEL_CIRCULARITY_KEEP = 0.85


@dataclass
class Lesion:
    kind: str
    x: int
    y: int
    area_um2: float
    major_axis_um: float
    minor_axis_um: float
    circularity: float
    contrast: float
    dist_to_macula_um: float

    def to_dict(self):
        d = asdict(self)
        for k in ("area_um2", "major_axis_um", "minor_axis_um", "circularity",
                  "contrast", "dist_to_macula_um"):
            d[k] = round(float(d[k]), 2)
        return d


@dataclass
class LesionMap:
    lesions: list
    vessels: np.ndarray
    optic_disc: tuple          # (x, y, radius_px)
    macula: tuple              # (x, y)
    overlays: dict             # kind -> binary uint8 mask

    def counts(self):
        c = {"MA": 0, "HEM": 0, "EX": 0, "CWS": 0}
        for l in self.lesions:
            c[l.kind] += 1
        return c

    def burden(self):
        """Total lesion area per class, in square microns."""
        b = {"MA": 0.0, "HEM": 0.0, "EX": 0.0, "CWS": 0.0}
        for l in self.lesions:
            b[l.kind] += l.area_um2
        return b

    def macula_involved(self, radius_um=1500.0):
        """Lesions inside the fovea-centred circle drive clinically significant
        macular oedema risk, which escalates referral urgency independent of
        the overall severity grade."""
        return [l for l in self.lesions if l.dist_to_macula_um <= radius_um]


def _strel(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def segment_vessels(green, mask):
    """Frangi vesselness. Vessels are dark and elongated, so they alias into the
    dark-lesion top-hat; they must be removed before haemorrhage counting."""
    inv = 255 - green
    v = frangi(inv.astype(np.float32) / 255.0, sigmas=range(1, 6),
               black_ridges=False)
    if v.max() > 0:
        v = v / v.max()
    ves = (v > 0.08).astype(np.uint8)
    ves = cv2.morphologyEx(ves, cv2.MORPH_CLOSE, _strel(2))
    return cv2.bitwise_and(ves, ves, mask=mask)


def locate_optic_disc(bgr, mask):
    """Brightest disc-sized region in the red channel.

    The smoothing scale must match the optic disc, not a lesion. Blurring at a
    small sigma leaves a tight cluster of hard exudates brighter than the disc
    itself, and the detector then places the disc on the lesions -- which in
    turn drives the macula estimate, the exclusion zone and every
    distance-to-fovea measurement off the same error.
    """
    radius = int(DISC_RADIUS_FRAC * max(mask.shape))
    red = cv2.bitwise_and(bgr[:, :, 2], bgr[:, :, 2], mask=mask).astype(np.float32)
    m = mask.astype(np.float32)

    def smooth_at(sigma):
        # Normalised convolution. Blurring the masked image alone averages the
        # black surround into every pixel near the rim, dimming it -- and the
        # optic disc usually sits near the rim, so a plain blur hides the very
        # thing being looked for.
        num = cv2.GaussianBlur(red, (0, 0), sigma)
        den = cv2.GaussianBlur(m, (0, 0), sigma)
        return np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)

    # Absolute brightness is the wrong signal: a healthy retina is brightest at
    # the posterior pole, so the global maximum lands mid-frame and the disc is
    # missed in almost half of images. What identifies the disc is being bright
    # RELATIVE TO ITS SURROUNDINGS at its own scale, so the broad background is
    # subtracted first.
    response = smooth_at(radius * 0.6) - smooth_at(radius * 3.0)
    response[mask == 0] = -1e9
    _, _, _, loc = cv2.minMaxLoc(response)
    return (int(loc[0]), int(loc[1]), radius)


def locate_macula(green, mask, disc):
    """Darkest region on the temporal side of the disc, ~2-3 disc diameters out.

    Direction is as important as distance. Searching the whole annulus and
    taking the darkest pixel finds the vignetted periphery every time -- the
    edge of the retina is far darker than the fovea -- which put the macula
    ~260 px from truth and corrupted every distance-to-fovea measurement.

    The fovea lies temporal to the disc, i.e. toward the centre of the frame,
    and close to the same height. Constraining the search to a wedge in that
    direction is what makes the estimate track the real anatomy.
    """
    h, w = green.shape
    dx, dy, dr = disc
    smooth = cv2.GaussianBlur(green.astype(np.float32), (0, 0), 14)
    yy, xx = np.mgrid[0:h, 0:w]
    vx, vy = (xx - dx).astype(np.float32), (yy - dy).astype(np.float32)
    dist = np.sqrt(vx ** 2 + vy ** 2) + 1e-6

    # Unit vector from the disc toward the frame centre = the temporal direction.
    tx, ty = (w / 2.0 - dx), (h / 2.0 - dy)
    tnorm = float(np.hypot(tx, ty))
    if tnorm < 1e-6:
        tx, ty, tnorm = 1.0, 0.0, 1.0
    cos = (vx * (tx / tnorm) + vy * (ty / tnorm)) / dist

    lo, hi = (f * max(h, w) for f in MACULA_DIST_FRAC)
    band = ((dist > lo) & (dist < hi)
            & (cos > MACULA_TEMPORAL_COS)
            & (mask > 0))
    if not band.any():
        # Fall back to the expected anatomical position rather than to the
        # frame centre, which could sit anywhere relative to the disc.
        step = 0.5 * sum(MACULA_DIST_FRAC) * max(h, w)
        fx = int(np.clip(dx + step * tx / tnorm, 0, w - 1))
        fy = int(np.clip(dy + step * ty / tnorm, 0, h - 1))
        return (fx, fy)
    cand = np.where(band, smooth, np.inf)
    idx = int(np.argmin(cand))
    return (idx % w, idx // w)


def _multiscale_tophat(img, scales, dark=True):
    """Keep the strongest response across scales, so both a 50 um microaneurysm
    and a 400 um blot haemorrhage survive the same pass."""
    op = cv2.MORPH_BLACKHAT if dark else cv2.MORPH_TOPHAT
    acc = np.zeros(img.shape, np.float32)
    for r in scales:
        th = cv2.morphologyEx(img, op, _strel(r)).astype(np.float32)
        acc = np.maximum(acc, th)
    return acc


def _shape_stats(comp_mask):
    """Return (circularity, major_axis_px, minor_axis_px).

    Circularity is the fraction of the minimum enclosing circle that the
    component actually fills, clamped to [0, 1]. The textbook 4*pi*A/P^2 is not
    usable here: microaneurysms are only a few pixels across, and at that size
    the discrete perimeter is so short that the ratio explodes -- a 2-pixel blob
    scores pi, well above the 1.0 a perfect disc should give. Since circularity
    is what separates a microaneurysm from a haemorrhage, that turned every
    degenerate speck into a confident microaneurysm.

    The fill ratio degrades gracefully instead: a compact blob approaches 1,
    an elongated vessel fragment falls toward 0.2-0.4, and both stay bounded
    at any size.
    """
    cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0, 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    area_px = float(comp_mask.sum())

    (_, _), radius = cv2.minEnclosingCircle(c)
    circle_area = np.pi * max(radius, 0.5) ** 2
    circularity = float(np.clip(area_px / circle_area, 0.0, 1.0))

    if len(c) >= 5:
        (_, _), (ax1, ax2), _ = cv2.fitEllipse(c)
        major, minor = max(ax1, ax2), min(ax1, ax2)
        # fitEllipse can return a degenerate zero axis on collinear points.
        if minor < 1e-3:
            major = minor = float(np.sqrt(max(area_px, 1.0)))
    else:
        major = minor = float(np.sqrt(max(area_px, 1.0)))
    return circularity, float(major), float(minor)


def _threshold(response, region, k, floor):
    """Robust absolute threshold: median + k * MAD of the top-hat response.

    A percentile threshold would be wrong here -- it declares a fixed fraction
    of pixels to be lesions no matter what, so a perfectly healthy retina still
    returns a full lesion inventory. Anchoring to the noise floor instead lets
    a grade-0 eye legitimately return zero detections.
    """
    vals = response[region > 0].astype(np.float32)
    if vals.size == 0:
        return np.zeros(response.shape, np.uint8)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826   # -> sigma equivalent
    thr = max(med + k * max(mad, 1.0), floor)
    return (response > thr).astype(np.uint8)


def detect(prep) -> LesionMap:
    green, mask, upp = prep.green, prep.mask, prep.microns_per_px
    flat = flatten_illumination(green, mask)
    vessels = segment_vessels(green, mask)
    disc = locate_optic_disc(prep.bgr, mask)
    macula = locate_macula(green, mask, disc)

    disc_mask = np.zeros_like(mask)
    cv2.circle(disc_mask, (disc[0], disc[1]), int(disc[2] * 1.4), 1, -1)
    # Keep clear of the rim by more than the largest structuring element, so no
    # detection is an artefact of the field-of-view boundary.
    margin = max(DARK_SCALES + BRIGHT_SCALES) + 4
    interior = cv2.erode(mask, _strel(margin))
    valid = cv2.bitwise_and(interior, 1 - disc_mask)
    # Frangi under-covers vessel edges, so the map is dilated before it is used
    # to judge overlap -- but it is used to JUDGE, not to erase. See below.
    vessel_wide = cv2.dilate(vessels, _strel(VESSEL_DILATE))

    lesions = []
    layers = {k: np.zeros(mask.shape, np.uint8) for k in ("MA", "HEM", "EX", "CWS")}

    def dist_to_macula(cx, cy):
        return float(np.hypot(cx - macula[0], cy - macula[1]) * upp)

    # ---- dark lesions: microaneurysms and haemorrhages ----
    dark = _multiscale_tophat(flat, DARK_SCALES, dark=True)
    dark = cv2.bitwise_and(dark, dark, mask=valid)
    # Threshold statistics are computed off-vessel so the vessel population does
    # not inflate the noise estimate, but detection itself runs over all of the
    # valid retina.
    off_vessel = cv2.bitwise_and(valid, 1 - vessel_wide)
    dbin = _threshold(dark, off_vessel, DARK_K, DARK_FLOOR)
    dbin = cv2.bitwise_and(dbin, valid)
    if DARK_OPEN_RADIUS > 0:
        dbin = cv2.morphologyEx(dbin, cv2.MORPH_OPEN, _strel(DARK_OPEN_RADIUS))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(dbin, 8)
    for i in range(1, n):
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        if area_px < DARK_MIN_PX:
            continue
        comp = (labels == i).astype(np.uint8)
        circ, major, minor = _shape_stats(comp)
        # Vessel-fragment rejection: only discard things that are both sitting
        # on a vessel and shaped like one.
        overlap = float((vessel_wide[comp > 0] > 0).mean())
        if overlap > VESSEL_OVERLAP_REJECT and circ < VESSEL_CIRCULARITY_KEEP:
            continue
        major_um = major * upp
        cx, cy = int(cents[i][0]), int(cents[i][1])
        # ICDR: a microaneurysm is round and under ~125 um across; anything
        # larger or irregular is counted as a haemorrhage.
        kind = "MA" if (major_um <= MA_MAX_DIAMETER_UM and circ > 0.55) else "HEM"
        layers[kind][labels == i] = 1
        lesions.append(Lesion(
            kind=kind, x=cx, y=cy,
            area_um2=area_px * prep.um2_per_px,
            major_axis_um=major_um, minor_axis_um=minor * upp,
            circularity=circ, contrast=float(dark[comp > 0].mean()),
            dist_to_macula_um=dist_to_macula(cx, cy),
        ))

    # ---- bright lesions: hard exudates and cotton-wool spots ----
    bright = _multiscale_tophat(flat, BRIGHT_SCALES, dark=False)
    bright = cv2.bitwise_and(bright, bright, mask=valid)
    bbin = _threshold(bright, valid, BRIGHT_K, BRIGHT_FLOOR)
    bbin = cv2.morphologyEx(bbin, cv2.MORPH_OPEN, _strel(BRIGHT_OPEN_RADIUS))
    # Edge sharpness separates lipid exudates (crisp) from CWS infarcts (fuzzy).
    gx, gy = cv2.spatialGradient(flat)
    grad = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(bbin, 8)
    for i in range(1, n):
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        if area_px < BRIGHT_MIN_PX:
            continue
        comp = (labels == i).astype(np.uint8)
        circ, major, minor = _shape_stats(comp)
        edge = cv2.dilate(comp, _strel(1)) - cv2.erode(comp, _strel(1))
        sharpness = float(grad[edge > 0].mean()) if (edge > 0).any() else 0.0
        cx, cy = int(cents[i][0]), int(cents[i][1])
        kind = "EX" if sharpness > EXUDATE_EDGE_SHARPNESS else "CWS"
        layers[kind][labels == i] = 1
        lesions.append(Lesion(
            kind=kind, x=cx, y=cy,
            area_um2=area_px * prep.um2_per_px,
            major_axis_um=major * upp, minor_axis_um=minor * upp,
            circularity=circ, contrast=float(bright[comp > 0].mean()),
            dist_to_macula_um=dist_to_macula(cx, cy),
        ))

    overlays = dict(layers)
    overlays["vessels"] = vessels
    return LesionMap(lesions=lesions, vessels=vessels, optic_disc=disc,
                     macula=macula, overlays=overlays)
