"""Synthetic fundus generator.

Real DR datasets (EyePACS, APTOS, IDRiD, Messidor) are licence-gated and large,
so the prototype ships with a generator that produces anatomically plausible
retinas with a KNOWN lesion inventory. That gives three things a downloaded
dataset would not:

  1. runnable demo and tests with zero download,
  2. pixel-level ground truth for the segmenter, which public grade-only
     datasets do not provide,
  3. controllable difficulty (blur, vignette, glare) for the quality gate.

These images are for pipeline validation, NOT for claiming clinical accuracy.
Swap in `data/loaders.py:load_aptos` to train and evaluate on real data.
"""
import cv2
import numpy as np

SIZE = 640

# ICDR grade -> plausible lesion inventory, drawn from the grading definitions.
GRADE_SPEC = {
    0: dict(ma=(0, 0), hem=(0, 0), ex=(0, 0), cws=(0, 0)),
    1: dict(ma=(1, 5), hem=(0, 1), ex=(0, 0), cws=(0, 0)),
    2: dict(ma=(6, 18), hem=(2, 6), ex=(1, 5), cws=(0, 2)),
    3: dict(ma=(18, 45), hem=(8, 20), ex=(5, 14), cws=(2, 6)),
    4: dict(ma=(25, 60), hem=(15, 35), ex=(10, 25), cws=(4, 10)),
}


def _rng(seed):
    return np.random.default_rng(seed)


def _base_retina(rng):
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    cy = cx = SIZE // 2
    radius = int(SIZE * 0.47)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    fov = dist <= radius

    # Choroidal background: orange-red, brighter centrally.
    falloff = np.clip(1.0 - (dist / radius) ** 2 * 0.55, 0, 1)
    base = np.array([38.0, 78.0, 185.0])            # BGR
    tint = rng.uniform(0.85, 1.15, size=3)          # inter-patient pigmentation
    for c in range(3):
        img[:, :, c] = base[c] * tint[c] * falloff

    # Low-frequency choroidal mottling.
    noise = rng.normal(0, 1, (SIZE // 8, SIZE // 8)).astype(np.float32)
    noise = cv2.resize(noise, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), 8) * 12.0
    img += noise[:, :, None]

    img[~fov] = 0
    return img, (cx, cy, radius), fov


def _draw_optic_disc(img, rng, fov_c):
    cx, cy, radius = fov_c
    side = rng.choice([-1, 1])
    dx = int(cx + side * radius * rng.uniform(0.48, 0.60))
    dy = int(cy + radius * rng.uniform(-0.12, 0.12))
    dr = int(radius * rng.uniform(0.11, 0.14))
    disc = np.zeros(img.shape[:2], np.float32)
    cv2.circle(disc, (dx, dy), dr, 1.0, -1)
    disc = cv2.GaussianBlur(disc, (0, 0), dr * 0.18)
    for c, v in enumerate((150.0, 215.0, 245.0)):
        img[:, :, c] = img[:, :, c] * (1 - disc) + v * disc
    return (dx, dy, dr)


def _draw_vessels(img, rng, disc, fov):
    """Recursive branching tree seeded at the optic disc."""
    dx, dy, dr = disc
    layer = np.zeros(img.shape[:2], np.float32)

    def branch(x, y, angle, width, length, depth):
        if depth > 5 or width < 0.8 or length < 6:
            return
        steps = int(length)
        curve = rng.normal(0, 0.05)
        for _ in range(steps):
            angle += curve + rng.normal(0, 0.04)
            nx, ny = x + np.cos(angle), y + np.sin(angle)
            if not (0 <= int(nx) < SIZE and 0 <= int(ny) < SIZE) or not fov[int(ny), int(nx)]:
                return
            cv2.line(layer, (int(x), int(y)), (int(nx), int(ny)), 1.0,
                     max(1, int(round(width))))
            x, y = nx, ny
            width *= 0.995
        for _ in range(2):
            branch(x, y, angle + rng.uniform(-0.7, 0.7), width * rng.uniform(0.55, 0.75),
                   length * rng.uniform(0.5, 0.75), depth + 1)

    # Vessels emerge from the disc RIM, not from its centre. Starting every
    # branch at the exact centre paints dark vessel straight through the disc
    # and drops its red value from 245 to ~150, which destroys any
    # brightness-based disc detector -- on healthy eyes especially, where there
    # is nothing else for the detector to latch onto.
    def emerge(angle, width, length, depth):
        sx = dx + 0.75 * dr * np.cos(angle)
        sy = dy + 0.75 * dr * np.sin(angle)
        branch(sx, sy, angle, width, length, depth)

    # Four arcades leave the disc superiorly and inferiorly, temporal side.
    toward_centre = np.arctan2(SIZE / 2 - dy, SIZE / 2 - dx)
    for k in range(4):
        a = toward_centre + rng.uniform(-0.9, 0.9) + (k - 1.5) * 0.35
        emerge(a, rng.uniform(4.0, 6.0), rng.uniform(55, 80), 0)
    for _ in range(4):
        emerge(rng.uniform(0, 2 * np.pi), rng.uniform(2.0, 3.5),
               rng.uniform(25, 45), 1)

    layer = cv2.GaussianBlur(layer, (0, 0), 1.0)
    layer = np.clip(layer, 0, 1)
    # Vessels are dark, and darker in green than red (haemoglobin absorption).
    for c, atten in enumerate((0.45, 0.30, 0.62)):
        img[:, :, c] *= (1 - layer * (1 - atten))
    return layer


def _macula(fov_c, disc, rng):
    cx, cy, radius = fov_c
    dx, dy, dr = disc
    side = -1 if dx > cx else 1
    return (int(dx + side * dr * rng.uniform(4.5, 5.5)), int(dy + dr * rng.uniform(-0.4, 0.4)))


def _draw_macula(img, mac, rng):
    m = np.zeros(img.shape[:2], np.float32)
    cv2.circle(m, mac, int(SIZE * 0.075), 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), SIZE * 0.045)
    m = np.clip(m, 0, 1) * 0.42
    img *= (1 - m[:, :, None])


def _place(rng, fov, mac, n, near_macula_bias=0.0):
    """Sample lesion centres inside the FOV, optionally clustered at the macula
    the way exudates and oedema-associated lesions actually are."""
    ys, xs = np.where(fov)
    pts = []
    for _ in range(n):
        if rng.random() < near_macula_bias:
            r = rng.uniform(0, SIZE * 0.14)
            a = rng.uniform(0, 2 * np.pi)
            x, y = int(mac[0] + r * np.cos(a)), int(mac[1] + r * np.sin(a))
            if 0 <= x < SIZE and 0 <= y < SIZE and fov[y, x]:
                pts.append((x, y))
                continue
        i = rng.integers(len(xs))
        pts.append((int(xs[i]), int(ys[i])))
    return pts


def _draw_dark_lesion(img, truth, pt, radius_px, irregular, rng):
    layer = np.zeros(img.shape[:2], np.float32)
    if irregular:
        # Blot/flame haemorrhage: overlapping lobes give the ragged outline.
        for _ in range(rng.integers(3, 7)):
            off = rng.normal(0, radius_px * 0.5, 2)
            cv2.circle(layer, (int(pt[0] + off[0]), int(pt[1] + off[1])),
                       max(1, int(radius_px * rng.uniform(0.5, 1.0))), 1.0, -1)
    else:
        cv2.circle(layer, pt, max(1, int(radius_px)), 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), max(0.6, radius_px * 0.25))
    layer = np.clip(layer, 0, 1)
    truth[layer > 0.4] = 1
    depth = rng.uniform(0.55, 0.8)
    for c, atten in enumerate((0.35, 0.20, 0.70)):
        img[:, :, c] *= (1 - layer * depth * (1 - atten))


def _draw_bright_lesion(img, truth, pt, radius_px, fuzzy, rng):
    layer = np.zeros(img.shape[:2], np.float32)
    for _ in range(rng.integers(1, 4)):
        off = rng.normal(0, radius_px * 0.4, 2)
        cv2.circle(layer, (int(pt[0] + off[0]), int(pt[1] + off[1])),
                   max(1, int(radius_px * rng.uniform(0.6, 1.1))), 1.0, -1)
    # Cotton-wool spots have soft borders; hard exudates have crisp ones.
    sigma = radius_px * (0.9 if fuzzy else 0.22)
    layer = cv2.GaussianBlur(layer, (0, 0), max(0.6, sigma))
    layer = np.clip(layer, 0, 1)
    truth[layer > 0.4] = 1
    colour = (170.0, 235.0, 250.0) if not fuzzy else (200.0, 225.0, 230.0)
    strength = layer * (0.85 if not fuzzy else 0.55)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - strength) + colour[c] * strength


def degrade(img, rng, blur=0.0, vignette=0.0, glare=0.0, exposure=1.0):
    """Simulate low-cost-camera failure modes for quality-gate testing."""
    out = img.copy()
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if vignette > 0:
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
        a = rng.uniform(0, 2 * np.pi)
        ramp = ((xx - SIZE / 2) * np.cos(a) + (yy - SIZE / 2) * np.sin(a)) / SIZE
        out *= np.clip(1.0 - vignette * (0.5 + ramp), 0.05, 1.0)[:, :, None]
    if glare > 0:
        g = np.zeros((SIZE, SIZE), np.float32)
        cx, cy = rng.integers(SIZE // 4, 3 * SIZE // 4, 2)
        cv2.circle(g, (int(cx), int(cy)), int(SIZE * 0.10 * glare), 1.0, -1)
        g = cv2.GaussianBlur(g, (0, 0), SIZE * 0.04)
        out = out * (1 - g[:, :, None]) + 255.0 * g[:, :, None]
    out *= exposure
    return out


def generate(grade, seed=0, **degradation):
    """Return (bgr_uint8, truth_dict) for one synthetic eye at the given ICDR grade."""
    rng = _rng(seed)
    img, fov_c, fov = _base_retina(rng)
    disc = _draw_optic_disc(img, rng, fov_c)
    _draw_vessels(img, rng, disc, fov)
    mac = _macula(fov_c, disc, rng)
    _draw_macula(img, mac, rng)

    spec = GRADE_SPEC[grade]
    truth = {k: np.zeros((SIZE, SIZE), np.uint8) for k in ("MA", "HEM", "EX", "CWS")}
    counts = {}

    n = int(rng.integers(spec["ma"][0], spec["ma"][1] + 1))
    counts["MA"] = n
    for pt in _place(rng, fov, mac, n, 0.25):
        _draw_dark_lesion(img, truth["MA"], pt, rng.uniform(1.6, 2.6), False, rng)

    n = int(rng.integers(spec["hem"][0], spec["hem"][1] + 1))
    counts["HEM"] = n
    for pt in _place(rng, fov, mac, n, 0.15):
        _draw_dark_lesion(img, truth["HEM"], pt, rng.uniform(4.0, 9.0), True, rng)

    n = int(rng.integers(spec["ex"][0], spec["ex"][1] + 1))
    counts["EX"] = n
    for pt in _place(rng, fov, mac, n, 0.55):
        _draw_bright_lesion(img, truth["EX"], pt, rng.uniform(2.5, 6.0), False, rng)

    n = int(rng.integers(spec["cws"][0], spec["cws"][1] + 1))
    counts["CWS"] = n
    for pt in _place(rng, fov, mac, n, 0.30):
        _draw_bright_lesion(img, truth["CWS"], pt, rng.uniform(5.0, 10.0), True, rng)

    if degradation:
        img = degrade(img, rng, **degradation)

    img += rng.normal(0, 2.2, img.shape).astype(np.float32)
    img[~fov] = 0
    bgr = np.clip(img, 0, 255).astype(np.uint8)
    return bgr, {"grade": grade, "counts": counts, "masks": truth,
                 "optic_disc": disc, "macula": mac}


def cohort(n_per_grade=20, seed=0, grades=(0, 1, 2, 3, 4)):
    """Yield a balanced labelled cohort for training and evaluation."""
    k = seed * 100000
    for g in grades:
        for i in range(n_per_grade):
            k += 1
            yield generate(g, seed=k)
