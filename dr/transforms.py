"""Fundus preprocessing and augmentation, in plain OpenCV.

Written directly against cv2 rather than pulling in albumentations for two
reasons: the transforms that matter here are fundus-specific (field-of-view
cropping, Graham normalisation) and not in any generic library, and keeping the
dependency list short means the same code runs unmodified on Kaggle, in the
Docker image, and on a CPU-only PHC edge box.

The single most important step is the FOV crop. Public DR corpora are a mess of
letterboxed, off-centre, differently-zoomed captures; without normalising the
retinal circle first, a network spends its capacity learning which hospital
took the photograph.
"""
from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------- FOV crop
def fov_bbox(bgr, threshold_scale=0.06):
    """Bounding box of the illuminated retinal circle.

    Thresholds relative to the image's own maximum rather than at a fixed grey
    level, so an underexposed capture is cropped as tightly as a bright one.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    thr = max(6.0, float(small.max()) * threshold_scale)
    mask = (small > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return 0, 0, bgr.shape[1], bgr.shape[0]
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sy, sx = bgr.shape[0] / 256.0, bgr.shape[1] / 256.0
    x = int(stats[k, cv2.CC_STAT_LEFT] * sx)
    y = int(stats[k, cv2.CC_STAT_TOP] * sy)
    w = int(stats[k, cv2.CC_STAT_WIDTH] * sx)
    h = int(stats[k, cv2.CC_STAT_HEIGHT] * sy)
    return x, y, max(w, 1), max(h, 1)


def crop_to_fov(bgr, pad=0.02):
    x, y, w, h = fov_bbox(bgr)
    side = int(max(w, h) * (1 + pad))
    cx, cy = x + w // 2, y + h // 2
    half = side // 2
    p = side
    padded = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded[cy - half + p:cy + half + p, cx - half + p:cx + half + p]


def circular_mask(size, shrink=0.97):
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), int(size / 2 * shrink), 1, -1)
    return m


def graham_normalise(bgr, size, sigma_frac=1 / 30.0, weight=4.0, bias=128.0):
    """Ben Graham's EyePACS-winning normalisation.

    Subtracts a heavily blurred copy of the image, which removes the
    camera-specific colour cast and vignetting while amplifying exactly the
    local contrast that microaneurysms and exudates live in. It remains the
    strongest single preprocessing step for this task.
    """
    blur = cv2.GaussianBlur(bgr, (0, 0), size * sigma_frac)
    out = cv2.addWeighted(bgr, weight, blur, -weight, bias)
    mask = circular_mask(out.shape[0])
    return cv2.bitwise_and(out, out, mask=mask)


def load_and_prepare(path, size=512, graham=True):
    """Disk -> normalised square BGR uint8. Used identically at train and serve
    time; any divergence between the two is a silent accuracy loss."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read image: {path}")
    bgr = crop_to_fov(bgr)
    if bgr.size == 0:
        raise ValueError(f"empty crop for image: {path}")
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    if graham:
        bgr = graham_normalise(bgr, size)
    else:
        bgr = cv2.bitwise_and(bgr, bgr, mask=circular_mask(size))
    return bgr


# ------------------------------------------------------------ augmentation
def augment(bgr, rng):
    """Geometric and photometric jitter appropriate to fundus photography.

    Retinal images have no canonical orientation once laterality is discarded,
    so full flips and rotations are safe and are the highest-value augmentation
    here. Photometric jitter is deliberately mild: push brightness or contrast
    too far and genuine microaneurysms are destroyed, which teaches the network
    to ignore the smallest and earliest sign of disease.
    """
    h, w = bgr.shape[:2]

    if rng.random() < 0.5:
        bgr = cv2.flip(bgr, 1)
    if rng.random() < 0.5:
        bgr = cv2.flip(bgr, 0)

    angle = rng.uniform(0, 360)
    scale = rng.uniform(0.92, 1.08)
    tx = rng.uniform(-0.02, 0.02) * w
    ty = rng.uniform(-0.02, 0.02) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    bgr = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    if rng.random() < 0.7:
        alpha = rng.uniform(0.9, 1.1)          # contrast
        beta = rng.uniform(-10, 10)            # brightness
        bgr = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)

    if rng.random() < 0.3:
        gamma = rng.uniform(0.85, 1.15)
        lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255, 0, 255)
        bgr = cv2.LUT(bgr, lut.astype(np.uint8))

    if rng.random() < 0.2:
        bgr = cv2.GaussianBlur(bgr, (0, 0), rng.uniform(0.4, 1.0))

    return cv2.bitwise_and(bgr, bgr, mask=circular_mask(h))


def to_tensor(bgr):
    """BGR uint8 HWC -> normalised RGB float CHW, as a numpy array.

    Returned as numpy rather than torch so this module stays importable without
    torch (the edge inference path uses onnxruntime and no torch at all).
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))
