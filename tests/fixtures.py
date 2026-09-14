"""Minimal deterministic test images.

These are NOT a synthetic fundus dataset and are never used for training,
tuning or any reported figure -- the generator that did that has been removed
in favour of real corpora. What remains here are the simplest geometric images
that can express a pipeline invariant:

    a featureless disc has no lesions, so the detector must return none
    a disc with a known dark spot has one, so the detector must find it

Real-image behaviour is measured against IDRiD (`dr/eval_lesions.py`) and the
real grading corpora; these fixtures only keep the invariants runnable in CI on
a machine with no dataset.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

SIZE = 512
FOV_FRAC = 0.45


def blank_retina(size=SIZE, brightness=1.0):
    """A smooth illuminated disc: correct answer is zero lesions."""
    img = np.zeros((size, size, 3), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
    radius = size * FOV_FRAC
    fov = r <= radius
    falloff = np.clip(1.0 - (r / radius) ** 2 * 0.55, 0, 1)
    for c, v in enumerate((38.0, 78.0, 185.0)):
        img[:, :, c] = v * brightness * falloff
    img[~fov] = 0
    return np.clip(img, 0, 255).astype(np.uint8)


def retina_with_spots(n_dark=0, n_bright=0, size=SIZE, seed=0, radius=4):
    """A blank retina plus a known number of high-contrast blobs."""
    img = blank_retina(size)
    rng = np.random.default_rng(seed)
    placed = []
    limit = size * FOV_FRAC * 0.6
    for kind, count in (("dark", n_dark), ("bright", n_bright)):
        for _ in range(count):
            while True:
                a = rng.uniform(0, 2 * np.pi)
                d = rng.uniform(size * 0.08, limit)
                p = (int(size / 2 + d * np.cos(a)), int(size / 2 + d * np.sin(a)))
                if all(np.hypot(p[0] - q[0], p[1] - q[1]) > radius * 5 for q in placed):
                    break
            placed.append(p)
            colour = (12, 20, 60) if kind == "dark" else (200, 245, 250)
            cv2.circle(img, p, radius, colour, -1)
    return cv2.GaussianBlur(img, (0, 0), 0.6), placed


def sharp_then_blurred(sigma, size=SIZE):
    """Textured retina at a controlled defocus, for the quality gate."""
    img = blank_retina(size)
    rng = np.random.default_rng(1)
    for _ in range(120):
        p = rng.integers(int(size * 0.1), int(size * 0.9), 2)
        cv2.circle(img, tuple(int(v) for v in p), 3, (25, 45, 120), -1)
    for _ in range(14):
        p = rng.integers(int(size * 0.15), int(size * 0.85), 2)
        q = p + rng.integers(-60, 60, 2)
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q),
                 (20, 40, 110), 3)
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    # Sensor noise is added AFTER optical blur, as in a real camera. The
    # ordering matters: a focus metric that ignores it scores a blurred frame
    # as sharper than a crisp one.
    img = np.clip(img.astype(np.float32)
                  + np.random.default_rng(2).normal(0, 2.2, img.shape), 0, 255)
    return img.astype(np.uint8)


# ------------------------------------------------------------ real corpora
def idrid_available():
    try:
        from dr import datasets as D
        return len(D.load_idrid_segmentation()) > 0
    except Exception:
        return False


requires_idrid = pytest.mark.skipif(
    not idrid_available(),
    reason="IDRiD not present. Download from https://idrid.grand-challenge.org/ "
           "and set DR_DATA_ROOT to run the real-image checks.")
