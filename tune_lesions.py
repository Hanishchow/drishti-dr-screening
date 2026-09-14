"""Threshold sweep for the lesion detector.

The expensive stages (Frangi vesselness, preprocessing, top-hat pyramids) do not
depend on the thresholds being tuned, so they are computed once per image and
cached. Only the threshold-and-label step is re-run per parameter combination,
which turns an overnight sweep into a sub-minute one.

NOTE: evaluate() models the threshold-and-label step only. It does NOT apply the
shape-aware vessel-fragment rejection that detect() uses, so its dark-lesion
numbers are a lower bound. Always confirm a chosen parameter set with
eval_lesions.py, which runs the real detector end to end.
"""
import itertools
import sys

import cv2
import numpy as np

from core import preprocess, lesions as L
from core.preprocess import flatten_illumination
from data.synth import cohort

TOL = 4


def build_cache(n_per_grade=6, seed=2):
    cache = []
    for bgr, truth in cohort(n_per_grade=n_per_grade, seed=seed):
        prep = preprocess.prepare(bgr)
        flat = flatten_illumination(prep.green, prep.mask)
        vessels = L.segment_vessels(prep.green, prep.mask)
        disc = L.locate_optic_disc(prep.bgr, prep.mask)
        disc_mask = np.zeros_like(prep.mask)
        cv2.circle(disc_mask, (disc[0], disc[1]), int(disc[2] * 1.4), 1, -1)
        margin = max(L.DARK_SCALES + L.BRIGHT_SCALES) + 4
        valid = cv2.bitwise_and(cv2.erode(prep.mask, L._strel(margin)), 1 - disc_mask)
        dark = L._multiscale_tophat(flat, L.DARK_SCALES, True)
        bright = L._multiscale_tophat(flat, L.BRIGHT_SCALES, False)
        # Ground truth must traverse the same crop+resize as the image.
        tmask = {k: prep.map_from_original(v) for k, v in truth["masks"].items()}
        k = np.ones((TOL * 2 + 1,) * 2, np.uint8)
        cache.append(dict(
            prep=prep, vessels=vessels, valid=valid, dark=dark, bright=bright,
            truth_dark=cv2.dilate(tmask["MA"] | tmask["HEM"], k),
            truth_bright=cv2.dilate(tmask["EX"] | tmask["CWS"], k),
            comp_dark=tmask["MA"] | tmask["HEM"],
            comp_bright=tmask["EX"] | tmask["CWS"],
        ))
    return cache


def _detect_points(response, region, k, floor, min_px, open_r):
    binary = L._threshold(response, region, k, floor)
    if open_r > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, L._strel(open_r))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
    return [(int(cents[i][0]), int(cents[i][1]))
            for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]


def evaluate(cache, group, k, floor, min_px, open_r, vessel_dilate=2):
    tp = fp = fn = 0
    for c in cache:
        if group == "dark":
            region = cv2.bitwise_and(
                c["valid"], 1 - cv2.dilate(c["vessels"], L._strel(vessel_dilate)))
            response = cv2.bitwise_and(c["dark"], c["dark"], mask=region)
            truth, comp_src = c["truth_dark"], c["comp_dark"]
        else:
            region = c["valid"]
            response = cv2.bitwise_and(c["bright"], c["bright"], mask=region)
            truth, comp_src = c["truth_bright"], c["comp_bright"]

        pts = _detect_points(response, region, k, floor, min_px, open_r)
        h, w = truth.shape
        det = np.zeros((h, w), np.uint8)
        for x, y in pts:
            if truth[min(y, h - 1), min(x, w - 1)] > 0:
                tp += 1
            else:
                fp += 1
            cv2.circle(det, (x, y), TOL, 1, -1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(comp_src, 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 2:
                continue
            if not (det[labels == i] > 0).any():
                fn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9), tp, fp, fn


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "dark"
    print(f"building cache for '{group}' sweep...", flush=True)
    cache = build_cache()
    print(f"{len(cache)} images cached", flush=True)

    if group == "dark":
        grid = itertools.product([1.5, 2.5, 3.5, 5.0], [2.0, 6.0, 10.0],
                                 [2, 4, 8], [0, 1], [1, 2, 3])
        keys = ("k", "floor", "min_px", "open_r", "vessel_dilate")
    else:
        grid = itertools.product([6.0, 9.0, 13.0, 18.0, 25.0], [14.0, 25.0, 40.0],
                                 [6, 12, 25, 45], [0, 1], [2])
        keys = ("k", "floor", "min_px", "open_r", "vessel_dilate")

    rows = []
    for combo in grid:
        k, floor, min_px, open_r, vd = combo
        prec, rec, f1, tp, fp, fn = evaluate(cache, group, k, floor, min_px, open_r, vd)
        rows.append((f1, prec, rec, combo, tp, fp, fn))
    rows.sort(reverse=True)
    print(f"\ntop 10 by F1 ({group}):")
    print(f"{'F1':>6} {'prec':>6} {'rec':>6}  {'  '.join(k[:7] for k in keys)}")
    for f1, prec, rec, combo, tp, fp, fn in rows[:10]:
        print(f"{f1:6.3f} {prec:6.3f} {rec:6.3f}  {combo}  tp={tp} fp={fp} fn={fn}")


if __name__ == "__main__":
    main()
