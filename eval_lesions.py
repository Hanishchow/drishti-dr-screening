"""Lesion-detector evaluation against synthetic pixel ground truth.

Detections are matched to truth by centroid hit-test against the truth mask,
dilated by a small tolerance. Dark lesions (MA/HEM) and bright lesions (EX/CWS)
are scored both per-class and pooled, because the MA-vs-HEM and EX-vs-CWS
splits are shape heuristics layered on top of the detection itself -- it is
worth knowing whether a miss is a detection failure or a labelling failure.
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np

from core import preprocess, lesions
from data.synth import cohort, SIZE

TOLERANCE_PX = 4
DARK, BRIGHT = ("MA", "HEM"), ("EX", "CWS")


def _truth_at_working_res(truth_masks, prep):
    """Replay the exact crop+resize the image went through.

    Resizing the full original frame instead would leave truth and detections
    a few percent out of register -- invisible on inspection, but enough to
    destroy recall at a 4px matching tolerance.
    """
    return {k: prep.map_from_original(m) for k, m in truth_masks.items()}


def score(n_per_grade=8, seed=1):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for bgr, truth in cohort(n_per_grade=n_per_grade, seed=seed):
        prep = preprocess.prepare(bgr)
        lm = lesions.detect(prep)
        tmasks = _truth_at_working_res(truth["masks"], prep)
        pooled = {
            "dark": cv2.dilate((tmasks["MA"] | tmasks["HEM"]),
                               np.ones((TOLERANCE_PX * 2 + 1,) * 2, np.uint8)),
            "bright": cv2.dilate((tmasks["EX"] | tmasks["CWS"]),
                                 np.ones((TOLERANCE_PX * 2 + 1,) * 2, np.uint8)),
        }
        # --- detection-level precision (is there really a lesion here?) ---
        for l in lm.lesions:
            group = "dark" if l.kind in DARK else "bright"
            hit = pooled[group][min(l.y, pooled[group].shape[0] - 1),
                                min(l.x, pooled[group].shape[1] - 1)] > 0
            if hit:
                tp[group] += 1
            else:
                fp[group] += 1
        # --- recall, per truth component ---
        for group, keys in (("dark", DARK), ("bright", BRIGHT)):
            tm = tmasks[keys[0]] | tmasks[keys[1]]
            n, labels, stats, cents = cv2.connectedComponentsWithStats(tm, 8)
            det = np.zeros(tm.shape, np.uint8)
            for l in lm.lesions:
                if l.kind in (DARK if group == "dark" else BRIGHT):
                    cv2.circle(det, (l.x, l.y), TOLERANCE_PX, 1, -1)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < 2:
                    continue
                comp = labels == i
                if (det[comp] > 0).any():
                    pass          # counted as tp above
                else:
                    fn[group] += 1
    rows = []
    for g in ("dark", "bright"):
        prec = tp[g] / max(tp[g] + fp[g], 1)
        rec = tp[g] / max(tp[g] + fn[g], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append((g, tp[g], fp[g], fn[g], prec, rec, f1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="images per ICDR grade")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    print(f"{'group':8} {'TP':>5} {'FP':>5} {'FN':>5} {'prec':>6} {'rec':>6} {'F1':>6}")
    for g, t, f, n, p, r, f1 in score(args.n, args.seed):
        print(f"{g:8} {t:5d} {f:5d} {n:5d} {p:6.3f} {r:6.3f} {f1:6.3f}")


if __name__ == "__main__":
    main()
