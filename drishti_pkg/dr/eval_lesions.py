"""Score the morphological segmenter against IDRiD's pixel-level lesion masks.

    python -m dr.eval_lesions --data-root /path/to/datasets

IDRiD is the only public corpus that ships per-class lesion masks, which makes
it the only way to measure the "where" channel honestly. APTOS, EyePACS and
Messidor-2 carry a grade per image and nothing else; a segmenter evaluated
against those can only ever be assessed indirectly.

Two scores are reported because they answer different questions:

  LESION-LEVEL   did we find each individual lesion? Detections are matched to
                 connected components of the truth mask within a tolerance.
                 This is what the clinician sees on the overlay.
  PIXEL-LEVEL    IoU and Dice against the mask. Harsher, and dominated by
                 boundary disagreement on large haemorrhages, but it is the
                 number the segmentation literature reports.

Ground truth is mapped through the SAME crop-and-resize the image went
through. Resizing the full original frame instead leaves truth a few percent
out of register -- invisible by eye, and it silently destroyed recall the last
time this evaluation was written.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import cv2
import numpy as np

from core import lesions, preprocess
from . import datasets as D

# Matching tolerance in microns, converted to pixels at the working resolution.
# 250 um is about the radius of a large microaneurysm -- tight enough that a
# detection must genuinely overlap the lesion.
TOLERANCE_UM = 250.0

DARK = ("MA", "HEM")
BRIGHT = ("EX", "CWS")


def load_masks(record, prep):
    """Read IDRiD masks and map them into the prepared image's frame."""
    out = {}
    for kind, path in record.lesion_masks.items():
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        out[kind] = (prep.map_from_original((m > 0).astype(np.uint8)) > 0).astype(np.uint8)
    return out


def score_record(record, tolerance_px):
    prep = preprocess.prepare(cv2.imread(record.image_path, cv2.IMREAD_COLOR))
    truth = load_masks(record, prep)
    if not truth:
        return None
    lmap = lesions.detect(prep)

    kernel = np.ones((tolerance_px * 2 + 1,) * 2, np.uint8)
    result = {}

    for group, kinds in (("dark", DARK), ("bright", BRIGHT)):
        gt = np.zeros(prep.green.shape, np.uint8)
        for k in kinds:
            if k in truth:
                gt |= truth[k]
        if not gt.any():
            continue

        pred = np.zeros(prep.green.shape, np.uint8)
        points = []
        for k in kinds:
            layer = lmap.overlays.get(k)
            if layer is not None:
                pred |= layer
        for l in lmap.lesions:
            if l.kind in kinds:
                points.append((l.x, l.y))

        # --- lesion level ---
        gt_dilated = cv2.dilate(gt, kernel)
        tp = fp = 0
        hit_canvas = np.zeros(gt.shape, np.uint8)
        h, w = gt.shape
        for x, y in points:
            if gt_dilated[min(y, h - 1), min(x, w - 1)] > 0:
                tp += 1
            else:
                fp += 1
            cv2.circle(hit_canvas, (x, y), tolerance_px, 1, -1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(gt, 8)
        fn = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 3:
                continue
            if not (hit_canvas[labels == i] > 0).any():
                fn += 1

        # --- pixel level ---
        inter = int((pred & gt).sum())
        union = int((pred | gt).sum())
        result[group] = {
            "tp": tp, "fp": fp, "fn": fn,
            "inter": inter, "union": union,
            "pred_px": int(pred.sum()), "gt_px": int(gt.sum()),
        }
    return result


def aggregate(per_image):
    totals = defaultdict(lambda: defaultdict(int))
    for res in per_image:
        for group, d in res.items():
            for k, v in d.items():
                totals[group][k] += v

    report = {}
    for group, d in totals.items():
        prec = d["tp"] / max(d["tp"] + d["fp"], 1)
        rec = d["tp"] / max(d["tp"] + d["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        iou = d["inter"] / max(d["union"], 1)
        dice = 2 * d["inter"] / max(d["pred_px"] + d["gt_px"], 1)
        report[group] = {"precision": prec, "recall": rec, "f1": f1,
                         "iou": iou, "dice": dice,
                         "tp": d["tp"], "fp": d["fp"], "fn": d["fn"]}
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lesion segmentation vs IDRiD masks")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        records = D.load_idrid_segmentation(args.data_root)
    except D.DatasetUnavailable as e:
        print(e)
        return 1
    if not records:
        print("IDRiD found, but no images carry lesion masks. Make sure the "
              "'A. Segmentation' folder is present.")
        return 1
    if args.limit:
        records = records[:args.limit]

    tolerance_px = max(2, int(round(TOLERANCE_UM / (13000.0 / preprocess.TARGET))))
    print(f"scoring {len(records)} IDRiD images "
          f"(tolerance {TOLERANCE_UM:.0f} um = {tolerance_px} px)\n")

    per_image = []
    for i, rec in enumerate(records, 1):
        try:
            res = score_record(rec, tolerance_px)
        except Exception as e:                    # noqa: BLE001
            print(f"  skip {rec.image_path}: {e}")
            continue
        if res:
            per_image.append(res)
        if i % 10 == 0:
            print(f"  {i}/{len(records)}", flush=True)

    if not per_image:
        print("no images produced a score")
        return 1

    report = aggregate(per_image)
    print(f"\n{'group':8} {'prec':>7} {'recall':>7} {'F1':>7} {'IoU':>7} {'Dice':>7}"
          f" {'TP':>7} {'FP':>7} {'FN':>7}")
    for group in ("dark", "bright"):
        if group not in report:
            continue
        r = report[group]
        print(f"{group:8} {r['precision']:7.3f} {r['recall']:7.3f} {r['f1']:7.3f} "
              f"{r['iou']:7.3f} {r['dice']:7.3f} {r['tp']:7d} {r['fp']:7d} {r['fn']:7d}")
    print(f"\nscored {len(per_image)} images with masks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
