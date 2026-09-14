"""Metrics for DR grading.

Quadratic weighted kappa is the field-standard metric for this task (it is what
both the EyePACS and APTOS challenges were scored on) because it is the only
common metric that understands that DR grades are ORDINAL: confusing grade 0
with grade 4 is a far worse error than confusing 3 with 4, and plain accuracy
scores those identically.

Everything here is pure numpy so it can be unit-tested without a dataset, a
GPU, or a trained model.
"""
import numpy as np

GRADES = 5
REFERABLE_THRESHOLD = 2      # ICDR >= 2 (moderate NPDR) needs an ophthalmologist
SIGHT_THREATENING_THRESHOLD = 3


def confusion(y_true, y_pred, n=GRADES):
    m = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()):
        m[int(t), int(p)] += 1
    return m


def quadratic_weighted_kappa(y_true, y_pred, n=GRADES):
    """Cohen's kappa with quadratic penalties.

    1.0 = perfect, 0.0 = no better than chance agreement, negative = worse than
    chance. Implemented directly rather than via sklearn so the expected-matrix
    construction is visible and testable.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    if y_true.size == 0:
        return 0.0

    O = confusion(y_true, y_pred, n).astype(np.float64)

    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    W = ((i - j) ** 2) / ((n - 1) ** 2)

    # Expected agreement under independence, scaled to the same total as O.
    hist_true = np.bincount(y_true, minlength=n).astype(np.float64)
    hist_pred = np.bincount(y_pred, minlength=n).astype(np.float64)
    E = np.outer(hist_true, hist_pred)
    if E.sum() == 0:
        return 0.0
    E = E * (O.sum() / E.sum())

    denom = (W * E).sum()
    if denom < 1e-12:
        # Happens when every label and prediction is the same single class.
        return 1.0 if (O.trace() == O.sum()) else 0.0
    return float(1.0 - (W * O).sum() / denom)


def binary_screening_metrics(y_true, y_pred, threshold=REFERABLE_THRESHOLD):
    """Sensitivity / specificity at a severity cut-point.

    This is what a screening programme is actually accountable for. The UK NHS
    DR screening standard is >=85% sensitivity and >=80% specificity for
    referable disease; those are the numbers to beat, not accuracy.
    """
    t = np.asarray(y_true).ravel() >= threshold
    p = np.asarray(y_pred).ravel() >= threshold
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    tn = int((~t & ~p).sum())
    fn = int((t & ~p).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {"sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def summarise(y_true, y_pred):
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    ref = binary_screening_metrics(y_true, y_pred, REFERABLE_THRESHOLD)
    stg = binary_screening_metrics(y_true, y_pred, SIGHT_THREATENING_THRESHOLD)
    return {
        "n": int(y_true.size),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
        "within_one": float((np.abs(y_true - y_pred) <= 1).mean()) if y_true.size else 0.0,
        "referable_sensitivity": ref["sensitivity"],
        "referable_specificity": ref["specificity"],
        "referable_ppv": ref["ppv"],
        "sight_threatening_sensitivity": stg["sensitivity"],
        "sight_threatening_specificity": stg["specificity"],
        "confusion": confusion(y_true, y_pred).tolist(),
    }


def format_report(stats, title="grading"):
    lines = [f"{title}  (n={stats['n']})",
             f"  QWK                          : {stats['qwk']:.4f}",
             f"  exact accuracy               : {stats['accuracy']:.4f}",
             f"  within +/-1 grade            : {stats['within_one']:.4f}",
             f"  referable (>=2) sens / spec  : "
             f"{stats['referable_sensitivity']:.4f} / {stats['referable_specificity']:.4f}",
             f"  sight-threatening (>=3) s/s  : "
             f"{stats['sight_threatening_sensitivity']:.4f} / "
             f"{stats['sight_threatening_specificity']:.4f}",
             "  confusion (rows = true):"]
    cm = np.array(stats["confusion"])
    lines.append("        " + " ".join(f"{i:6d}" for i in range(cm.shape[1])))
    for i, row in enumerate(cm):
        lines.append(f"  true {i}: " + " ".join(f"{v:6d}" for v in row))
    return "\n".join(lines)


# --------------------------------------------------------------- thresholds
MIN_THRESHOLD_GAP = 0.15


def optimise_thresholds(y_true, y_score, init=(0.5, 1.5, 2.5, 3.5),
                        min_gap=MIN_THRESHOLD_GAP):
    """Fit cut-points that convert a continuous severity score into grades.

    A regression head trained on DR outputs something like 2.37; turning that
    into a grade with plain rounding assumes the classes are evenly spaced on
    the score axis, which they are not -- the datasets are dominated by grade 0,
    which drags scores down and systematically under-grades disease. Fitting the
    cut-points to maximise QWK typically adds several points of kappa and, more
    importantly, recovers sensitivity on the rare severe grades.

    Coordinate ascent: each boundary is swept in turn while the others are held,
    which is stable and needs no gradient.

    `min_gap` is not cosmetic. On a small or easy validation split the search
    happily collapses all four cut-points into a near-identical cluster
    (e.g. [1.995, 1.996, 1.997, 1.998]), which maximises kappa on THAT split
    while making grades 1, 2 and 3 unreachable for any future patient -- the
    model could then only ever output 0 or 4. Forcing a minimum separation
    keeps every grade addressable.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel().astype(float)
    if y_score.size == 0:
        return np.array(init, dtype=float), 0.0
    th = np.array(init, dtype=float)

    def evaluate(candidate):
        return quadratic_weighted_kappa(y_true, apply_thresholds(y_score, candidate))

    best = evaluate(th)
    lo_bound = float(y_score.min()) - 0.5
    hi_bound = float(y_score.max()) + 0.5

    for _ in range(12):
        improved = False
        for k in range(len(th)):
            lo = th[k - 1] + min_gap if k > 0 else lo_bound
            hi = th[k + 1] - min_gap if k < len(th) - 1 else hi_bound
            if hi <= lo:
                continue
            for cand in np.linspace(lo, hi, 60):
                trial = th.copy()
                trial[k] = cand
                score = evaluate(trial)
                if score > best + 1e-9:
                    best, th, improved = score, trial, True
        if not improved:
            break

    th = enforce_threshold_separation(th, min_gap)
    return th, evaluate(th)


def enforce_threshold_separation(thresholds, min_gap=MIN_THRESHOLD_GAP):
    """Push cut-points apart so every grade stays reachable."""
    th = np.sort(np.asarray(thresholds, dtype=float)).copy()
    for k in range(1, len(th)):
        if th[k] - th[k - 1] < min_gap:
            th[k] = th[k - 1] + min_gap
    return th


def apply_thresholds(y_score, thresholds):
    """Map continuous scores to integer grades via ordered cut-points."""
    y_score = np.asarray(y_score, dtype=float).ravel()
    out = np.zeros_like(y_score, dtype=np.int64)
    for t in np.sort(np.asarray(thresholds, dtype=float)):
        out += (y_score > t).astype(np.int64)
    return np.clip(out, 0, GRADES - 1)


def coral_expected_grade(logits):
    """Continuous expected grade from CORAL cumulative logits.

    For a non-negative integer variable, E[y] = sum_k P(y > k), which is exactly
    the sum of the cumulative units. This is the QWK-optimal point estimate:
    kappa penalises SQUARED distance, and the mean is what minimises squared
    error -- the mode (argmax of the distribution) does not, and on a broad or
    bimodal posterior the two can differ by two whole grades.

    Returned unrounded so thresholds can be fitted against it.
    """
    dist = coral_logits_to_distribution(logits)
    if dist.ndim == 1:
        dist = dist[None, :]
    return (dist * np.arange(GRADES)).sum(axis=1)


def coral_logits_to_grade(logits):
    """Decode CORAL ordinal logits into an integer grade.

    Derived from the same distribution the report displays, so the headline
    grade can never disagree with the probability bar shown beside it. Using
    the raw sigmoid sum here instead would let the two drift apart whenever the
    network emits non-monotone cumulative outputs.
    """
    return np.clip(np.round(coral_expected_grade(logits)).astype(int), 0, GRADES - 1)


def coral_logits_to_distribution(logits):
    """Turn cumulative logits into a proper 5-way probability distribution.

    P(y = k) = P(y > k-1) - P(y > k), with the cumulative probabilities forced
    to be non-increasing first so no class can come out negative.
    """
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))
    single = probs.ndim == 1
    if single:
        probs = probs[None, :]
    n, k = probs.shape
    # Enforce monotonicity: P(y>0) >= P(y>1) >= ...
    probs = np.minimum.accumulate(probs, axis=1)
    cum = np.concatenate([np.ones((n, 1)), probs, np.zeros((n, 1))], axis=1)
    dist = cum[:, :-1] - cum[:, 1:]
    dist = np.clip(dist, 0.0, None)
    dist /= np.maximum(dist.sum(axis=1, keepdims=True), 1e-12)
    return dist[0] if single else dist
