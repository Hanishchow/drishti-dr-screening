"""Patient-grouped, grade-stratified cross-validation splits.

Two mistakes are easy to make here and both silently inflate the reported
score, which is the worst kind of bug in a clinical prototype:

  1. Splitting by IMAGE. EyePACS and Messidor-2 contain both eyes of the same
     patient, and the two eyes of one diabetic are highly correlated. A random
     image split puts a patient's left eye in train and the right eye in
     validation, and the model is scored partly on memorisation. Every split
     here is grouped by patient.

  2. Ignoring grade imbalance. Grades 3 and 4 are only a few percent of these
     corpora, so an ungrouped random fold can end up with almost no severe
     cases and a meaningless sensitivity estimate. Folds are stratified by the
     patient's worst grade.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def patient_groups(records):
    """patient_id -> (indices, worst grade seen for that patient)."""
    by_patient = defaultdict(list)
    for i, r in enumerate(records):
        by_patient[r.patient_id].append(i)
    return {pid: (idx, max(records[i].grade for i in idx))
            for pid, idx in by_patient.items()}


def stratified_group_folds(records, n_folds=5, seed=0):
    """Assign every patient to exactly one fold.

    Patients are bucketed by their worst grade and dealt round-robin into
    folds, which keeps the rare severe grades evenly spread instead of leaving
    them to chance.
    """
    groups = patient_groups(records)
    by_grade = defaultdict(list)
    for pid, (_, worst) in groups.items():
        by_grade[worst].append(pid)

    rng = np.random.default_rng(seed)
    fold_of_patient = {}
    for grade in sorted(by_grade):
        pids = by_grade[grade]
        rng.shuffle(pids)
        # Offset the starting fold per grade so small strata do not all pile
        # into fold 0.
        offset = rng.integers(0, n_folds)
        for k, pid in enumerate(pids):
            fold_of_patient[pid] = int((k + offset) % n_folds)

    folds = [[] for _ in range(n_folds)]
    for pid, (idx, _) in groups.items():
        folds[fold_of_patient[pid]].extend(idx)
    return [sorted(f) for f in folds]


def train_val_split(records, val_fraction=0.2, seed=0):
    n_folds = max(2, int(round(1.0 / max(val_fraction, 1e-6))))
    folds = stratified_group_folds(records, n_folds=n_folds, seed=seed)
    val = folds[0]
    train = [i for f in folds[1:] for i in f]
    return sorted(train), sorted(val)


def assert_no_patient_leakage(records, train_idx, val_idx):
    """Fail loudly rather than quietly reporting an inflated score."""
    tr = {records[i].patient_id for i in train_idx}
    va = {records[i].patient_id for i in val_idx}
    overlap = tr & va
    if overlap:
        raise AssertionError(
            f"{len(overlap)} patient(s) appear in both train and validation, "
            f"e.g. {sorted(overlap)[:5]}. The reported score would be inflated.")


def class_balanced_weights(records, indices=None, power=0.5):
    """Per-sample weights for a WeightedRandomSampler.

    EyePACS is roughly 73% grade 0 and under 3% grade 4. Training on the raw
    distribution produces a model that is excellent at saying "healthy" and
    close to useless on the grades that cost people their sight.

    Full inverse-frequency weighting (power=1.0) over-corrects: grade-4 images
    then repeat so often the model memorises them. power=0.5 (inverse sqrt
    frequency) is the usual compromise and is the default here.
    """
    indices = list(range(len(records))) if indices is None else list(indices)
    counts = np.zeros(5, dtype=np.float64)
    for i in indices:
        counts[records[i].grade] += 1
    freq = np.maximum(counts, 1.0)
    w = (freq.sum() / freq) ** power
    w = w / w.sum() * 5.0                      # keep weights near 1.0 on average
    return np.array([w[records[i].grade] for i in indices], dtype=np.float64)


def summarise_split(records, train_idx, val_idx):
    def hist(idx):
        h = np.zeros(5, dtype=int)
        for i in idx:
            h[records[i].grade] += 1
        return h
    th, vh = hist(train_idx), hist(val_idx)
    lines = ["         " + " ".join(f"{g:>7}" for g in range(5)) + "     total",
             "  train: " + " ".join(f"{v:>7}" for v in th) + f" {th.sum():>9}",
             "  val  : " + " ".join(f"{v:>7}" for v in vh) + f" {vh.sum():>9}"]
    return "\n".join(lines)
