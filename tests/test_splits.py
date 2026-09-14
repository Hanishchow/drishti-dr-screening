"""Split-integrity tests.

These run on synthetic RECORDS (paths and ids only, no images), so they need no
dataset. Their job is to catch the two failure modes that inflate a reported
score without ever raising an error.
"""
import numpy as np
import pytest

from dr import splits
from dr.datasets import Record


def make_records(n_patients=200, seed=0, two_eyes=True):
    """Records shaped like EyePACS: two correlated eyes per patient."""
    rng = np.random.default_rng(seed)
    # Realistic DR imbalance: mostly healthy, few severe.
    p = [0.73, 0.15, 0.07, 0.03, 0.02]
    out = []
    for i in range(n_patients):
        base = int(rng.choice(5, p=p))
        for eye in (["L", "R"] if two_eyes else ["L"]):
            g = int(np.clip(base + rng.integers(-1, 2), 0, 4))
            out.append(Record(f"/img/{i}_{eye}.jpg", g, f"p{i}", "eyepacs", eye))
    return out


def test_folds_never_split_a_patient():
    """The core leakage guard: both eyes of one patient must land in the same
    fold, or the model is validated partly on memorisation."""
    recs = make_records()
    folds = splits.stratified_group_folds(recs, n_folds=5, seed=1)
    seen = {}
    for f_i, fold in enumerate(folds):
        for i in fold:
            pid = recs[i].patient_id
            assert seen.setdefault(pid, f_i) == f_i, f"{pid} spans folds"


def test_folds_partition_every_record_exactly_once():
    recs = make_records()
    folds = splits.stratified_group_folds(recs, n_folds=5, seed=2)
    flat = [i for f in folds for i in f]
    assert sorted(flat) == list(range(len(recs)))


def test_train_val_split_has_no_leakage():
    recs = make_records()
    tr, va = splits.train_val_split(recs, 0.2, seed=3)
    splits.assert_no_patient_leakage(recs, tr, va)
    assert 0.1 < len(va) / len(recs) < 0.35


def test_leakage_assertion_actually_fires():
    """A guard that cannot fail is not a guard."""
    recs = make_records(n_patients=10)
    with pytest.raises(AssertionError, match="both train and validation"):
        splits.assert_no_patient_leakage(recs, [0, 1], [1, 2])


def test_rare_grades_reach_every_fold():
    """Grades 3 and 4 are a few percent of these corpora. If stratification
    fails, a fold can contain almost no severe disease and its sensitivity
    estimate becomes meaningless."""
    recs = make_records(n_patients=400, seed=5)
    folds = splits.stratified_group_folds(recs, n_folds=5, seed=5)
    for f_i, fold in enumerate(folds):
        severe = sum(1 for i in fold if recs[i].grade >= 3)
        assert severe > 0, f"fold {f_i} contains no sight-threatening cases"


def test_stratification_keeps_fold_prevalence_close():
    recs = make_records(n_patients=600, seed=6)
    folds = splits.stratified_group_folds(recs, n_folds=5, seed=6)
    overall = np.mean([r.grade >= 2 for r in recs])
    for fold in folds:
        rate = np.mean([recs[i].grade >= 2 for i in fold])
        assert abs(rate - overall) < 0.06, (rate, overall)


# --------------------------------------------------------- sampler weights
def test_balanced_weights_lift_rare_grades():
    recs = make_records(n_patients=500, seed=7)
    w = splits.class_balanced_weights(recs)
    g = np.array([r.grade for r in recs])
    assert w[g == 4].mean() > w[g == 0].mean() * 2


def test_weight_power_controls_aggressiveness():
    """power=1.0 is full inverse frequency and over-repeats the rare classes;
    0.5 is the intended compromise and must sit between."""
    recs = make_records(n_patients=500, seed=8)
    g = np.array([r.grade for r in recs])

    def ratio(power):
        w = splits.class_balanced_weights(recs, power=power)
        return w[g == 4].mean() / w[g == 0].mean()

    assert ratio(0.0) == pytest.approx(1.0)
    assert 1.0 < ratio(0.5) < ratio(1.0)


def test_weights_align_with_the_indices_given():
    recs = make_records(n_patients=50, seed=9)
    idx = list(range(10, 30))
    w = splits.class_balanced_weights(recs, idx)
    assert len(w) == len(idx)
    assert np.all(w > 0)
