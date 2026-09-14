"""Metric and ordinal-decoding tests.

These need no dataset, no GPU and no trained model, so they run in
milliseconds and guard the parts of the pipeline that are easiest to get
silently wrong.
"""
import numpy as np
import pytest

from dr import metrics as M


# ------------------------------------------------------------------- QWK
def test_qwk_perfect_and_chance():
    y = np.array([0, 1, 2, 3, 4, 0, 2, 4])
    assert M.quadratic_weighted_kappa(y, y) == pytest.approx(1.0)
    # Systematically inverted predictions must score well below zero.
    assert M.quadratic_weighted_kappa(y, 4 - y) < -0.5


def test_qwk_penalises_distant_errors_more():
    """The whole reason for using QWK: grades are ordinal. Confusing 0 with 4
    must cost far more than confusing 3 with 4, though both are one wrong
    answer and score identically under plain accuracy."""
    y = np.array([0, 1, 2, 3, 4] * 4)
    near = y.copy(); near[0] = 1        # true 0 predicted 1
    far = y.copy(); far[0] = 4          # true 0 predicted 4
    acc_near = (near == y).mean()
    acc_far = (far == y).mean()
    assert acc_near == acc_far, "accuracy should be blind to the difference"
    assert (M.quadratic_weighted_kappa(y, near)
            > M.quadratic_weighted_kappa(y, far))


def test_qwk_handles_degenerate_single_class():
    """A model that has collapsed to one class must not score 1.0."""
    y = np.array([0, 1, 2, 3, 4])
    collapsed = np.zeros_like(y)
    k = M.quadratic_weighted_kappa(y, collapsed)
    assert k == pytest.approx(0.0), k
    # But genuinely unanimous agreement is perfect.
    same = np.array([2, 2, 2])
    assert M.quadratic_weighted_kappa(same, same) == pytest.approx(1.0)


def test_qwk_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 400)
    p = np.clip(y + rng.integers(-2, 3, 400), 0, 4)
    from sklearn.metrics import cohen_kappa_score
    assert M.quadratic_weighted_kappa(y, p) == pytest.approx(
        cohen_kappa_score(y, p, weights="quadratic"), abs=1e-9)


# ------------------------------------------------------- screening metrics
def test_referable_metrics_use_the_right_cut_point():
    y = np.array([0, 1, 2, 3, 4])
    p = np.array([0, 1, 1, 3, 4])        # one referable case missed (true 2 -> 1)
    r = M.binary_screening_metrics(y, p, M.REFERABLE_THRESHOLD)
    assert r["tp"] == 2 and r["fn"] == 1 and r["tn"] == 2 and r["fp"] == 0
    assert r["sensitivity"] == pytest.approx(2 / 3)
    assert r["specificity"] == pytest.approx(1.0)


def test_summarise_is_json_safe():
    import json
    s = M.summarise([0, 1, 2, 3, 4], [0, 1, 2, 3, 3])
    json.dumps(s)                        # must not raise on numpy scalars
    assert s["n"] == 5


# ---------------------------------------------------------- CORAL decoding
def test_coral_decode_counts_firing_units():
    # 4 cumulative units, all strongly positive -> grade 4
    assert M.coral_logits_to_grade(np.array([8.0, 8.0, 8.0, 8.0]))[0] == 4
    # none fire -> grade 0
    assert M.coral_logits_to_grade(np.array([-8.0, -8.0, -8.0, -8.0]))[0] == 0
    # first two fire -> grade 2
    assert M.coral_logits_to_grade(np.array([8.0, 8.0, -8.0, -8.0]))[0] == 2


def test_coral_distribution_is_a_valid_distribution():
    rng = np.random.default_rng(3)
    for _ in range(50):
        logits = rng.normal(0, 4, 4)
        d = M.coral_logits_to_distribution(logits)
        assert d.shape == (5,)
        assert np.all(d >= 0), d
        assert d.sum() == pytest.approx(1.0)


def test_coral_distribution_survives_non_monotone_logits():
    """A network can emit P(>1)=0.9 alongside P(>2)=0.95, which is
    incoherent. Naive differencing then yields a NEGATIVE probability, which
    would propagate into the fused confidence and the UI bar."""
    logits = np.array([2.2, 2.9, -1.0, -3.0])     # unit 1 < unit 2: impossible
    d = M.coral_logits_to_distribution(logits)
    assert np.all(d >= 0)
    assert d.sum() == pytest.approx(1.0)


def test_coral_grade_is_derived_from_the_distribution_it_reports():
    """The headline grade and the probability bar shown beside it must come
    from one computation.

    The grade is the MEAN of the distribution (QWK-optimal, since kappa
    penalises squared distance), not its mode -- on a broad posterior those
    differ by up to two grades, and a report whose stated grade contradicts its
    own chart is not explainable.
    """
    rng = np.random.default_rng(11)
    for _ in range(200):
        logits = rng.normal(0, 3, 4)
        d = M.coral_logits_to_distribution(logits)
        expected = float((d * np.arange(5)).sum())
        assert M.coral_expected_grade(logits)[0] == pytest.approx(expected)
        assert M.coral_logits_to_grade(logits)[0] == int(np.clip(round(expected), 0, 4))


# -------------------------------------------------------------- thresholds
def test_threshold_optimisation_beats_naive_rounding():
    """Regression scores on an imbalanced set sit low; plain rounding then
    under-grades disease. Fitted cut-points must recover it."""
    rng = np.random.default_rng(7)
    y = np.concatenate([np.zeros(300), np.ones(90), np.full(60, 2),
                        np.full(30, 3), np.full(20, 4)]).astype(int)
    # Simulate a model whose scores are compressed toward zero.
    score = y * 0.55 + rng.normal(0, 0.25, y.size)

    naive = M.quadratic_weighted_kappa(y, np.clip(np.round(score), 0, 4).astype(int))
    th, fitted = M.optimise_thresholds(y, score)
    assert fitted > naive, (fitted, naive)
    assert np.all(np.diff(th) > 0), "cut-points must stay ordered"


def test_apply_thresholds_is_monotone():
    th = [0.5, 1.5, 2.5, 3.5]
    scores = np.linspace(-2, 6, 200)
    grades = M.apply_thresholds(scores, th)
    assert np.all(np.diff(grades) >= 0)
    assert grades.min() == 0 and grades.max() == 4


def test_apply_thresholds_sorts_unordered_cut_points():
    a = M.apply_thresholds([1.2], [0.5, 3.5, 1.5, 2.5])
    b = M.apply_thresholds([1.2], [0.5, 1.5, 2.5, 3.5])
    assert a == b
