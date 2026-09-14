"""Pipeline invariants.

Two tiers:

  * Invariant tests on minimal geometric fixtures. These assert properties that
    must hold for ANY image and run anywhere with no dataset.
  * Real-image tests against IDRiD, which skip cleanly when the corpus is
    absent. These are where actual detection quality is measured; run
    `python -m dr.eval_lesions` for the full benchmark.

No figure in this repo is derived from generated images.
"""
import numpy as np
import pytest

from core import explain, features, grade, lesions, preprocess, quality, triage
from tests.fixtures import (blank_retina, requires_idrid, retina_with_spots,
                            sharp_then_blurred)


# ------------------------------------------------------------ quality gate
def test_clean_image_passes():
    q = quality.assess(sharp_then_blurred(0.0))
    assert q.passed, q.failures


@pytest.mark.parametrize("sigma,expected", [(7.0, "blurred"), (9.0, "blurred")])
def test_blur_is_detected(sigma, expected):
    assert expected in quality.assess(sharp_then_blurred(sigma)).failures


def test_underexposure_is_detected():
    dim = (blank_retina().astype(np.float32) * 0.2).astype(np.uint8)
    assert "underexposed" in quality.assess(dim).failures


def test_sharpness_decreases_monotonically_with_blur():
    """Guards the focus metric. Two earlier implementations were non-monotonic:
    contrast-normalised Laplacian variance scored an underexposed frame at 869
    against a clean frame's 85, and a high/mid band ratio inverted entirely
    because sensor noise survives optical blur."""
    vals = [quality.assess(sharp_then_blurred(s)).metrics["sharpness"]
            for s in (0.0, 2.0, 4.0, 6.0, 8.0)]
    assert all(a > b for a, b in zip(vals, vals[1:])), vals


def test_sharpness_is_exposure_invariant():
    """A dark image is not a blurred image."""
    bright = quality.assess(sharp_then_blurred(0.0)).metrics["sharpness"]
    img = (sharp_then_blurred(0.0).astype(np.float32) * 0.45).astype(np.uint8)
    dim = quality.assess(img).metrics["sharpness"]
    assert abs(bright - dim) / bright < 0.2, (bright, dim)


# ---------------------------------------------------------------- geometry
def test_crop_transform_round_trips():
    """The prepared image must map original-resolution annotations into its own
    frame. A few percent of drift here is invisible by eye and destroyed lesion
    recall (F1 0.23 vs 0.78) when ground truth was compared at a tight
    tolerance."""
    import cv2
    for brightness in (0.7, 1.0, 1.3):
        bgr = blank_retina(brightness=brightness)
        prep = preprocess.prepare(bgr)
        orig_fov = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 8).astype(np.uint8)
        mapped = prep.map_from_original(orig_fov)
        ref = cv2.dilate(prep.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        iou = (mapped & ref).sum() / max((mapped | ref).sum(), 1)
        assert iou > 0.95, iou


def test_point_mapping_matches_mask_mapping():
    import cv2
    bgr = blank_retina()
    marker = np.zeros(bgr.shape[:2], np.uint8)
    cv2.circle(marker, (300, 240), 6, 1, -1)
    prep = preprocess.prepare(bgr)
    mapped = prep.map_from_original(marker)
    px, py = prep.map_point_from_original(300, 240)
    ys, xs = np.where(mapped > 0)
    assert xs.size and abs(xs.mean() - px) < 4 and abs(ys.mean() - py) < 4


# ----------------------------------------------------------------- lesions
def test_featureless_retina_returns_no_lesions():
    """The threshold must be absolute, not a percentile.

    A percentile threshold declares a fixed fraction of pixels to be lesions no
    matter what the image contains, so "no disease" is not representable. A
    smooth disc is the direct test: the honest answer is zero.
    """
    counts = lesions.detect(preprocess.prepare(blank_retina())).counts()
    assert sum(counts.values()) == 0, counts


def test_detects_planted_dark_spots():
    img, placed = retina_with_spots(n_dark=8, seed=4)
    lmap = lesions.detect(preprocess.prepare(img))
    found = lmap.counts()["MA"] + lmap.counts()["HEM"]
    assert found >= len(placed) // 2, (found, len(placed))


def test_lesion_burden_increases_with_lesion_count():
    def burden(n):
        img, _ = retina_with_spots(n_dark=n, seed=n)
        return sum(lesions.detect(preprocess.prepare(img)).counts().values())
    assert burden(12) > burden(0)


def test_lesion_measurements_are_physical():
    img, _ = retina_with_spots(n_dark=10, n_bright=6, seed=5)
    for l in lesions.detect(preprocess.prepare(img)).lesions:
        assert l.area_um2 > 0
        assert l.major_axis_um >= l.minor_axis_um
        # Bounded at every size: the perimeter formula 4*pi*A/P^2 returns pi for
        # a 2-pixel blob, which mislabelled specks as confident microaneurysms.
        assert 0.0 <= l.circularity <= 1.0
        assert 0 <= l.dist_to_macula_um <= 20000


def test_healthy_image_stays_within_the_graders_noise_floor():
    """The detector reports the odd spurious speck. That is tolerable only
    because the grader knows the size of that error bar, so the two must be
    kept consistent -- when they drifted apart, rule-grader specificity fell
    from 0.90 to 0.63 with no other symptom."""
    counts = lesions.detect(preprocess.prepare(blank_retina())).counts()
    for kind, floor in grade.NOISE_FLOOR.items():
        assert counts[kind] <= floor


# ---------------------------------------------------------------- grading
def test_rule_grader_reports_what_it_cannot_assess():
    prep = preprocess.prepare(blank_retina())
    _, vals = features.extract(lesions.detect(prep), prep)
    out = grade.rule_grade(vals)
    assert "neovascularisation" in out.detail["cannot_assess"]
    assert out.detail["criteria_fired"]


def test_fusion_escalates_to_the_most_severe_confident_grader():
    mild = grade.GraderOutput(1, grade._onehot(1), "a")
    severe = grade.GraderOutput(4, grade._onehot(4, 0.9), "b")
    fused = grade.fuse([mild, mild, severe])
    assert fused.grade == 4, "a confident severe vote must not be averaged away"
    assert fused.detail["escalated"]


def test_fusion_agreement_flag():
    same = [grade.GraderOutput(2, grade._onehot(2), s) for s in ("a", "b")]
    assert grade.fuse(same).detail["agreement"]
    assert grade.fuse(same).detail["spread"] == 0


# ----------------------------------------------------------------- triage
def test_macular_exudate_escalates_urgency_regardless_of_grade():
    """A grade-2 eye with exudates at the fovea is sight-threatening now."""
    vals = {"macula_ex_count": 3, "quadrants_with_hem": 1}
    fused = grade.GraderOutput(2, grade._onehot(2), "fused",
                               {"members": {}, "spread": 0})
    d = triage.decide(2, vals, fused)
    assert d.urgency == "urgent"
    assert "macular_exudate" in d.escalations
    assert d.days_to_review <= 28


def test_grader_disagreement_forces_human_review():
    fused = grade.GraderOutput(3, grade._onehot(3), "fused",
                               {"members": {"a": 1, "b": 3}, "spread": 2})
    d = triage.decide(3, {"macula_ex_count": 0, "quadrants_with_hem": 0}, fused)
    assert d.needs_human_review


def test_healthy_eye_gets_routine_annual_pathway():
    fused = grade.GraderOutput(0, grade._onehot(0, 0.9), "fused",
                               {"members": {}, "spread": 0})
    d = triage.decide(0, {"macula_ex_count": 0, "quadrants_with_hem": 0}, fused)
    assert d.urgency == "routine" and d.days_to_review == 365


def test_report_never_contradicts_itself_about_macular_exudate():
    """The noise floor must not suppress exudates at the fovea, or the report
    asserts 'no significant exudate' beside 'exudates 970 um from the fovea'."""
    vals = {"ma_count": 20, "hem_count": 1, "ex_count": 3, "cws_count": 1,
            "quadrants_with_hem": 1, "macula_ex_count": 2}
    out = grade.rule_grade(vals)
    assert out.grade >= 2
    assert "none lie near the fovea" not in " ".join(out.detail["criteria_fired"])


# ------------------------------------------------------------- simulation
def test_simulation_counts_unseen_patients_as_window_breaches():
    """Without end-of-run accounting, an arm that never reaches its queue
    scores 100% because unreached patients are excluded."""
    from sim.district import Config, simulate
    m = simulate(Config(phcs=30, ophthalmologists=1, days=60), "manual")
    assert m.unseen > 0
    assert float(np.mean(m.within_window["urgent"])) < 1.0


def test_ai_arm_reduces_specialist_reading_load():
    from sim.district import Config, compare
    assert compare(Config(days=60))["summary"]["read_workload_reduction"] > 0.5


# ------------------------------------------------------- real images only
@requires_idrid
def test_real_fundus_passes_quality_and_grades():
    from dr import datasets as D
    import cv2
    for rec in D.load_idrid_segmentation()[:3]:
        bgr = cv2.imread(rec.image_path, cv2.IMREAD_COLOR)
        assert bgr is not None
        q = quality.assess(bgr)
        assert q.metrics["fov_coverage"] > 0.2, "FOV crop failed on a real image"
        prep = preprocess.prepare(bgr)
        lmap = lesions.detect(prep)
        _, vals = features.extract(lmap, prep)
        assert grade.rule_grade(vals).grade in range(5)


@requires_idrid
def test_real_lesion_detection_beats_chance():
    """Lower bound only. The full benchmark is `python -m dr.eval_lesions`."""
    from dr.eval_lesions import TOLERANCE_UM, aggregate, score_record
    from dr import datasets as D

    tol = max(2, int(round(TOLERANCE_UM / (13000.0 / preprocess.TARGET))))
    scored = [r for r in (score_record(rec, tol)
                          for rec in D.load_idrid_segmentation()[:8]) if r]
    assert scored, "no IDRiD images produced a score"
    report = aggregate(scored)
    assert report["dark"]["recall"] > 0.05, report
