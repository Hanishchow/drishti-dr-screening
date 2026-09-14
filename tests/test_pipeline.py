"""Behavioural tests for the screening pipeline.

These assert properties that must hold for the system to be safe, not exact
numbers that would break on every retune:

  * a healthy eye must be able to return zero lesions (no forced detections)
  * lesion burden must increase with severity
  * the quality gate must reject the degradations it claims to detect
  * geometry must round-trip, since a silent misalignment destroys recall
  * triage must escalate on macular involvement regardless of grade
  * a quality failure must block grading unless explicitly overridden

Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest

from core import explain, features, grade, lesions, preprocess, quality, triage
from core.pipeline import Screener
from data.synth import generate


# --------------------------------------------------------------- quality gate
def test_clean_images_pass_across_all_grades():
    for g in range(5):
        for seed in (1, 2, 3):
            q = quality.assess(generate(g, seed=100 + seed + 7 * g)[0])
            assert q.passed, f"grade {g} seed {seed} falsely rejected: {q.failures}"


@pytest.mark.parametrize("kwargs,expected", [
    (dict(blur=7.0), "blurred"),
    (dict(vignette=0.9), "uneven_illumination"),
    (dict(exposure=0.22), "underexposed"),
    (dict(glare=1.5), "glare"),
])
def test_degradations_are_detected(kwargs, expected):
    fails = set()
    for seed in (1, 2, 3):
        fails |= set(quality.assess(generate(2, seed=seed, **kwargs)[0]).failures)
    assert expected in fails


def test_sharpness_decreases_monotonically_with_blur():
    """Guards the focus metric. Earlier implementations were non-monotonic --
    a badly blurred image scored as sharper than a clean one."""
    vals = []
    for b in (0.0, 2.0, 4.0, 6.0, 8.0):
        kw = dict(blur=b) if b else {}
        s = [quality.assess(generate(2, seed=i, **kw)[0]).metrics["sharpness"]
             for i in (1, 2, 3)]
        vals.append(float(np.mean(s)))
    assert all(a > b for a, b in zip(vals, vals[1:])), vals


def test_sharpness_is_exposure_invariant():
    """A dark image is not a blurred image; the metric must not confuse them."""
    bright = np.mean([quality.assess(generate(2, seed=i)[0]).metrics["sharpness"]
                      for i in (1, 2, 3)])
    dim = np.mean([quality.assess(generate(2, seed=i, exposure=0.4)[0]).metrics["sharpness"]
                   for i in (1, 2, 3)])
    assert abs(bright - dim) / bright < 0.15


# ------------------------------------------------------------------- geometry
def test_crop_transform_round_trips():
    """The prepared image must be able to map original-resolution annotations
    into its own frame. A few percent of drift here silently destroyed lesion
    recall before this was made explicit."""
    import cv2
    for seed in (1, 5, 9):
        bgr, _ = generate(3, seed=seed)
        prep = preprocess.prepare(bgr)
        orig_fov = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 8).astype(np.uint8)
        mapped = prep.map_from_original(orig_fov)
        ref = cv2.dilate(prep.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        iou = (mapped & ref).sum() / max((mapped | ref).sum(), 1)
        assert iou > 0.95, f"seed {seed}: FOV IoU {iou:.3f}"


# ------------------------------------------------------------------- lesions
def test_featureless_retina_returns_no_lesions():
    """The threshold must be absolute, not a percentile.

    A percentile threshold declares a fixed fraction of pixels to be lesions no
    matter what the image contains, so "no disease" is not even representable.
    A smooth synthetic retina with no vessels and no lesions is the direct test
    of that property: the honest answer is zero.
    """
    import cv2
    from data.synth import SIZE
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    r = np.sqrt((xx - SIZE / 2) ** 2 + (yy - SIZE / 2) ** 2)
    fov = r <= SIZE * 0.47
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    falloff = np.clip(1.0 - (r / (SIZE * 0.47)) ** 2 * 0.55, 0, 1)
    for c, v in enumerate((38.0, 78.0, 185.0)):
        img[:, :, c] = v * falloff
    img[~fov] = 0
    prep = preprocess.prepare(np.clip(img, 0, 255).astype(np.uint8))
    counts = lesions.detect(prep).counts()
    assert sum(counts.values()) == 0, counts


def test_healthy_eye_stays_within_the_graders_noise_floor():
    """The detector reports a few spurious specks on a real healthy retina.
    That is tolerable only because the grader knows the size of that error bar,
    so the two must be kept consistent -- when they drifted apart, the rule
    grader's specificity silently fell from 0.90 to 0.63."""
    for seed in (1, 2, 3, 4, 5, 6):
        prep = preprocess.prepare(generate(0, seed=seed)[0])
        c = lesions.detect(prep).counts()
        for kind, floor in grade.NOISE_FLOOR.items():
            assert c[kind] <= floor, (
                f"seed {seed}: {c[kind]} {kind} on a healthy eye exceeds the "
                f"grader's noise floor of {floor}; recalibrate NOISE_FLOOR")


def test_lesion_burden_increases_with_severity():
    def burden(g):
        tot = []
        for seed in (1, 2, 3):
            prep = preprocess.prepare(generate(g, seed=seed * 10 + g)[0])
            tot.append(sum(lesions.detect(prep).counts().values()))
        return float(np.mean(tot))

    low, high = burden(0), burden(4)
    assert high > low * 3, f"grade 0 -> {low}, grade 4 -> {high}"


def test_lesion_measurements_are_physical():
    prep = preprocess.prepare(generate(4, seed=2)[0])
    for l in lesions.detect(prep).lesions:
        assert l.area_um2 > 0
        assert l.major_axis_um >= l.minor_axis_um
        # Bounded at every size: the perimeter-based formula returned pi for a
        # 2-pixel blob, which mislabelled specks as microaneurysms.
        assert 0.0 <= l.circularity <= 1.0
        assert 0 <= l.dist_to_macula_um <= 20000


# -------------------------------------------------------------------- grading
def test_rule_grader_reports_what_it_cannot_assess():
    """Honesty about the limits of the method is part of explainability:
    neovascularisation is not visible to this pipeline and must be declared."""
    prep = preprocess.prepare(generate(0, seed=1)[0])
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


# --------------------------------------------------------------------- triage
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


# ------------------------------------------------------------------- pipeline
def test_failed_quality_blocks_grading_but_override_works():
    bgr, _ = generate(2, seed=4, blur=8.0)
    s = Screener(use_cnn=False)
    blocked = s.run(bgr, want_images=False)
    assert not blocked.graded and blocked.grade is None

    forced = s.run(bgr, want_images=False, force_grade=True)
    assert forced.graded and forced.grade is not None
    assert any("quality gate" in line for line in forced.explanation)


def test_report_is_self_consistent():
    s = Screener(use_cnn=False)
    r = s.run(generate(3, seed=6)[0], want_images=False)
    assert r.graded
    assert sum(r.lesion_counts.values()) == len(r.lesions)
    assert r.features["total_count"] == sum(r.lesion_counts.values())
    assert 0 <= r.grade <= 4
    assert r.explanation and r.triage["urgency"] in (
        "routine", "soon", "urgent", "emergency")


def test_explanation_cites_measured_counts():
    """Every sentence must be checkable against the overlay, so the narrative
    has to quote the actual inventory rather than model internals."""
    s = Screener(use_cnn=False)
    r = s.run(generate(4, seed=8)[0], want_images=False)
    joined = " ".join(r.explanation)
    assert "Measured findings" in joined
    if r.lesion_counts["HEM"]:
        assert str(r.lesion_counts["HEM"]) in joined


# ---------------------------------------------------------------- simulation
def test_simulation_counts_unseen_patients_as_window_breaches():
    """Without end-of-run accounting, an arm that never reaches its queue
    scores 100% simply because unreached patients are excluded."""
    from sim.district import Config, simulate
    cfg = Config(phcs=30, ophthalmologists=1, days=60)
    m = simulate(cfg, "manual")
    assert m.unseen > 0
    urgent = m.within_window["urgent"]
    assert urgent and float(np.mean(urgent)) < 1.0


def test_ai_arm_reduces_specialist_reading_load():
    from sim.district import Config, compare
    out = compare(Config(days=60))
    assert out["summary"]["read_workload_reduction"] > 0.5


# ------------------------------------------------------------------- anatomy
def test_optic_disc_and_macula_are_localised():
    """Both anchor the distance-to-fovea measurement that drives macular
    escalation in triage, so an error here is a silent clinical error.

    Regressions this guards against:
      * picking the globally brightest pixel finds the posterior pole, not the
        disc, because a healthy retina is brightest at its centre
      * taking the darkest point of an annulus finds the vignetted periphery,
        not the fovea
    """
    fovea_radius_px = 1500.0 / (13000.0 / 512)     # ~59 px
    disc_err, mac_err = [], []
    for g in range(5):
        for s in range(1, 7):
            bgr, truth = generate(g, seed=s * 7 + g)
            prep = preprocess.prepare(bgr)
            lm = lesions.detect(prep)
            tdx, tdy = prep.map_point_from_original(*truth["optic_disc"][:2])
            tmx, tmy = prep.map_point_from_original(*truth["macula"])
            disc_err.append(np.hypot(lm.optic_disc[0] - tdx, lm.optic_disc[1] - tdy))
            mac_err.append(np.hypot(lm.macula[0] - tmx, lm.macula[1] - tmy))
    disc_err, mac_err = np.array(disc_err), np.array(mac_err)
    assert np.median(disc_err) < 25, f"disc median error {np.median(disc_err):.1f}px"
    assert np.median(mac_err) < fovea_radius_px, f"macula median {np.median(mac_err):.1f}px"
    assert (disc_err > 100).mean() < 0.15, "too many gross disc failures"


def test_report_never_contradicts_itself_about_macular_exudate():
    """The rule grader's noise floor must not suppress exudates at the fovea.

    Otherwise the report asserts 'no significant exudate' in one sentence while
    triage escalates on 'exudates 970 um from the fovea' in the next -- an
    internal contradiction that would rightly destroy clinical trust.
    """
    vals = {"ma_count": 20, "hem_count": 1, "ex_count": 3, "cws_count": 1,
            "quadrants_with_hem": 1, "macula_ex_count": 2}
    out = grade.rule_grade(vals)
    text = " ".join(out.detail["criteria_fired"])
    assert out.grade >= 2, "macular exudate must not be dismissed as noise"
    assert "none lie near the fovea" not in text
