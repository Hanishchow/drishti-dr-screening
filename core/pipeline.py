"""End-to-end screening pipeline.

One entry point, one report. Stages are ordered so the cheapest rejection
happens first: an unusable image is bounced back to the ASHA worker in
milliseconds, before any segmentation or inference cost is paid.
"""
import base64
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import explain, features, grade as grading, lesions, preprocess, quality


def _png_b64(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class Report:
    quality: dict
    graded: bool
    grade: int = None
    grade_name: str = None
    confidence: float = None
    graders: list = field(default_factory=list)
    lesion_counts: dict = field(default_factory=dict)
    lesions: list = field(default_factory=list)
    features: dict = field(default_factory=dict)
    explanation: list = field(default_factory=list)
    attention: dict = None
    triage: dict = None
    images: dict = field(default_factory=dict)
    timing_ms: dict = field(default_factory=dict)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class Screener:
    """Holds the loaded models so they are not re-read per image."""

    def __init__(self, use_cnn=True):
        self.feature_grader = grading.FeatureGrader.load()
        self.cnn = grading.CnnGrader.load() if use_cnn else grading.CnnGrader(None)

    @property
    def loaded(self):
        return {"feature_gbm": self.feature_grader.available(),
                "efficientnet_b0": self.cnn.available()}

    def run(self, bgr, want_images=True, force_grade=False):
        t = {}
        t0 = time.perf_counter()

        q = quality.assess(bgr)
        t["quality"] = (time.perf_counter() - t0) * 1000
        # An image that fails the gate stops here unless the caller explicitly
        # overrides -- the ASHA worker needs the recapture prompt, not a grade
        # derived from an unreadable photo.
        if not q.passed and not force_grade:
            return Report(quality=q.to_dict(), graded=False,
                          images={"input": _png_b64(bgr)} if want_images else {},
                          timing_ms={k: round(v, 1) for k, v in t.items()})

        t1 = time.perf_counter()
        prep = preprocess.prepare(bgr)
        t["preprocess"] = (time.perf_counter() - t1) * 1000

        t1 = time.perf_counter()
        lmap = lesions.detect(prep)
        t["segmentation"] = (time.perf_counter() - t1) * 1000

        t1 = time.perf_counter()
        fvec, fvals = features.extract(lmap, prep)
        outputs = [grading.rule_grade(fvals)]
        if self.feature_grader.available():
            outputs.append(self.feature_grader.predict(fvec))
        if self.cnn.available():
            outputs.append(self.cnn.predict(prep.bgr))
        fused = grading.fuse(outputs)
        t["grading"] = (time.perf_counter() - t1) * 1000

        t1 = time.perf_counter()
        cam = explain.grad_cam(self.cnn, prep.bgr) if self.cnn.available() else None
        agreement = explain.attention_agreement(cam, lmap, prep)
        narrative = explain.narrative(fvals, outputs[0], fused, agreement, q)
        t["explain"] = (time.perf_counter() - t1) * 1000

        from . import triage as triage_mod
        decision = triage_mod.decide(fused.grade, fvals, fused, q, agreement)

        images = {}
        if want_images:
            images["input"] = _png_b64(prep.bgr)
            images["overlay"] = _png_b64(explain.render_overlay(prep, lmap))
            images["vessels"] = _png_b64(
                explain.render_overlay(prep, lmap, show_vessels=True))
            hm = explain.render_heatmap(prep.bgr, cam)
            if hm is not None:
                images["heatmap"] = _png_b64(hm)

        t["total"] = (time.perf_counter() - t0) * 1000
        return Report(
            quality=q.to_dict(),
            graded=True,
            grade=int(fused.grade),
            grade_name=grading.GRADE_NAMES[int(fused.grade)],
            confidence=round(fused.confidence, 3),
            graders=[o.to_dict() for o in outputs] + [fused.to_dict()],
            lesion_counts=lmap.counts(),
            lesions=[l.to_dict() for l in lmap.lesions],
            features={k: (round(float(v), 3) if isinstance(v, float) else v)
                      for k, v in fvals.items()},
            explanation=narrative,
            attention=agreement,
            triage=decision.to_dict(),
            images=images,
            timing_ms={k: round(v, 1) for k, v in t.items()},
        )
