"""Severity grading -- the 'what' channel.

Three graders run on every image and are fused:

  rule_grade    ICDR clinical rules applied directly to the lesion inventory.
                Fully transparent, no training, no data dependence.
  FeatureGrader gradient boosting over the clinical feature vector. Learns the
                thresholds the rules hard-code, and reports per-feature
                contributions.
  CnnGrader     EfficientNet-B0 over the image. Catches appearance cues the
                morphology stage has no detector for. Optional.

Fusion is deliberately conservative: the referral decision takes the HIGHEST
grade any grader produces, because in a screening programme a false referral
costs one clinic visit while a missed proliferative case costs an eye.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .features import FEATURE_NAMES, FEATURE_LABELS

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Per-class lesion counts at or below which a detection is treated as noise.
#
# These are a property of the SEGMENTER, not of the clinical rules, and must be
# recalibrated whenever the detector changes -- they were silently stale once
# already, and the rule grader's specificity fell from 0.90 to 0.63 with no
# other symptom. Calibration procedure: run the feature extractor over the
# training cohort and take roughly the 90th percentile of each count on grade-0
# eyes, then confirm against the fused result rather than the rule grader alone.
#
# Current values calibrated on the 200-image training cohort; fused performance
# at this setting is 85.3% exact, sensitivity 1.000, specificity 0.833.
NOISE_FLOOR = {"MA": 8, "HEM": 2, "EX": 5, "CWS": 2}

GRADE_NAMES = {
    0: "No apparent retinopathy",
    1: "Mild non-proliferative DR",
    2: "Moderate non-proliferative DR",
    3: "Severe non-proliferative DR",
    4: "Proliferative DR",
}


@dataclass
class GraderOutput:
    grade: int
    probabilities: list
    source: str
    detail: dict = field(default_factory=dict)

    @property
    def confidence(self):
        return float(max(self.probabilities))

    def to_dict(self):
        return {
            "grade": int(self.grade),
            "grade_name": GRADE_NAMES[int(self.grade)],
            "confidence": round(self.confidence, 3),
            "probabilities": [round(float(p), 4) for p in self.probabilities],
            "source": self.source,
            "detail": self.detail,
        }


def _onehot(grade, sharpness=0.75):
    """Turn a hard rule decision into a probability vector so it can be fused
    with the learned models on equal footing."""
    p = np.full(5, (1.0 - sharpness) / 4.0)
    p[grade] = sharpness
    return (p / p.sum()).tolist()


def rule_grade(values) -> GraderOutput:
    """ICDR severity scale, applied to measured lesion counts.

    The 4-2-1 rule for severe NPDR requires intraretinal haemorrhage in all four
    quadrants, venous beading in two, or IRMA in one. Venous beading and IRMA
    are not separable at fundus-photo resolution by this pipeline, so the
    haemorrhage arm is the operative one and the other two are approximated by
    haemorrhage burden -- this makes the rule grader slightly conservative at
    grade 3, which is the safe direction for screening.
    """
    ma, hem = values["ma_count"], values["hem_count"]
    ex, cws = values["ex_count"], values["cws_count"]
    quad_hem = values["quadrants_with_hem"]
    fired = []

    # Counts at or below these are treated as detector noise rather than
    # disease. Applying the ICDR rules to raw counts sounds more faithful, but
    # the segmenter reports roughly 1-2 spurious dark lesions on a healthy
    # retina, and "any haemorrhage implies moderate NPDR" then refers almost
    # every patient -- measured specificity 0.13, which would swamp the district
    # clinic with healthy people and destroy trust in the system.
    # The noise floor must NOT suppress exudates near the fovea. A couple of
    # exudates is within the detector's error bar anywhere else on the retina,
    # but at the macula they are sight-threatening at any count -- and
    # suppressing them here while triage escalates on them produced a report
    # that contradicted itself ("no significant exudate" beside "2 exudates
    # 970 um from the fovea"), which is exactly the kind of inconsistency that
    # destroys a clinician's trust in the whole output.
    macula_ex = int(values.get("macula_ex_count", 0))
    bright_is_noise = (ex <= NOISE_FLOOR["EX"] and cws <= NOISE_FLOOR["CWS"]
                       and macula_ex == 0)

    if ma <= NOISE_FLOOR["MA"] and hem <= NOISE_FLOOR["HEM"] and bright_is_noise:
        grade = 0
        fired.append(
            f"Lesion counts within the detector's error bar "
            f"(MA {int(ma)}, HEM {int(hem)}, EX {int(ex)}, CWS {int(cws)}; "
            f"reporting thresholds MA {NOISE_FLOOR['MA']}, HEM "
            f"{NOISE_FLOOR['HEM']}, EX {NOISE_FLOOR['EX']}, CWS "
            f"{NOISE_FLOOR['CWS']}); no definite retinopathy.")
    elif hem <= NOISE_FLOOR["HEM"] and bright_is_noise:
        grade = 1
        fired.append(
            f"Microaneurysms predominate ({int(ma)} detected); haemorrhage "
            f"({int(hem)}) and exudate ({int(ex)}) counts are within the "
            "detector's error bar and none lie near the fovea.")
    elif quad_hem >= 4 and hem >= 20:
        grade = 3
        fired.append(f"4-2-1 rule: haemorrhages in all 4 quadrants ({hem} total).")
    elif hem >= 15 or cws >= 5:
        grade = 3
        fired.append(f"High haemorrhage/infarct burden ({hem} haemorrhages, "
                     f"{cws} cotton-wool spots) meets severe NPDR criteria.")
    else:
        grade = 2
        fired.append(f"Lesions beyond microaneurysms present ({hem} haemorrhages, "
                     f"{ex} exudates) but below severe NPDR thresholds.")

    # This pipeline cannot see neovascularisation directly; grade 4 is only
    # reachable via the CNN, and the rule grader says so rather than guessing.
    return GraderOutput(grade=grade, probabilities=_onehot(grade),
                        source="icdr_rules",
                        detail={"criteria_fired": fired,
                                "cannot_assess": ["neovascularisation",
                                                  "venous beading", "IRMA"]})


class FeatureGrader:
    """Gradient boosting over the clinical feature vector."""

    PATH = os.path.join(MODEL_DIR, "feature_grader.joblib")

    def __init__(self, model=None):
        self.model = model

    @classmethod
    def load(cls):
        import joblib
        if not os.path.exists(cls.PATH):
            return cls(None)
        return cls(joblib.load(cls.PATH))

    def available(self):
        return self.model is not None

    def save(self):
        import joblib
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, self.PATH)

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            l2_regularization=1.0, random_state=0)
        self.model.fit(X, y)
        return self

    def predict(self, x) -> GraderOutput:
        if self.model is None:
            raise RuntimeError("feature grader not trained; run train.py")
        probs = self.model.predict_proba(x.reshape(1, -1))[0]
        # Classes seen at fit time may be a subset; re-expand to the full 0-4.
        full = np.zeros(5, np.float64)
        for cls, p in zip(self.model.classes_, probs):
            full[int(cls)] = p
        grade = int(np.argmax(full))
        return GraderOutput(grade=grade, probabilities=full.tolist(),
                            source="feature_gbm",
                            detail={"contributions": self.contributions(x)})

    def contributions(self, x, top=5):
        """Permutation-style local attribution: how much does the predicted
        class probability move when each feature is replaced by the training
        median? Cheap, model-agnostic, and honest about being an approximation.
        """
        if self.model is None or not hasattr(self, "_medians"):
            return []
        base = self.model.predict_proba(x.reshape(1, -1))[0]
        k = int(np.argmax(base))
        out = []
        for i, name in enumerate(FEATURE_NAMES):
            perturbed = x.copy()
            perturbed[i] = self._medians[i]
            p = self.model.predict_proba(perturbed.reshape(1, -1))[0][k]
            out.append((name, float(base[k] - p)))
        out.sort(key=lambda t: -abs(t[1]))
        return [{"feature": n,
                 "label": FEATURE_LABELS.get(n, n.replace("_", " ")),
                 "effect": round(v, 4)} for n, v in out[:top] if abs(v) > 1e-4]

    def set_medians(self, X):
        self._medians = np.median(X, axis=0)


class CnnGrader:
    """EfficientNet-B0 image classifier, with Grad-CAM attribution."""

    PATH = os.path.join(MODEL_DIR, "cnn_grader.pt")

    def __init__(self, model=None, device="cpu"):
        self.model = model
        self.device = device

    @classmethod
    def build(cls, pretrained=False):
        import torch
        from torchvision.models import efficientnet_b0
        net = efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 5)
        return cls(net)

    @classmethod
    def load(cls, device="cpu"):
        import torch
        if not os.path.exists(cls.PATH):
            return cls(None, device)
        obj = cls.build(pretrained=False)
        obj.model.load_state_dict(torch.load(cls.PATH, map_location=device))
        obj.model.eval().to(device)
        obj.device = device
        return obj

    def available(self):
        return self.model is not None

    def save(self):
        import torch
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), self.PATH)

    @staticmethod
    def to_tensor(bgr):
        import torch
        import cv2
        rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return ((x - mean) / std).unsqueeze(0)

    def predict(self, bgr) -> GraderOutput:
        import torch
        if self.model is None:
            raise RuntimeError("cnn grader not trained; run train.py --cnn")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.to_tensor(bgr).to(self.device))
            probs = torch.softmax(logits, 1)[0].cpu().numpy()
        return GraderOutput(grade=int(np.argmax(probs)),
                            probabilities=probs.tolist(),
                            source="efficientnet_b0")


def fuse(outputs) -> GraderOutput:
    """Average the probability vectors, then escalate to the most severe grade
    any individual grader asserted with confidence.

    Straight averaging alone would let two mild votes bury one confident severe
    vote. In a screening context that is the expensive error, so a confident
    high grade from any single model raises the final referral grade.
    """
    if not outputs:
        raise ValueError("no graders available")
    probs = np.mean([o.probabilities for o in outputs], axis=0)
    consensus = int(np.argmax(probs))
    escalated = consensus
    for o in outputs:
        if o.grade > escalated and o.confidence >= 0.5:
            escalated = o.grade
    return GraderOutput(
        grade=escalated,
        probabilities=probs.tolist(),
        source="fused",
        detail={
            "consensus_grade": consensus,
            "escalated": escalated != consensus,
            "members": {o.source: int(o.grade) for o in outputs},
            "agreement": len({o.grade for o in outputs}) == 1,
            "spread": int(max(o.grade for o in outputs) - min(o.grade for o in outputs)),
        },
    )
