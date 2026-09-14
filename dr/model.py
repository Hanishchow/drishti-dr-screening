"""Grading network: timm backbone + CORAL ordinal head.

Why ordinal rather than 5-way softmax:

A softmax head treats the five ICDR grades as unrelated categories, so
predicting 0 for a grade-4 eye costs it exactly what predicting 3 costs. That
is wrong clinically and wrong for the metric the task is scored on (QWK
penalises squared distance). CORAL (Cao et al., 2020) instead learns K-1
cumulative units -- "is the grade > 0?", "> 1?", "> 2?", "> 3?" -- sharing one
feature vector and one weight vector, differing only in a per-unit bias. That
shared weight is what guarantees the predicted cumulative probabilities stay
monotonic, so the model can never assert P(grade>2) > P(grade>1).

It also gives the triage layer something a softmax cannot: a calibrated
P(grade >= 2), which is precisely the referral decision.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

GRADES = 5


class CoralHead(nn.Module):
    """K-1 cumulative logits from one shared projection plus per-unit biases."""

    def __init__(self, in_features, num_classes=GRADES):
        super().__init__()
        self.fc = nn.Linear(in_features, 1, bias=False)
        # Biases initialised in decreasing order so the head starts out
        # monotone and ordered rather than having to learn the ordering.
        self.bias = nn.Parameter(torch.linspace(1.5, -1.5, num_classes - 1))

    def forward(self, x):
        return self.fc(x) + self.bias          # (N, K-1)


def coral_targets(grades, num_classes=GRADES):
    """Grade -> binary cumulative targets. Grade 3 becomes [1, 1, 1, 0]."""
    levels = torch.arange(num_classes - 1, device=grades.device)[None, :]
    return (grades[:, None] > levels).float()


class CoralLoss(nn.Module):
    """Binary cross-entropy over the cumulative units.

    `importance` lets the sight-threatening boundaries carry more weight than
    the 0-vs-1 boundary. That is a deliberate clinical choice: separating "no
    retinopathy" from "mild" is the least consequential decision the model
    makes, while the >=2 boundary is the referral itself.
    """

    def __init__(self, importance=(1.0, 1.6, 1.6, 1.4)):
        super().__init__()
        self.register_buffer("importance", torch.tensor(importance, dtype=torch.float32))

    def forward(self, logits, grades):
        targets = coral_targets(grades, logits.shape[1] + 1)
        per_unit = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        return (per_unit * self.importance[None, :]).mean()


class DRGrader(nn.Module):
    def __init__(self, backbone="tf_efficientnet_b3_ns", pretrained=True,
                 drop_rate=0.3, num_classes=GRADES, in_chans=3):
        super().__init__()
        import timm
        self.backbone_name = backbone
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0,
            drop_rate=drop_rate, in_chans=in_chans)
        self.head = CoralHead(self.encoder.num_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.head(self.encoder(x))

    @torch.no_grad()
    def predict(self, x):
        """Cumulative logits -> (expected grade, 5-way distribution)."""
        from .metrics import coral_expected_grade, coral_logits_to_distribution
        logits = self(x).float().cpu().numpy()
        return coral_expected_grade(logits), coral_logits_to_distribution(logits)

    def gradcam_layer(self):
        """Last convolutional stage, for Grad-CAM.

        timm backbones expose their stages under different attribute names, so
        the final 4-D-output module is located by inspection rather than by
        hard-coding a path that silently breaks when the backbone changes.
        """
        candidates = [m for m in self.encoder.modules()
                      if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
        if not candidates:
            raise RuntimeError("no convolutional layer found for Grad-CAM")
        return candidates[-1]


def build(backbone="tf_efficientnet_b3_ns", pretrained=True, **kw):
    return DRGrader(backbone=backbone, pretrained=pretrained, **kw)


# ------------------------------------------------------------------- export
def export_onnx(model, path, size=512, opset=17):
    """Export for onnxruntime serving.

    The edge/PHC path runs CPU-only with no torch installed at all, and the GPU
    server uses ONNX for a stable, versioned artefact that does not depend on
    the training code still being importable.
    """
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, size, size)
    common = dict(input_names=["image"], output_names=["cumulative_logits"],
                  dynamic_axes={"image": {0: "batch"},
                                "cumulative_logits": {0: "batch"}},
                  opset_version=opset)
    try:
        # torch>=2.5 defaults to the dynamo exporter, which needs onnxscript.
        # Fall back to the long-stable TorchScript exporter when it is absent,
        # so a missing optional package cannot cost a finished training run
        # its deployable artefact.
        torch.onnx.export(model, dummy, str(path), dynamo=False,
                          do_constant_folding=True, **common)
    except TypeError:
        torch.onnx.export(model, dummy, str(path),
                          do_constant_folding=True, **common)
    return path


class OnnxGrader:
    """Inference-only grader with no torch dependency."""

    def __init__(self, path, providers=None):
        import onnxruntime as ort
        if providers is None:
            available = ort.get_available_providers()
            providers = ([p for p in ("CUDAExecutionProvider",) if p in available]
                         + ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, batch):
        """batch: (N,3,H,W) float32 -> (expected grade, distribution)."""
        from .metrics import coral_expected_grade, coral_logits_to_distribution
        logits = self.session.run(None, {self.input_name:
                                         np.ascontiguousarray(batch, np.float32)})[0]
        return coral_expected_grade(logits), coral_logits_to_distribution(logits)
