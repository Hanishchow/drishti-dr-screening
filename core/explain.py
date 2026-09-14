"""Explainability layer.

Three independent kinds of explanation, because they fail in different ways:

  visual    lesion overlay -- deterministic, pixel-exact, shows WHAT was measured
  saliency  Grad-CAM -- shows where the CNN actually looked
  textual   narrative built from measured quantities, not from model internals

The fourth output is the one that matters most in deployment: an agreement
score between the CNN's attention and the segmenter's lesions. A high grade
produced while looking at a region with no detectable lesions is exactly the
failure mode that erodes clinician trust, and it is detectable.
"""
import cv2
import numpy as np

from .features import FEATURE_LABELS
from .grade import GRADE_NAMES

# BGR. Dark lesions warm, bright lesions cool, so the two families stay
# distinguishable for red-green colour-blind viewers.
COLOURS = {
    "MA": (60, 80, 240),      # red-orange
    "HEM": (40, 30, 160),     # deep red
    "EX": (40, 210, 240),     # amber
    "CWS": (200, 200, 90),    # pale cyan
}
LESION_FULL_NAME = {
    "MA": "Microaneurysm", "HEM": "Haemorrhage",
    "EX": "Hard exudate", "CWS": "Cotton-wool spot",
}


def render_overlay(prep, lesion_map, show_vessels=False, show_anatomy=True):
    """Draw the lesion inventory onto the colour fundus image."""
    out = prep.bgr.copy()
    if show_vessels:
        out[lesion_map.vessels > 0] = (0, 255, 0)

    for kind in ("CWS", "EX", "HEM", "MA"):   # small lesions drawn last, on top
        mask = lesion_map.overlays.get(kind)
        if mask is None or not mask.any():
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (x, y), r = cv2.minEnclosingCircle(c)
            # Draw a ring around the lesion rather than filling it, so the
            # underlying retina stays visible for the clinician to check.
            cv2.circle(out, (int(x), int(y)), int(max(r, 2) + 3), COLOURS[kind], 1,
                       cv2.LINE_AA)

    if show_anatomy:
        dx, dy, dr = lesion_map.optic_disc
        cv2.circle(out, (dx, dy), dr, (255, 255, 255), 1, cv2.LINE_AA)
        mx, my = lesion_map.macula
        cv2.circle(out, (mx, my), int(1500.0 / prep.microns_per_px),
                   (180, 180, 180), 1, cv2.LINE_AA)
        cv2.drawMarker(out, (mx, my), (220, 220, 220), cv2.MARKER_CROSS, 9, 1)
    return out


def render_legend_counts(lesion_map):
    return [{"kind": k, "name": LESION_FULL_NAME[k],
             "colour": "#%02x%02x%02x" % (COLOURS[k][2], COLOURS[k][1], COLOURS[k][0]),
             "count": v}
            for k, v in lesion_map.counts().items()]


def grad_cam(cnn, bgr, target_class=None):
    """Grad-CAM on the last convolutional block of EfficientNet-B0.

    Returns a float map in [0,1] at the working resolution, or None if no CNN
    is loaded -- the rest of the report must still render without it.
    """
    if cnn is None or not cnn.available():
        return None
    import torch

    model = cnn.model
    model.eval()
    activations, gradients = {}, {}
    target_layer = model.features[-1]

    def fwd_hook(_m, _i, o):
        activations["v"] = o

    def bwd_hook(_m, _gi, go):
        gradients["v"] = go[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        x = cnn.to_tensor(bgr).to(cnn.device)
        logits = model(x)
        cls = int(torch.argmax(logits, 1)) if target_class is None else int(target_class)
        model.zero_grad()
        logits[0, cls].backward()
        act, grad = activations["v"][0], gradients["v"][0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(0)).detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()

    if cam.max() <= 0:
        return None
    cam = cam / cam.max()
    return cv2.resize(cam, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)


def render_heatmap(bgr, cam, alpha=0.45):
    if cam is None:
        return None
    hm = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, hm, alpha, 0)


def attention_agreement(cam, lesion_map, prep, top_frac=0.15):
    """Does the CNN's attention land where the lesions actually are?

    Computes the fraction of the CNN's most-attended area that contains
    segmented lesions, against the base rate for a random region of the same
    size. A lift near or below 1.0 means the network reached its grade by
    looking at something the segmenter cannot corroborate -- which is a reason
    to route the case to a human, not to suppress the result.
    """
    if cam is None:
        return None
    lesion_any = np.zeros(prep.green.shape, np.uint8)
    for k in ("MA", "HEM", "EX", "CWS"):
        m = lesion_map.overlays.get(k)
        if m is not None:
            lesion_any |= m
    # Give each lesion a small catchment: Grad-CAM is coarse (7x7 upsampled).
    lesion_any = cv2.dilate(lesion_any, np.ones((9, 9), np.uint8))

    inside = prep.mask > 0
    if not inside.any() or not lesion_any.any():
        return {"lift": None, "hot_lesion_fraction": 0.0,
                "base_rate": 0.0, "verdict": "no_lesions_to_corroborate"}

    vals = cam[inside]
    thr = float(np.quantile(vals, 1.0 - top_frac))
    hot = (cam >= thr) & inside
    hot_frac = float((lesion_any[hot] > 0).mean()) if hot.any() else 0.0
    base = float((lesion_any[inside] > 0).mean())
    lift = hot_frac / base if base > 1e-9 else None

    if lift is None:
        verdict = "indeterminate"
    elif lift >= 2.0:
        verdict = "attention_matches_lesions"
    elif lift >= 1.2:
        verdict = "partial_match"
    else:
        verdict = "attention_unexplained"
    return {"lift": None if lift is None else round(lift, 2),
            "hot_lesion_fraction": round(hot_frac, 4),
            "base_rate": round(base, 4),
            "verdict": verdict}


def _fmt(v):
    return f"{v:.0f}" if isinstance(v, float) and v >= 10 else (
        f"{v:.2f}" if isinstance(v, float) else str(v))


def narrative(values, rule_out, fused, agreement=None, quality=None):
    """Plain-language rationale assembled from measured quantities.

    Deliberately not generated from model weights: every sentence cites a number
    the clinician can verify on the overlay image.
    """
    lines = []
    g = fused.grade
    lines.append(f"Assessment: {GRADE_NAMES[g]} (ICDR grade {g}), "
                 f"confidence {fused.confidence:.0%}.")

    inventory = []
    for key, label in (("ma_count", "microaneurysm"), ("hem_count", "haemorrhage"),
                       ("ex_count", "hard exudate"), ("cws_count", "cotton-wool spot")):
        n = int(values[key])
        if n:
            inventory.append(f"{n} {label}{'s' if n != 1 else ''}")
    lines.append("Measured findings: " + (", ".join(inventory) if inventory
                                          else "no retinal lesions detected") + ".")

    for c in rule_out.detail.get("criteria_fired", []):
        lines.append("Clinical rule: " + c)

    if int(values["macula_ex_count"]) > 0:
        lines.append(
            f"Macular involvement: {int(values['macula_ex_count'])} exudate(s) within "
            f"1500 um of the fovea, closest at {values['min_ex_dist_to_macula_um']:.0f} um. "
            "This raises the risk of clinically significant macular oedema and shortens "
            "the referral window independently of the severity grade.")

    if int(values["quadrants_with_hem"]) >= 3:
        lines.append(f"Haemorrhages are distributed across "
                     f"{int(values['quadrants_with_hem'])} of 4 retinal quadrants, "
                     "which is a severity criterion in the ICDR 4-2-1 rule.")

    members = fused.detail.get("members", {})
    if len(members) > 1:
        if fused.detail.get("agreement"):
            lines.append(f"All {len(members)} independent graders agreed on grade {g}.")
        else:
            detail = ", ".join(f"{k} said {v}" for k, v in members.items())
            lines.append(f"Graders disagreed ({detail}); the more severe grade was "
                         "carried forward, which is the safe direction for screening.")

    if agreement and agreement.get("verdict") == "attention_unexplained":
        lines.append("Trust check FAILED: the neural network's attention did not "
                     "concentrate on any segmented lesion, so its grade is not "
                     "corroborated by measurable evidence. Human review required.")
    elif agreement and agreement.get("lift"):
        lines.append(f"Trust check: network attention was {agreement['lift']}x more "
                     "concentrated on segmented lesions than chance.")

    if quality and not quality.passed:
        lines.append("Caution: this grade was produced from an image that failed the "
                     "quality gate and should not be acted on without recapture.")

    return lines
