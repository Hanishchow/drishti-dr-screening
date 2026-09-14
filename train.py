"""Train the graders on the synthetic cohort.

  python train.py              # feature grader only (seconds)
  python train.py --cnn        # also fine-tune EfficientNet-B0 (minutes on CPU)

Train and test cohorts use disjoint seeds, so the reported accuracy is
held-out. Accuracy here measures whether the pipeline can recover the grade it
was generated from -- it is a pipeline correctness check, not a clinical claim.
Retrain on EyePACS/APTOS/IDRiD before any accuracy is quoted as clinical.
"""
import argparse
import os
import time

import numpy as np

from core import features, grade as grading, lesions, preprocess
from data.synth import cohort


def build_feature_dataset(n_per_grade, seed, verbose=True, cache=True):
    """Extract features for a cohort, caching to disk.

    Feature extraction runs the full segmenter and costs ~0.7s per image, which
    makes any iteration on the grading rules painfully slow. The cohort is
    deterministic in its seed, so the result is safe to cache.
    """
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "models", f"features_n{n_per_grade}_s{seed}.npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        if verbose:
            print(f"  loaded cached features from {os.path.basename(path)}")
        return d["X"], d["y"]

    X, y = [], []
    t0 = time.time()
    for i, (bgr, truth) in enumerate(cohort(n_per_grade=n_per_grade, seed=seed)):
        prep = preprocess.prepare(bgr)
        lmap = lesions.detect(prep)
        vec, _ = features.extract(lmap, prep)
        X.append(vec)
        y.append(truth["grade"])
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1} images ({time.time() - t0:.0f}s)", flush=True)
    X, y = np.array(X, np.float32), np.array(y, np.int64)
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, X=X, y=y)
    return X, y


def confusion(y_true, y_pred, n=5):
    m = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def print_confusion(m):
    print("        predicted")
    print("        " + " ".join(f"{i:4d}" for i in range(m.shape[1])))
    for i, row in enumerate(m):
        print(f"true {i}: " + " ".join(f"{v:4d}" for v in row))


def referable_metrics(y_true, y_pred, threshold=2):
    """Referable DR = grade >= 2. This is the number a screening programme is
    actually judged on: sensitivity to disease that needs an ophthalmologist."""
    t = np.array(y_true) >= threshold
    p = np.array(y_pred) >= threshold
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    tn = int((~t & ~p).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return sens, spec, (tp, fp, tn, fn)


def train_features(args):
    print(f"building training set ({args.train} per grade)...", flush=True)
    Xtr, ytr = build_feature_dataset(args.train, seed=1)
    print(f"building held-out test set ({args.test} per grade)...", flush=True)
    Xte, yte = build_feature_dataset(args.test, seed=77)

    g = grading.FeatureGrader().fit(Xtr, ytr)
    g.set_medians(Xtr)
    g.save()
    print(f"saved -> {grading.FeatureGrader.PATH}")

    pred = np.array([g.predict(x).grade for x in Xte])
    acc = float((pred == yte).mean())
    within1 = float((np.abs(pred - yte) <= 1).mean())
    sens, spec, counts = referable_metrics(yte, pred)
    print(f"\nfeature grader, held-out n={len(yte)}")
    print(f"  exact accuracy      : {acc:.3f}")
    print(f"  within +/-1 grade   : {within1:.3f}")
    print(f"  referable (>=2) sens: {sens:.3f}   spec: {spec:.3f}   {counts}")
    print_confusion(confusion(yte, pred))

    # The rule grader is untrained, so evaluating it on the same held-out set
    # shows what the learned model actually adds over the clinical rules.
    rule_pred = []
    for x in Xte:
        vals = dict(zip(features.FEATURE_NAMES, x))
        rule_pred.append(grading.rule_grade(vals).grade)
    rule_pred = np.array(rule_pred)
    rs, rp, _ = referable_metrics(yte, rule_pred)
    print(f"\nICDR rule grader (no training), same test set")
    print(f"  exact accuracy      : {(rule_pred == yte).mean():.3f}")
    print(f"  within +/-1 grade   : {(np.abs(rule_pred - yte) <= 1).mean():.3f}")
    print(f"  referable (>=2) sens: {rs:.3f}   spec: {rp:.3f}")
    return Xtr, ytr


def train_cnn(args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    print("building CNN tensors...", flush=True)

    def tensors(n, seed):
        xs, ys = [], []
        for bgr, truth in cohort(n_per_grade=n, seed=seed):
            prep = preprocess.prepare(bgr)
            xs.append(grading.CnnGrader.to_tensor(prep.bgr)[0])
            ys.append(truth["grade"])
        return torch.stack(xs), torch.tensor(ys)

    Xtr, ytr = tensors(args.train, 11)
    Xte, yte = tensors(args.test, 88)

    net = grading.CnnGrader.build(pretrained=args.pretrained)
    model = net.model
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb)
        model.eval()
        with torch.no_grad():
            pred = model(Xte).argmax(1)
        acc = float((pred == yte).float().mean())
        print(f"  epoch {ep + 1}/{args.epochs}  loss={tot / len(Xtr):.4f}  "
              f"test_acc={acc:.3f}", flush=True)

    with torch.no_grad():
        pred = model(Xte).argmax(1).numpy()
    sens, spec, counts = referable_metrics(yte.numpy(), pred)

    # Refuse to ship a collapsed model. Trained from scratch on a few hundred
    # images, EfficientNet-B0 reliably converges to predicting one class for
    # everything -- which scores at chance but is far worse than having no CNN
    # at all: fusion would drag every grade toward that class, and Grad-CAM over
    # a degenerate model produces saliency that means nothing while looking
    # entirely convincing. Use --pretrained, or more data.
    distinct = len(set(pred.tolist()))
    accuracy = float((pred == yte.numpy()).mean())
    if distinct < 3 or accuracy < 0.40:
        print(f"\nREFUSED TO SAVE: model predicts only {distinct} distinct "
              f"grade(s) at {accuracy:.1%} accuracy -- this is a collapsed model.")
        print("  Retry with --pretrained (ImageNet init) or a larger --train.")
        print("  The pipeline runs correctly without a CNN; the rule and feature")
        print("  graders carry it, and Grad-CAM is simply omitted from reports.")
        return

    net.save()
    print(f"saved -> {grading.CnnGrader.PATH}")
    print(f"\nCNN grader, held-out n={len(yte)}")
    print(f"  exact accuracy      : {(pred == yte.numpy()).mean():.3f}")
    print(f"  referable (>=2) sens: {sens:.3f}   spec: {spec:.3f}   {counts}")
    print_confusion(confusion(yte.numpy(), pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=40, help="images per grade, train")
    ap.add_argument("--test", type=int, default=15, help="images per grade, test")
    ap.add_argument("--cnn", action="store_true", help="also train EfficientNet-B0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--pretrained", action="store_true",
                    help="start from ImageNet weights (needs network access)")
    args = ap.parse_args()

    train_features(args)
    if args.cnn:
        train_cnn(args)


if __name__ == "__main__":
    main()
