"""Train the DR grader on real data.

Typical Kaggle run (P100/T4, ~9h for the full thing):

    python -m dr.train --datasets aptos idrid --size 512 \
        --backbone tf_efficientnet_b3_ns --epochs 15 --batch-size 12

    python -m dr.train --datasets eyepacs aptos idrid --size 640 \
        --backbone tf_efficientnet_b4_ns --epochs 8 --batch-size 8 \
        --cache-dir /kaggle/working/cache

Design points that matter:

* CHECKPOINT EVERY EPOCH. Kaggle kills sessions at 12h and on idle; a run that
  cannot resume is a run that never finishes on EyePACS.
* Thresholds are fitted on VALIDATION predictions, never on training ones, and
  are saved with the weights. A checkpoint without its cut-points is unusable,
  because the raw expected-grade scores are calibrated to nothing.
* Selection is on QWK, not loss. Loss improves while the model gets better at
  the majority grade; QWK is what the task is actually scored on.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import datasets as D, metrics as M, model as MD, splits as S
from .torchdata import build_loaders, build_cache


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the DR grader on real data")
    p.add_argument("--datasets", nargs="+", default=["aptos"],
                   choices=sorted(D.LOADERS), help="corpora to train on")
    p.add_argument("--external", nargs="+", default=[],
                   choices=sorted(D.LOADERS),
                   help="corpora held out entirely for external validation")
    p.add_argument("--data-root", default=None)
    p.add_argument("--backbone", default="tf_efficientnet_b3_ns")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--build-cache", action="store_true",
                   help="materialise the resized cache, then exit")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--resume", default=None)
    p.add_argument("--balance-power", type=float, default=0.5)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="debug: cap dataset size")
    return p.parse_args(argv)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, tta=True):
    """Validation pass. Horizontal+vertical flip TTA is cheap and reliably
    worth ~0.005-0.01 QWK on this task."""
    model.eval()
    scores, labels = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logit_sum = model(x).float()
            n = 1
            if tta:
                for dims in ([3], [2], [2, 3]):
                    logit_sum = logit_sum + model(torch.flip(x, dims=dims)).float()
                    n += 1
            logits = logit_sum / n
        scores.append(M.coral_expected_grade(logits.cpu().numpy()))
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    print("loading datasets:")
    records = D.load(args.datasets, args.data_root)
    if not records:
        raise SystemExit(
            "No images loaded. Check --data-root / DR_DATA_ROOT, or see the "
            "download instructions printed above.")
    if args.limit:
        records = records[:args.limit]
    print(D.describe(records))

    if args.build_cache:
        n = build_cache(records, args.size, True, args.cache_dir, workers=args.workers)
        print(f"cached {n} images to {args.cache_dir}")
        return

    train_idx, val_idx = S.train_val_split(records, args.val_fraction, args.seed)
    S.assert_no_patient_leakage(records, train_idx, val_idx)
    print("split (patient-grouped, grade-stratified):")
    print(S.summarise_split(records, train_idx, val_idx))

    train_dl, val_dl = build_loaders(
        records, train_idx, val_idx, args.size, args.batch_size, args.workers,
        balanced=True, cache_dir=args.cache_dir, seed=args.seed,
        balance_power=args.balance_power)

    model = MD.build(args.backbone, pretrained=not args.no_pretrained).to(device)
    criterion = MD.CoralLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = max(1, len(train_dl) // args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch, best_qwk = 0, -1.0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_qwk = ck.get("best_qwk", -1.0)
        print(f"resumed from {args.resume} at epoch {start_epoch} (best QWK {best_qwk:.4f})")

    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)

        for step, (x, y, _) in enumerate(train_dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y) / args.accum
            scaler.scale(loss).backward()

            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()

            running += float(loss.detach()) * args.accum * x.size(0)
            seen += x.size(0)
            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(train_dl)} "
                      f"loss {running / max(seen, 1):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

        val_scores, val_labels = evaluate(model, val_dl, device)
        # Cut-points are fitted on validation scores only. Fitting them on
        # training scores would leak and would also be calibrated to a
        # distribution the model has already overfitted.
        thresholds, fitted_qwk = M.optimise_thresholds(val_labels, val_scores)
        preds = M.apply_thresholds(val_scores, thresholds)
        stats = M.summarise(val_labels, preds)

        print(f"\nepoch {epoch + 1}/{args.epochs}  "
              f"train_loss {running / max(seen, 1):.4f}  "
              f"{time.time() - t0:.0f}s")
        print(M.format_report(stats, "validation"))
        print(f"  thresholds: {np.round(thresholds, 3).tolist()}\n", flush=True)

        history.append({"epoch": epoch + 1, "train_loss": running / max(seen, 1),
                        **{k: v for k, v in stats.items() if k != "confusion"}})

        ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best_qwk": max(best_qwk, stats["qwk"]),
                "thresholds": thresholds.tolist(), "args": vars(args),
                "backbone": args.backbone, "size": args.size}
        torch.save(ckpt, out / "last.pt")

        # Select on QWK, not on loss: loss keeps improving on the majority
        # grade long after the clinically useful ranking has stopped improving.
        if stats["qwk"] > best_qwk:
            best_qwk = stats["qwk"]
            torch.save(ckpt, out / "best.pt")
            (out / "metrics.json").write_text(json.dumps(
                {"val": stats, "thresholds": thresholds.tolist(),
                 "epoch": epoch + 1, "args": vars(args)}, indent=2))
            print(f"  new best QWK {best_qwk:.4f} -> {out / 'best.pt'}\n")

        (out / "history.json").write_text(json.dumps(history, indent=2))

    # ------------------------------------------------- external validation
    if args.external:
        print("\n=== external validation (never trained on) ===")
        ck = torch.load(out / "best.pt", map_location=device)
        model.load_state_dict(ck["model"])
        th = np.array(ck["thresholds"])
        ext = D.load(args.external, args.data_root)
        if ext:
            print(D.describe(ext))
            _, ext_dl = build_loaders(ext, [], list(range(len(ext))), args.size,
                                      args.batch_size, args.workers,
                                      balanced=False, cache_dir=args.cache_dir)
            scores, labels = evaluate(model, ext_dl, device)
            ext_stats = M.summarise(labels, M.apply_thresholds(scores, th))
            print(M.format_report(ext_stats, "external"))
            (out / "external_metrics.json").write_text(json.dumps(ext_stats, indent=2))

    # Export for serving. The ONNX artefact is what the API loads, so the
    # server never has to import the training code.
    try:
        ck = torch.load(out / "best.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        MD.export_onnx(model, out / "grader.onnx", size=args.size)
        (out / "grader.json").write_text(json.dumps({
            "backbone": args.backbone, "size": args.size,
            "thresholds": ck["thresholds"], "graham": True,
            "val_qwk": ck.get("best_qwk"), "datasets": args.datasets}, indent=2))
        print(f"exported {out / 'grader.onnx'}")
    except Exception as e:
        print(f"ONNX export skipped: {e}")

    print(f"\nbest validation QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
