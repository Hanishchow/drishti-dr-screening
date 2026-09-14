"""Model registry and batched GPU inference.

Two things this solves that a naive `model(image)` in the request handler does
not:

1. GPU BATCHING. Screening traffic arrives as many small independent requests,
   one per capture. Running them one at a time leaves the GPU almost idle while
   each request pays full kernel-launch latency. Requests are coalesced into a
   batch for a few milliseconds, which raises throughput several-fold at a cost
   of latency far below what a clinician can perceive.

2. NOT BLOCKING THE EVENT LOOP. Inference and the morphological segmenter are
   CPU/GPU-bound synchronous work. Run inline in an async handler they stall
   every other request on the process, so one slow capture freezes the whole
   PHC. All heavy work happens on a worker thread.

The registry deliberately loads an ONNX artefact rather than a torch
checkpoint: the served graph is then frozen and versioned, and the API process
never has to import the training code.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import get_settings

PIPELINE_VERSION = "2.0"


@dataclass
class ModelBundle:
    """A grader plus everything needed to reproduce its decisions."""
    grader: object | None
    version: str
    thresholds: list
    size: int
    graham: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.grader is not None


def load_bundle(model_dir=None, name=None, meta_name=None) -> ModelBundle:
    s = get_settings()
    d = Path(model_dir or s.model_dir)
    onnx_path = d / (name or s.model_name)
    meta_path = d / (meta_name or s.model_meta)

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}

    thresholds = meta.get("thresholds") or [0.5, 1.5, 2.5, 3.5]
    size = int(meta.get("size") or s.image_size)

    if not onnx_path.exists():
        # No trained artefact yet. This is a normal state on a fresh checkout,
        # so it must degrade to a clear "model unavailable" rather than
        # crashing the server at import time.
        return ModelBundle(None, "absent", thresholds, size, True, meta)

    from dr.model import OnnxGrader
    providers = None
    if s.device == "cpu":
        providers = ["CPUExecutionProvider"]
    grader = OnnxGrader(onnx_path, providers=providers)

    stat = onnx_path.stat()
    version = meta.get("version") or (
        f"{meta.get('backbone', 'unknown')}@{size}"
        f"-{int(stat.st_mtime)}-{stat.st_size}")
    return ModelBundle(grader, version, thresholds, size,
                       bool(meta.get("graham", True)), meta)


# --------------------------------------------------------------- batching
@dataclass
class _Job:
    tensor: np.ndarray
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop


class BatchedGrader:
    """Coalesces concurrent single-image requests into GPU batches.

    A dedicated thread owns the model. Callers `await submit(...)`; the thread
    drains up to `batch_size` jobs that arrived within `batch_wait_ms` and runs
    them as one forward pass.
    """

    def __init__(self, bundle: ModelBundle, batch_size=None, wait_ms=None):
        s = get_settings()
        self.bundle = bundle
        self.batch_size = batch_size or s.batch_size
        self.wait_s = (wait_ms if wait_ms is not None else s.batch_wait_ms) / 1000.0
        self._queue: list[_Job] = []
        self._lock = threading.Condition()
        self._stop = False
        self._thread = None
        self.stats = {"batches": 0, "images": 0, "max_batch": 0, "errors": 0}

    def start(self):
        if self._thread is None and self.bundle.available:
            self._stop = False
            self._thread = threading.Thread(target=self._run, name="grader",
                                            daemon=True)
            self._thread.start()
        return self

    def stop(self):
        with self._lock:
            self._stop = True
            self._lock.notify_all()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    async def submit(self, tensor: np.ndarray):
        if not self.bundle.available:
            raise RuntimeError(
                "No grading model is loaded. Train one with dr/train.py and "
                f"place grader.onnx in '{get_settings().model_dir}'.")
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        with self._lock:
            self._queue.append(_Job(tensor, fut, loop))
            self._lock.notify()
        return await fut

    def _collect(self):
        with self._lock:
            while not self._queue and not self._stop:
                self._lock.wait(timeout=0.25)
            if self._stop and not self._queue:
                return []
            # Wait briefly for stragglers so a burst forms one batch instead of
            # many single-image passes.
            deadline = time.monotonic() + self.wait_s
            while len(self._queue) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._lock.wait(timeout=remaining)
            batch, self._queue = self._queue[:self.batch_size], self._queue[self.batch_size:]
            return batch

    def _run(self):
        while not self._stop:
            batch = self._collect()
            if not batch:
                continue
            try:
                stacked = np.stack([j.tensor for j in batch]).astype(np.float32)
                grades, dists = self.bundle.grader(stacked)
                self.stats["batches"] += 1
                self.stats["images"] += len(batch)
                self.stats["max_batch"] = max(self.stats["max_batch"], len(batch))
                for k, job in enumerate(batch):
                    self._resolve(job, (float(grades[k]), dists[k].tolist()))
            except Exception as e:                      # noqa: BLE001
                self.stats["errors"] += 1
                for job in batch:
                    self._reject(job, e)

    @staticmethod
    def _resolve(job: _Job, value):
        if not job.future.done():
            job.loop.call_soon_threadsafe(
                lambda: job.future.done() or job.future.set_result(value))

    @staticmethod
    def _reject(job: _Job, exc):
        if not job.future.done():
            job.loop.call_soon_threadsafe(
                lambda: job.future.done() or job.future.set_exception(exc))


# ----------------------------------------------------------- full pipeline
class ScreeningService:
    """Quality gate -> lesions -> CNN grade -> explanation -> triage.

    The quality gate runs FIRST and short-circuits, so an unusable capture
    costs milliseconds instead of a GPU slot, and the ASHA worker is told to
    recapture while the patient is still present.
    """

    def __init__(self, bundle: ModelBundle | None = None):
        self.bundle = bundle or load_bundle()
        self.batcher = BatchedGrader(self.bundle).start()

    @property
    def model_version(self):
        return self.bundle.version

    def close(self):
        self.batcher.stop()

    # -- synchronous stages, run on a worker thread ------------------------
    @staticmethod
    def _quality(bgr):
        from core import quality
        return quality.assess(bgr)

    def _prepare_tensor(self, bgr):
        from dr import transforms as T
        import cv2
        img = T.crop_to_fov(bgr)
        img = cv2.resize(img, (self.bundle.size, self.bundle.size),
                         interpolation=cv2.INTER_AREA)
        if self.bundle.graham:
            img = T.graham_normalise(img, self.bundle.size)
        else:
            img = cv2.bitwise_and(img, img, mask=T.circular_mask(self.bundle.size))
        return T.to_tensor(img)

    @staticmethod
    def _lesions(bgr):
        from core import features, lesions, preprocess
        prep = preprocess.prepare(bgr)
        lmap = lesions.detect(prep)
        _, values = features.extract(lmap, prep)
        return prep, lmap, values

    async def run(self, bgr, want_images=False):
        from core import explain, grade as grading, triage as triage_mod
        from dr import metrics as dr_metrics

        t0 = time.perf_counter()
        timing = {}
        loop = asyncio.get_running_loop()

        q = await loop.run_in_executor(None, self._quality, bgr)
        timing["quality"] = (time.perf_counter() - t0) * 1000
        if not q.passed:
            return {"graded": False, "quality": q.to_dict(),
                    "model_version": self.model_version,
                    "pipeline_version": PIPELINE_VERSION,
                    "timing_ms": {k: round(v, 1) for k, v in timing.items()}}

        t1 = time.perf_counter()
        prep, lmap, values = await loop.run_in_executor(None, self._lesions, bgr)
        timing["segmentation"] = (time.perf_counter() - t1) * 1000

        t1 = time.perf_counter()
        tensor = await loop.run_in_executor(None, self._prepare_tensor, bgr)
        score, distribution = await self.batcher.submit(tensor)
        cnn_grade = int(dr_metrics.apply_thresholds([score], self.bundle.thresholds)[0])
        timing["grading"] = (time.perf_counter() - t1) * 1000

        # Referable probability comes straight from the ordinal head: the sum
        # of P(grade=k) for k >= 2. A softmax model could not offer this.
        referable_p = float(sum(distribution[get_settings().referable_grade:]))

        cnn_out = grading.GraderOutput(
            grade=cnn_grade, probabilities=list(distribution),
            source="cnn_ordinal",
            detail={"expected_grade": round(float(score), 3),
                    "referable_probability": round(referable_p, 4)})
        rule_out = grading.rule_grade(values)
        fused = grading.fuse([rule_out, cnn_out])

        t1 = time.perf_counter()
        agreement = explain.attention_agreement(None, lmap, prep)
        narrative = explain.narrative(values, rule_out, fused, agreement, q)
        decision = triage_mod.decide(fused.grade, values, fused, q, agreement)
        timing["explain"] = (time.perf_counter() - t1) * 1000
        timing["total"] = (time.perf_counter() - t0) * 1000

        result = {
            "graded": True,
            "quality": q.to_dict(),
            "grade": int(fused.grade),
            "grade_name": grading.GRADE_NAMES[int(fused.grade)],
            "confidence": round(fused.confidence, 3),
            "grade_distribution": [round(float(p), 4) for p in fused.probabilities],
            "referable_probability": round(referable_p, 4),
            "graders": [o.to_dict() for o in (rule_out, cnn_out)] + [fused.to_dict()],
            "lesion_counts": lmap.counts(),
            "explanation": narrative,
            "attention": agreement,
            "triage": decision.to_dict(),
            "model_version": self.model_version,
            "model_thresholds": list(self.bundle.thresholds),
            "pipeline_version": PIPELINE_VERSION,
            "timing_ms": {k: round(v, 1) for k, v in timing.items()},
        }
        if want_images:
            from core.pipeline import _png_b64
            result["images"] = {
                "input": _png_b64(prep.bgr),
                "overlay": _png_b64(explain.render_overlay(prep, lmap)),
            }
        return result
