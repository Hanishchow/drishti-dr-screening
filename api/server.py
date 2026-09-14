"""FastAPI service.

  uvicorn api.server:app --reload --port 8000

Endpoints:
  GET  /                    the dashboard
  GET  /api/health          loaded models and pipeline status
  POST /api/screen          upload a fundus image -> full report
  POST /api/demo            generate a synthetic case at a chosen grade
  POST /api/simulate        run the district telemedicine simulation
"""
import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from core.pipeline import Screener
from sim.district import Config, compare

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

app = FastAPI(title="Drishti DR Screening", version="0.1.0")
_screener = None


def screener():
    # Lazy: model files may not exist until train.py has been run, and the
    # server should still start and serve the UI in that state.
    global _screener
    if _screener is None:
        _screener = Screener()
    return _screener


def _decode(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image. Send PNG or JPEG.")
    # Very large captures are downscaled: the pipeline normalises to 512px
    # internally anyway, and full-size decoding dominates request latency.
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        s = 1600.0 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


@app.get("/")
def index():
    path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(path):
        return JSONResponse({"error": "web/index.html not found"}, 404)
    return FileResponse(path)


@app.get("/api/health")
def health():
    s = screener()
    return {"status": "ok", "models": s.loaded,
            "note": ("Feature grader untrained - run train.py. The ICDR rule "
                     "grader still works without training."
                     if not s.loaded["feature_gbm"] else "ready")}


@app.post("/api/screen")
async def screen(file: UploadFile = File(...), force: bool = Form(False)):
    img = _decode(await file.read())
    report = screener().run(img, want_images=True, force_grade=force)
    return report.to_dict()


@app.post("/api/demo")
async def demo(grade: int = Form(2), seed: int = Form(0), blur: float = Form(0.0),
               vignette: float = Form(0.0), glare: float = Form(0.0),
               exposure: float = Form(1.0), force: bool = Form(False)):
    """Generate a synthetic case so the UI is demonstrable without a dataset."""
    from data.synth import generate
    if not 0 <= grade <= 4:
        raise HTTPException(400, "grade must be 0-4")
    deg = {}
    if blur > 0:
        deg["blur"] = blur
    if vignette > 0:
        deg["vignette"] = vignette
    if glare > 0:
        deg["glare"] = glare
    if exposure != 1.0:
        deg["exposure"] = exposure
    img, truth = generate(grade, seed=seed, **deg)
    report = screener().run(img, want_images=True, force_grade=force)
    out = report.to_dict()
    # Ground truth is returned alongside so the UI can show measured-vs-actual.
    out["ground_truth"] = {"grade": truth["grade"], "counts": truth["counts"]}
    return out


@app.post("/api/simulate")
async def simulate_endpoint(phcs: int = Form(25), ophthalmologists: int = Form(2),
                            days: int = Form(180),
                            screenings_per_phc_per_day: int = Form(12),
                            ai_sensitivity_referable: float = Form(0.92),
                            ai_specificity_referable: float = Form(0.88)):
    cfg = Config(phcs=phcs, ophthalmologists=ophthalmologists, days=days,
                 screenings_per_phc_per_day=screenings_per_phc_per_day,
                 ai_sensitivity_referable=ai_sensitivity_referable,
                 ai_specificity_referable=ai_specificity_referable)
    return compare(cfg)
