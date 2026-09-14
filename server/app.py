"""Drishti API.

    uvicorn server.app:app --port 8000

The model is loaded once at startup and shared. Loading it per request would
dominate latency; loading it at import time would make the module unimportable
without an artefact, which breaks the test suite and a fresh checkout.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import get_settings
from .db import get_sessionmaker, init_db
from .inference import PIPELINE_VERSION, ScreeningService, load_bundle
from .routers import auth, clinical, review, sync

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def bootstrap_admin():
    """Create the first district admin from the environment, if configured.

    Without this a fresh deployment has no way in, since user creation itself
    requires an admin. Only ever creates when no users exist at all.
    """
    from sqlalchemy import func, select

    from .models import Role, User
    from .security import hash_password

    s = get_settings()
    if not (s.bootstrap_admin_email and s.bootstrap_admin_password):
        return None
    with get_sessionmaker()() as db:
        if (db.scalar(select(func.count()).select_from(User)) or 0) > 0:
            return None
        admin = User(email=s.bootstrap_admin_email.lower().strip(),
                     full_name="District Administrator",
                     password_hash=hash_password(s.bootstrap_admin_password),
                     role=Role.DISTRICT_ADMIN)
        db.add(admin)
        db.commit()
        print(f"bootstrapped district admin: {admin.email}")
        return admin.id


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings().check()
    init_db()
    bootstrap_admin()
    bundle = load_bundle()
    app.state.screening_service = ScreeningService(bundle)
    app.state.model_bundle = bundle
    if not bundle.available:
        print(f"WARNING: no grading model at {s.model_dir}/{s.model_name}. "
              "Capture and quality gating work; grading will return 503. "
              "Train one with: python -m dr.train --datasets aptos")
    try:
        yield
    finally:
        app.state.screening_service.close()


app = FastAPI(title="Drishti DR Screening", version=PIPELINE_VERSION,
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Browsers are told exactly which origins may call this; a wildcard would
    # let any page a clinician has open drive the API with their session.
    allow_origins=[o for o in os.environ.get(
        "DRISHTI_CORS_ORIGINS", "http://localhost:8000").split(",") if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(clinical.router)
app.include_router(review.router)
app.include_router(sync.router)


@app.get("/api/health")
def health():
    s = get_settings()
    bundle = getattr(app.state, "model_bundle", None)
    service = getattr(app.state, "screening_service", None)
    return {
        "status": "ok",
        "node_id": s.node_id,
        "node_role": s.node_role,
        "pipeline_version": PIPELINE_VERSION,
        "model": {
            "available": bool(bundle and bundle.available),
            "version": bundle.version if bundle else "absent",
            "input_size": bundle.size if bundle else None,
            "thresholds": list(bundle.thresholds) if bundle else None,
        },
        "inference": service.batcher.stats if service else {},
    }


@app.get("/")
def index():
    path = WEB_DIR / "index.html"
    if not path.exists():
        return JSONResponse({"service": "drishti", "docs": "/docs"})
    return FileResponse(path)
