"""Patients and screenings: capture, grade, history."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import (APIRouter, Depends, File, Form, HTTPException, Request,
                     UploadFile, status)
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..config import get_settings
from ..db import get_db
from ..models import (Patient, Role, Screening, ScreeningStatus, Urgency, User)
from ..schemas import (PatientCreate, PatientHistory, PatientOut,
                       ProgressionPoint, ScreeningOut)
from ..security import can, current_user, require

router = APIRouter(prefix="/api", tags=["clinical"])


# ------------------------------------------------------------------ helpers
def _decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "could not decode image; send PNG or JPEG")
    return img


def _store_image(data: bytes, screening_id: str) -> tuple[str, str]:
    s = get_settings()
    digest = hashlib.sha256(data).hexdigest()
    # Shard by digest prefix: a district accumulates hundreds of thousands of
    # files and a single flat directory becomes unusable on most filesystems.
    folder = Path(s.storage_dir) / digest[:2] / digest[2:4]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{screening_id}.png"
    path.write_bytes(data)
    return str(path), digest


def _visible_patient(db: Session, user: User, patient_id: str) -> Patient:
    p = db.get(Patient, patient_id)
    if p is None:
        raise HTTPException(404, "patient not found")
    # An ASHA worker sees only their own facility. Enforced here rather than in
    # the query so the 404/403 distinction stays explicit.
    if user.role == Role.ASHA and p.facility_id and p.facility_id != user.facility_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "patient belongs to another facility")
    return p


def _apply_result(sc: Screening, result: dict):
    """Copy a pipeline result onto a screening row.

    The machine's output is written exactly once. A later human correction
    lands in the Review table instead, so both survive.
    """
    q = result.get("quality") or {}
    sc.quality_passed = q.get("passed")
    sc.quality_score = q.get("score")
    sc.quality_failures = q.get("failures")
    sc.quality_guidance = q.get("guidance")
    sc.model_version = result.get("model_version")
    sc.model_thresholds = result.get("model_thresholds")
    sc.pipeline_version = result.get("pipeline_version")
    sc.latency_ms = result.get("timing_ms")

    if not result.get("graded"):
        sc.status = ScreeningStatus.QUALITY_FAILED
        return sc

    triage = result.get("triage") or {}
    sc.grade = result.get("grade")
    sc.confidence = result.get("confidence")
    sc.grade_distribution = result.get("grade_distribution")
    sc.referable_probability = result.get("referable_probability")
    sc.lesion_counts = result.get("lesion_counts")
    sc.explanation = result.get("explanation")
    sc.attention = result.get("attention")
    sc.urgency = Urgency(triage["urgency"]) if triage.get("urgency") else None
    sc.days_to_review = triage.get("days_to_review")
    sc.needs_human_review = bool(triage.get("needs_human_review"))
    sc.triage_reasons = triage.get("reasons")
    sc.status = (ScreeningStatus.NEEDS_REVIEW if sc.needs_human_review
                 else ScreeningStatus.GRADED)
    return sc


# ----------------------------------------------------------------- patients
@router.post("/patients", response_model=PatientOut, status_code=201)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db),
                   user: User = Depends(require("patient:create"))):
    existing = db.scalar(select(Patient).where(Patient.mrn == payload.mrn))
    if existing:
        # Returning the existing record rather than erroring: a returning
        # patient is the normal case in an annual screening programme, and the
        # whole value of the record is the longitudinal series.
        return existing
    p = Patient(**payload.model_dump(exclude_none=False))
    if p.facility_id is None:
        p.facility_id = user.facility_id
    db.add(p)
    db.flush()
    record(db, user, "patient_created", "patient", p.id, {"mrn": p.mrn})
    db.commit()
    return p


@router.get("/patients", response_model=list[PatientOut])
def list_patients(q: str | None = None, limit: int = 50,
                  db: Session = Depends(get_db),
                  user: User = Depends(require("patient:read"))):
    stmt = select(Patient)
    if user.role == Role.ASHA and user.facility_id:
        stmt = stmt.where(Patient.facility_id == user.facility_id)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(Patient.name.ilike(like) | Patient.mrn.ilike(like))
    return list(db.scalars(stmt.order_by(Patient.created_at.desc())
                           .limit(min(limit, 200))))


@router.get("/patients/{patient_id}/history", response_model=PatientHistory)
def patient_history(patient_id: str, db: Session = Depends(get_db),
                    user: User = Depends(require("patient:read"))):
    """Longitudinal view.

    Progression between visits is the signal an annual programme exists to
    catch -- a stable grade-2 and a grade-2 that was grade-0 last year are very
    different patients.
    """
    p = _visible_patient(db, user, patient_id)
    graded = [s for s in p.screenings if s.final_grade is not None]
    timeline = [ProgressionPoint(
        screening_id=s.id, captured_at=s.captured_at, eye=s.eye,
        grade=s.grade, final_grade=s.final_grade, urgency=s.urgency)
        for s in p.screenings]

    worst = max((s.final_grade for s in graded), default=None)
    progressed, note = False, None
    if len(graded) >= 2:
        first, last = graded[0], graded[-1]
        delta = last.final_grade - first.final_grade
        if delta > 0:
            progressed = True
            days = (last.captured_at - first.captured_at).days
            note = (f"Grade rose from {first.final_grade} to {last.final_grade} "
                    f"over {days} days across {len(graded)} screenings.")
        elif delta < 0:
            note = (f"Grade fell from {first.final_grade} to {last.final_grade}; "
                    "confirm this is treatment response and not a grading error.")
        else:
            note = f"Stable at grade {last.final_grade} across {len(graded)} screenings."

    return PatientHistory(patient=PatientOut.model_validate(p), timeline=timeline,
                          worst_grade=worst, progressed=progressed,
                          progression_note=note)


# --------------------------------------------------------------- screenings
@router.post("/screenings", response_model=ScreeningOut, status_code=201)
async def create_screening(request: Request,
                           patient_id: str = Form(...),
                           eye: str = Form(...),
                           client_uuid: str | None = Form(None),
                           file: UploadFile = File(...),
                           db: Session = Depends(get_db),
                           user: User = Depends(require("screening:create"))):
    s = get_settings()
    if eye not in ("L", "R"):
        raise HTTPException(400, "eye must be 'L' or 'R'")

    data = await file.read()
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"image exceeds {s.max_upload_mb} MB")
    patient = _visible_patient(db, user, patient_id)

    # Idempotency: a flaky rural link means the client may retry a capture it
    # already delivered. Same client_uuid must return the same screening rather
    # than creating a duplicate clinical record.
    if client_uuid:
        existing = db.scalar(select(Screening).where(
            Screening.client_uuid == client_uuid))
        if existing:
            return _out(existing)

    img = _decode_image(data)
    sc = Screening(patient_id=patient.id, eye=eye,
                   facility_id=patient.facility_id or user.facility_id,
                   captured_by_id=user.id, image_path="",
                   status=ScreeningStatus.PENDING)
    if client_uuid:
        sc.client_uuid = client_uuid
    db.add(sc)
    db.flush()

    path, digest = _store_image(data, sc.id)
    sc.image_path, sc.image_sha256 = path, digest

    service = request.app.state.screening_service
    try:
        result = await service.run(img)
        _apply_result(sc, result)
    except Exception as e:                       # noqa: BLE001
        # A failed capture must remain in the database as FAILED rather than
        # vanishing: an image that never got graded is a patient who never got
        # an answer, and that has to be visible.
        sc.status = ScreeningStatus.FAILED
        sc.error = str(e)[:2000]
        record(db, user, "screening_failed", "screening", sc.id, {"error": str(e)[:500]})
        db.commit()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"grading failed: {e}")

    record(db, user, "screening_graded", "screening", sc.id,
           {"grade": sc.grade, "urgency": sc.urgency.value if sc.urgency else None,
            "model_version": sc.model_version,
            "needs_review": sc.needs_human_review})
    db.commit()
    return _out(sc)


def _out(sc: Screening) -> ScreeningOut:
    data = ScreeningOut.model_validate(sc)
    data.final_grade = sc.final_grade
    return data


@router.get("/screenings/{screening_id}", response_model=ScreeningOut)
def get_screening(screening_id: str, db: Session = Depends(get_db),
                  user: User = Depends(require("screening:read"))):
    sc = db.get(Screening, screening_id)
    if sc is None:
        raise HTTPException(404, "screening not found")
    if (user.role == Role.ASHA and sc.facility_id
            and sc.facility_id != user.facility_id):
        raise HTTPException(403, "screening belongs to another facility")
    return _out(sc)


@router.get("/patients/{patient_id}/screenings", response_model=list[ScreeningOut])
def patient_screenings(patient_id: str, db: Session = Depends(get_db),
                       user: User = Depends(require("screening:read"))):
    p = _visible_patient(db, user, patient_id)
    return [_out(s) for s in p.screenings]
