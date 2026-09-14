"""Ophthalmologist review queue, sign-off, and programme statistics.

This is the module that turns a grade into a workflow. The triage layer marks
cases as needing a human; without a queue a specialist actually logs into, that
flag is only a field in a database.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..audit import record
from ..db import get_db
from ..models import (Patient, Review, Screening, ScreeningStatus,
                      URGENCY_RANK, Urgency, User)
from ..schemas import QueueItem, ReviewCreate, ReviewOut, ScreeningOut
from ..security import require

router = APIRouter(prefix="/api/review", tags=["review"])

# Clinical windows, in days, by urgency. A case is "breaching" once it has
# waited longer than its window.
WINDOW_DAYS = {Urgency.EMERGENCY: 7, Urgency.URGENT: 28,
               Urgency.SOON: 180, Urgency.ROUTINE: 365}


def _waiting_days(sc: Screening) -> int:
    captured = sc.captured_at
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - captured).days)


@router.get("/queue", response_model=list[QueueItem])
def queue(limit: int = Query(50, le=200), include_breaching_first: bool = True,
          db: Session = Depends(get_db),
          _: User = Depends(require("review:read_queue"))):
    """Cases awaiting a human, most clinically urgent first.

    Ordering is by urgency and then by how long the patient has already waited.
    Strict FIFO would be wrong here: a proliferative case captured this morning
    must outrank a routine one from last month.
    """
    stmt = (select(Screening, Patient)
            .join(Patient, Screening.patient_id == Patient.id)
            .where(Screening.status == ScreeningStatus.NEEDS_REVIEW)
            .limit(limit * 4))
    rows = list(db.execute(stmt))

    def sort_key(row):
        sc = row[0]
        rank = URGENCY_RANK.get(sc.urgency, 3)
        return (rank, -_waiting_days(sc))

    items = []
    for sc, patient in sorted(rows, key=sort_key)[:limit]:
        waited = _waiting_days(sc)
        window = WINDOW_DAYS.get(sc.urgency, 365)
        out = ScreeningOut.model_validate(sc)
        out.final_grade = sc.final_grade
        items.append(QueueItem(screening=out, patient_name=patient.name,
                               patient_mrn=patient.mrn, waiting_days=waited,
                               breaching=waited > window))
    if include_breaching_first:
        items.sort(key=lambda i: (not i.breaching,
                                  URGENCY_RANK.get(i.screening.urgency, 3),
                                  -i.waiting_days))
    return items


@router.post("/{screening_id}", response_model=ReviewOut, status_code=201)
def sign_off(screening_id: str, payload: ReviewCreate,
             db: Session = Depends(get_db),
             reviewer: User = Depends(require("review:sign_off"))):
    """Record an ophthalmologist's decision.

    The machine's grade is never overwritten. A correction is appended as a
    Review that supersedes it, so the disagreement itself stays visible -- that
    record is what lets the programme measure where the model is weak.
    """
    sc = db.get(Screening, screening_id)
    if sc is None:
        raise HTTPException(404, "screening not found")
    if sc.status == ScreeningStatus.PENDING:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "screening has not been graded yet")

    corrected = payload.corrected_grade
    agrees = payload.agrees_with_model
    if corrected is not None and sc.grade is not None:
        # Trust the grades, not the checkbox: a reviewer who types a different
        # grade has disagreed regardless of what the flag says.
        agrees = (corrected == sc.grade)

    review = Review(screening_id=sc.id, reviewer_id=reviewer.id,
                    corrected_grade=corrected, agrees_with_model=agrees,
                    urgency_override=payload.urgency_override,
                    notes=payload.notes)
    db.add(review)

    sc.status = ScreeningStatus.REVIEWED
    sc.needs_human_review = False
    if payload.urgency_override:
        sc.urgency = payload.urgency_override

    record(db, reviewer, "screening_reviewed", "screening", sc.id,
           {"model_grade": sc.grade, "corrected_grade": corrected,
            "agrees": agrees, "urgency": sc.urgency.value if sc.urgency else None})
    db.commit()
    return review


@router.get("/{screening_id}/reviews", response_model=list[ReviewOut])
def screening_reviews(screening_id: str, db: Session = Depends(get_db),
                      _: User = Depends(require("review:read_queue"))):
    sc = db.get(Screening, screening_id)
    if sc is None:
        raise HTTPException(404, "screening not found")
    return sc.reviews


@router.get("/stats/summary")
def stats(db: Session = Depends(get_db), _: User = Depends(require("stats:read"))):
    """Programme-level numbers a district officer actually asks for."""
    total = db.scalar(select(func.count()).select_from(Screening)) or 0
    by_status = {s.value: (db.scalar(
        select(func.count()).select_from(Screening)
        .where(Screening.status == s)) or 0) for s in ScreeningStatus}
    by_urgency = {u.value: (db.scalar(
        select(func.count()).select_from(Screening)
        .where(Screening.urgency == u)) or 0) for u in Urgency}

    graded = db.scalar(select(func.count()).select_from(Screening)
                       .where(Screening.grade.isnot(None))) or 0
    referable = db.scalar(select(func.count()).select_from(Screening)
                          .where(Screening.grade >= 2)) or 0
    quality_failed = by_status.get(ScreeningStatus.QUALITY_FAILED.value, 0)

    # Reviewer agreement is the most useful single number for model trust: it
    # is measured against real ophthalmologists on real local patients, not
    # against a held-out split of a public dataset.
    reviews = list(db.scalars(select(Review)))
    corrections = [r for r in reviews if r.corrected_grade is not None]
    agree = sum(1 for r in corrections if r.agrees_with_model)

    pending = list(db.scalars(select(Screening).where(
        Screening.status == ScreeningStatus.NEEDS_REVIEW)))
    breaching = sum(1 for sc in pending
                    if _waiting_days(sc) > WINDOW_DAYS.get(sc.urgency, 365))

    return {
        "total_screenings": total,
        "graded": graded,
        "quality_failure_rate": round(quality_failed / total, 4) if total else 0.0,
        "referable_rate": round(referable / graded, 4) if graded else 0.0,
        "by_status": by_status,
        "by_urgency": by_urgency,
        "queue_depth": len(pending),
        "queue_breaching_window": breaching,
        "reviews_recorded": len(reviews),
        "reviewer_agreement": (round(agree / len(corrections), 4)
                               if corrections else None),
        "reviewer_corrections": len(corrections) - agree,
    }
