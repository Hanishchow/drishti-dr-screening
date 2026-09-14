"""Database schema.

Design constraints that shaped this:

* A screening is CLINICAL EVIDENCE. Once a grade is issued it is never mutated;
  an ophthalmologist's disagreement is recorded as a separate Review that
  supersedes it, so the machine's original output and the human's correction
  both survive. Overwriting would destroy the audit trail that makes the system
  defensible.

* Every automated decision records the model version and the exact thresholds
  used. Without those a past grade cannot be reproduced, which makes it
  impossible to answer "why did it say that in March?" after a model update.

* Screenings carry a CLIENT-GENERATED uuid. A PHC laptop that is offline must
  be able to create records that will not collide when it finally syncs, and
  re-syncing the same batch must be idempotent rather than duplicating patients.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (Boolean, DateTime, Enum, Float, ForeignKey, Index,
                        Integer, JSON, String, Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow():
    return datetime.now(timezone.utc)


def new_uuid():
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------- enums
class Role(str, enum.Enum):
    ASHA = "asha"                      # captures images at a PHC
    OPHTHALMOLOGIST = "ophthalmologist"  # reads the review queue, signs off
    DISTRICT_ADMIN = "district_admin"  # programme oversight, user management
    SERVICE = "service"                # edge node syncing, machine-to-machine


class ScreeningStatus(str, enum.Enum):
    PENDING = "pending"                # uploaded, not yet graded
    QUALITY_FAILED = "quality_failed"  # rejected at capture; recapture needed
    GRADED = "graded"                  # machine grade issued, no review needed
    NEEDS_REVIEW = "needs_review"      # routed to a human
    REVIEWED = "reviewed"              # a human has signed it off
    FAILED = "failed"                  # inference error


class Urgency(str, enum.Enum):
    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"
    EMERGENCY = "emergency"


URGENCY_RANK = {Urgency.EMERGENCY: 0, Urgency.URGENT: 1,
                Urgency.SOON: 2, Urgency.ROUTINE: 3}


# ------------------------------------------------------------------ tables
class Facility(Base):
    __tablename__ = "facilities"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(200))
    district: Mapped[str] = mapped_column(String(120), index=True)
    kind: Mapped[str] = mapped_column(String(40), default="phc")  # phc | hospital
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    users: Mapped[list["User"]] = relationship(back_populates="facility")


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(200))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.ASHA, index=True)
    facility_id: Mapped[str | None] = mapped_column(ForeignKey("facilities.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    facility: Mapped[Facility | None] = relationship(back_populates="users")


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    # Programme identifier (ABHA number, or a district-local roll). Unique so a
    # returning patient is matched rather than duplicated -- longitudinal
    # comparison is the whole point of keeping records.
    mrn: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    year_of_birth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(16), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    facility_id: Mapped[str | None] = mapped_column(ForeignKey("facilities.id"), nullable=True)
    diabetes_since_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    screenings: Mapped[list["Screening"]] = relationship(
        back_populates="patient", order_by="Screening.captured_at")


class Screening(Base):
    """One eye, one capture, one machine decision."""
    __tablename__ = "screenings"
    __table_args__ = (
        Index("ix_screening_queue", "status", "urgency", "captured_at"),
        UniqueConstraint("client_uuid", name="uq_screening_client_uuid"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    # Minted by the capturing client so an offline PHC can create records that
    # never collide, and so a repeated sync is idempotent.
    client_uuid: Mapped[str] = mapped_column(String(36), index=True, default=new_uuid)

    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    facility_id: Mapped[str | None] = mapped_column(ForeignKey("facilities.id"), nullable=True)
    captured_by_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    eye: Mapped[str] = mapped_column(String(2))                # "L" | "R"
    image_path: Mapped[str] = mapped_column(String(500))
    image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  default=utcnow, index=True)

    status: Mapped[ScreeningStatus] = mapped_column(
        Enum(ScreeningStatus), default=ScreeningStatus.PENDING, index=True)

    # --- quality gate ---
    quality_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_failures: Mapped[list | None] = mapped_column(JSON, nullable=True)
    quality_guidance: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- machine grade (immutable once written) ---
    grade: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    grade_distribution: Mapped[list | None] = mapped_column(JSON, nullable=True)
    referable_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    lesion_counts: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    explanation: Mapped[list | None] = mapped_column(JSON, nullable=True)
    attention: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # --- triage ---
    urgency: Mapped[Urgency | None] = mapped_column(Enum(Urgency), nullable=True, index=True)
    days_to_review: Mapped[int | None] = mapped_column(Integer, nullable=True)
    needs_human_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    triage_reasons: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # --- reproducibility ---
    # Without these a past decision cannot be reconstructed after a model
    # update, which makes the audit trail worthless.
    model_version: Mapped[str | None] = mapped_column(String(120), nullable=True)
    model_thresholds: Mapped[list | None] = mapped_column(JSON, nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    latency_ms: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="screenings")
    reviews: Mapped[list["Review"]] = relationship(
        back_populates="screening", order_by="Review.created_at")

    @property
    def final_grade(self) -> int | None:
        """The grade that counts: the latest human sign-off if one exists,
        otherwise the machine's. Callers must use this rather than `.grade`
        anywhere a clinical decision is made."""
        if self.reviews:
            latest = self.reviews[-1]
            if latest.corrected_grade is not None:
                return latest.corrected_grade
        return self.grade


class Review(Base):
    """An ophthalmologist's sign-off. Appended, never overwriting the machine."""
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    screening_id: Mapped[str] = mapped_column(ForeignKey("screenings.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    corrected_grade: Mapped[int | None] = mapped_column(Integer, nullable=True)
    agrees_with_model: Mapped[bool] = mapped_column(Boolean, default=True)
    urgency_override: Mapped[Urgency | None] = mapped_column(Enum(Urgency), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=utcnow, index=True)

    screening: Mapped[Screening] = relationship(back_populates="reviews")


class AuditLog(Base):
    """Append-only record of everything that touched a clinical decision."""
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    actor_role: Mapped[str | None] = mapped_column(String(40), nullable=True)
    action: Mapped[str] = mapped_column(String(80), index=True)
    subject_type: Mapped[str] = mapped_column(String(40))
    subject_id: Mapped[str] = mapped_column(String(36), index=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    node_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class SyncBatch(Base):
    """Bookkeeping for offline edge nodes pushing captures upstream."""
    __tablename__ = "sync_batches"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    node_id: Mapped[str] = mapped_column(String(64), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    screenings_received: Mapped[int] = mapped_column(Integer, default=0)
    screenings_created: Mapped[int] = mapped_column(Integer, default=0)
    screenings_duplicate: Mapped[int] = mapped_column(Integer, default=0)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
