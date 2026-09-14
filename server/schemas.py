"""Request/response models."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from .models import Role, ScreeningStatus, Urgency


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role
    user_id: str
    full_name: str
    expires_in_minutes: int


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=10, max_length=72)
    role: Role = Role.ASHA
    facility_id: str | None = None


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: str
    full_name: str
    role: Role
    facility_id: str | None
    is_active: bool


class FacilityCreate(BaseModel):
    name: str
    district: str
    kind: str = "phc"


class FacilityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    district: str
    kind: str


class PatientCreate(BaseModel):
    mrn: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    year_of_birth: int | None = Field(default=None, ge=1900, le=2100)
    sex: str | None = None
    phone: str | None = None
    diabetes_since_year: int | None = Field(default=None, ge=1900, le=2100)
    facility_id: str | None = None


class PatientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    mrn: str
    name: str
    year_of_birth: int | None
    sex: str | None
    phone: str | None
    diabetes_since_year: int | None
    facility_id: str | None
    created_at: datetime


class ScreeningOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    client_uuid: str
    patient_id: str
    eye: str
    status: ScreeningStatus
    captured_at: datetime
    quality_passed: bool | None
    quality_score: float | None
    quality_failures: list | None
    quality_guidance: str | None
    grade: int | None
    confidence: float | None
    grade_distribution: list | None
    referable_probability: float | None
    lesion_counts: dict | None
    explanation: list | None
    attention: dict | None
    urgency: Urgency | None
    days_to_review: int | None
    needs_human_review: bool
    triage_reasons: list | None
    model_version: str | None
    pipeline_version: str | None
    latency_ms: dict | None
    error: str | None
    final_grade: int | None = None


class ReviewCreate(BaseModel):
    corrected_grade: int | None = Field(default=None, ge=0, le=4)
    agrees_with_model: bool = True
    urgency_override: Urgency | None = None
    notes: str | None = None


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    screening_id: str
    reviewer_id: str
    corrected_grade: int | None
    agrees_with_model: bool
    urgency_override: Urgency | None
    notes: str | None
    created_at: datetime


class QueueItem(BaseModel):
    screening: ScreeningOut
    patient_name: str
    patient_mrn: str
    waiting_days: int
    breaching: bool


class ProgressionPoint(BaseModel):
    screening_id: str
    captured_at: datetime
    eye: str
    grade: int | None
    final_grade: int | None
    urgency: Urgency | None


class PatientHistory(BaseModel):
    patient: PatientOut
    timeline: list[ProgressionPoint]
    worst_grade: int | None
    progressed: bool
    progression_note: str | None


# ---------------------------------------------------------------- sync
class SyncScreening(BaseModel):
    """One capture pushed up from an offline edge node."""
    client_uuid: str
    patient_mrn: str
    patient_name: str
    eye: str
    captured_at: datetime
    image_base64: str | None = None
    image_sha256: str | None = None
    result: dict | None = None          # edge-computed result, if it graded locally


class SyncPush(BaseModel):
    node_id: str
    screenings: list[SyncScreening]


class SyncResult(BaseModel):
    batch_id: str
    received: int
    created: int
    duplicates: int
    errors: list[str] = []
