"""Append-only audit trail.

Every clinical decision and every human action that touches one is recorded.
This is not observability nice-to-have: a screening programme that cannot say
who graded a patient, with which model, and who overrode it, cannot defend a
missed diagnosis.

Entries are never updated or deleted. `record()` only stages the row; the
caller commits it inside the same transaction as the action itself, so an audit
entry can never exist for an action that was rolled back, nor an action without
its entry.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from .config import get_settings
from .models import AuditLog


def record(db: Session, actor, action: str, subject_type: str,
           subject_id: str, detail: dict | None = None):
    entry = AuditLog(
        actor_id=getattr(actor, "id", None) if actor else None,
        actor_role=(getattr(actor, "role").value
                    if actor is not None and getattr(actor, "role", None) else None),
        action=action,
        subject_type=subject_type,
        subject_id=str(subject_id),
        detail=detail or {},
        node_id=get_settings().node_id,
    )
    db.add(entry)
    return entry
