"""Offline edge synchronisation.

The real rural constraint is not compute, it is connectivity. A PHC laptop must
keep screening when the link is down and reconcile later without losing or
duplicating anything.

The contract:

* The EDGE mints `client_uuid` for every capture. The district treats it as the
  idempotency key, so a batch retried after a timeout is recognised rather than
  duplicated -- a duplicate here is a duplicate clinical record, not a cosmetic
  problem.
* Patients are matched on `mrn`. An edge node that registered a patient offline
  and a district that already knows them converge on one record.
* A partial batch is not an error. Individual failures are reported per item so
  the edge can retry only what did not land, instead of resending gigabytes.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..config import get_settings
from ..db import get_db
from ..models import (Patient, Screening, ScreeningStatus, SyncBatch, Urgency,
                      User)
from ..schemas import SyncPush, SyncResult
from ..security import require
from .clinical import _apply_result, _decode_image, _store_image

router = APIRouter(prefix="/api/sync", tags=["sync"])


@router.post("/push", response_model=SyncResult)
async def push(payload: SyncPush, request: Request,
               db: Session = Depends(get_db),
               actor: User = Depends(require("sync:push"))):
    batch = SyncBatch(node_id=payload.node_id,
                      screenings_received=len(payload.screenings))
    db.add(batch)
    db.flush()

    created = duplicates = 0
    errors: list[str] = []
    service = getattr(request.app.state, "screening_service", None)

    for item in payload.screenings:
        try:
            existing = db.scalar(select(Screening).where(
                Screening.client_uuid == item.client_uuid))
            if existing:
                duplicates += 1
                continue

            patient = db.scalar(select(Patient).where(Patient.mrn == item.patient_mrn))
            if patient is None:
                patient = Patient(mrn=item.patient_mrn, name=item.patient_name,
                                  facility_id=actor.facility_id)
                db.add(patient)
                db.flush()

            captured = item.captured_at
            if captured.tzinfo is None:
                captured = captured.replace(tzinfo=timezone.utc)

            sc = Screening(client_uuid=item.client_uuid, patient_id=patient.id,
                           eye=item.eye, captured_at=captured,
                           facility_id=patient.facility_id or actor.facility_id,
                           captured_by_id=actor.id, image_path="",
                           status=ScreeningStatus.PENDING,
                           synced_at=datetime.now(timezone.utc))
            db.add(sc)
            db.flush()

            data = None
            if item.image_base64:
                try:
                    data = base64.b64decode(item.image_base64, validate=True)
                except (binascii.Error, ValueError) as e:
                    raise ValueError(f"invalid base64 image: {e}")
                digest = hashlib.sha256(data).hexdigest()
                if item.image_sha256 and item.image_sha256 != digest:
                    # A corrupted upload must not be graded and silently
                    # attributed to the patient.
                    raise ValueError("image checksum mismatch; refusing to store")
                path, digest = _store_image(data, sc.id)
                sc.image_path, sc.image_sha256 = path, digest

            if item.result:
                # The edge already graded it. Trust its result rather than
                # re-running: the edge used the same pipeline, and re-grading
                # would burn district GPU on work already done.
                _apply_result(sc, item.result)
            elif data is not None and service is not None:
                result = await service.run(_decode_image(data))
                _apply_result(sc, result)

            created += 1
        except Exception as e:                    # noqa: BLE001
            errors.append(f"{item.client_uuid}: {e}")

    batch.screenings_created = created
    batch.screenings_duplicate = duplicates
    batch.detail = {"errors": errors[:50]}
    record(db, actor, "sync_push", "sync_batch", batch.id,
           {"node_id": payload.node_id, "received": len(payload.screenings),
            "created": created, "duplicates": duplicates, "errors": len(errors)})
    db.commit()

    return SyncResult(batch_id=batch.id, received=len(payload.screenings),
                      created=created, duplicates=duplicates, errors=errors)


@router.get("/status")
def sync_status(db: Session = Depends(get_db),
                _: User = Depends(require("sync:push"))):
    s = get_settings()
    batches = list(db.scalars(select(SyncBatch)
                              .order_by(SyncBatch.received_at.desc()).limit(20)))
    unsynced = db.scalar(select(Screening).where(Screening.synced_at.is_(None)).limit(1))
    return {
        "node_id": s.node_id,
        "node_role": s.node_role,
        "recent_batches": [
            {"id": b.id, "node_id": b.node_id,
             "received_at": b.received_at.isoformat(),
             "received": b.screenings_received, "created": b.screenings_created,
             "duplicates": b.screenings_duplicate} for b in batches],
        "has_unsynced_local_records": unsynced is not None,
    }
