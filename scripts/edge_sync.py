"""Push an edge node's backlog to the district server.

Run on a timer at each PHC:

    python scripts/edge_sync.py --district https://district.example.in \
        --email edge-node-7@svc.gov.in --password ... --batch 25

Behaviour that matters on a bad link:

* Only records with `synced_at IS NULL` are sent, so a completed push is never
  repeated.
* Records are sent in small batches and marked synced per batch. A connection
  dropped halfway costs one batch, not the day's work.
* `client_uuid` is the idempotency key, so re-sending a batch the district
  already accepted is harmless -- it comes back as a duplicate, not a second
  patient record.
* Nothing is deleted locally. The edge keeps its own copy; sync is replication,
  not a handover.
"""
from __future__ import annotations

import argparse
import base64
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from sqlalchemy import select

from server.config import get_settings
from server.db import get_sessionmaker, init_db
from server.models import Patient, Screening


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sync an edge node to the district")
    p.add_argument("--district", required=True, help="district base URL")
    p.add_argument("--email", required=True, help="service account email")
    p.add_argument("--password", required=True)
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--include-images", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def collect(db, limit):
    stmt = (select(Screening, Patient)
            .join(Patient, Screening.patient_id == Patient.id)
            .where(Screening.synced_at.is_(None))
            .order_by(Screening.captured_at)
            .limit(limit))
    return list(db.execute(stmt))


def to_payload(sc: Screening, patient: Patient, include_images: bool):
    item = {
        "client_uuid": sc.client_uuid,
        "patient_mrn": patient.mrn,
        "patient_name": patient.name,
        "eye": sc.eye,
        "captured_at": sc.captured_at.isoformat(),
        "image_sha256": sc.image_sha256,
    }
    if include_images and sc.image_path and Path(sc.image_path).exists():
        item["image_base64"] = base64.b64encode(
            Path(sc.image_path).read_bytes()).decode()
    # Ship the grade the edge already computed. The district trusts it rather
    # than re-running: same pipeline version, and district GPU time is the
    # scarce resource the whole design is trying to conserve.
    if sc.grade is not None or sc.quality_passed is not None:
        item["result"] = {
            "graded": sc.grade is not None,
            "quality": {"passed": sc.quality_passed, "score": sc.quality_score,
                        "failures": sc.quality_failures,
                        "guidance": sc.quality_guidance},
            "grade": sc.grade,
            "confidence": sc.confidence,
            "grade_distribution": sc.grade_distribution,
            "referable_probability": sc.referable_probability,
            "lesion_counts": sc.lesion_counts,
            "explanation": sc.explanation,
            "attention": sc.attention,
            "triage": {"urgency": sc.urgency.value if sc.urgency else None,
                       "days_to_review": sc.days_to_review,
                       "needs_human_review": sc.needs_human_review,
                       "reasons": sc.triage_reasons},
            "model_version": sc.model_version,
            "model_thresholds": sc.model_thresholds,
            "pipeline_version": sc.pipeline_version,
            "timing_ms": sc.latency_ms,
        }
    return item


def main(argv=None):
    args = parse_args(argv)
    settings = get_settings()
    init_db()
    node_id = settings.node_id

    with httpx.Client(base_url=args.district.rstrip("/"),
                      timeout=args.timeout) as http:
        r = http.post("/api/auth/token",
                      data={"username": args.email, "password": args.password})
        if r.status_code != 200:
            print(f"authentication failed: {r.status_code} {r.text}")
            return 1
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

        total_sent = total_created = total_dup = 0
        while True:
            with get_sessionmaker()() as db:
                rows = collect(db, args.batch)
                if not rows:
                    break
                payload = {"node_id": node_id,
                           "screenings": [to_payload(sc, p, args.include_images)
                                          for sc, p in rows]}
                ids = [sc.id for sc, _ in rows]

            if args.dry_run:
                print(f"[dry-run] would send {len(ids)} screening(s)")
                break

            resp = http.post("/api/sync/push", headers=headers, json=payload)
            if resp.status_code != 200:
                print(f"push failed: {resp.status_code} {resp.text[:400]}")
                return 1
            body = resp.json()
            total_sent += body["received"]
            total_created += body["created"]
            total_dup += body["duplicates"]
            for err in body.get("errors", []):
                print(f"  district rejected {err}")

            # Mark synced only after the district confirms. A crash before this
            # point means the batch is resent and deduplicated upstream, which
            # is the safe direction.
            with get_sessionmaker()() as db:
                now = datetime.now(timezone.utc)
                for sid in ids:
                    sc = db.get(Screening, sid)
                    if sc:
                        sc.synced_at = now
                db.commit()
            print(f"  synced {len(ids)} (created {body['created']}, "
                  f"duplicate {body['duplicates']})", flush=True)

        print(f"done: sent {total_sent}, created {total_created}, "
              f"already present {total_dup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
