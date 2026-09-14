"""Backend tests: auth, roles, clinical records, review queue, offline sync.

Runs against a throwaway SQLite database with a stubbed grader, so the whole
suite needs no GPU, no trained model and no Postgres.
"""
from __future__ import annotations

import base64
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DRISHTI_ALLOW_INSECURE", "1")


# --------------------------------------------------------------- fixtures
@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DRISHTI_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("DRISHTI_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("DRISHTI_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DRISHTI_BOOTSTRAP_ADMIN_EMAIL", "admin@district.gov.in")
    monkeypatch.setenv("DRISHTI_BOOTSTRAP_ADMIN_PASSWORD", "bootstrap-secret-123")
    monkeypatch.setenv("DRISHTI_SECRET_KEY", "test-signing-key-not-the-default")

    from server import db as db_mod
    db_mod.reset_engine()

    from server.app import app
    from server.inference import ModelBundle

    # Stub the grader: deterministic, instant, no model file needed.
    class StubGrader:
        def __call__(self, batch):
            n = len(batch)
            # Mean pixel intensity stands in for severity, so different images
            # deterministically produce different grades.
            score = np.clip(np.array([b.mean() for b in batch]) * 2.0 + 1.5, 0, 4)
            dist = np.zeros((n, 5))
            for i, s in enumerate(score):
                lo = int(np.floor(s))
                dist[i, lo] = 1 - (s - lo)
                dist[i, min(lo + 1, 4)] += (s - lo)
            return score, dist

    bundle = ModelBundle(StubGrader(), "stub-v1", [0.5, 1.5, 2.5, 3.5], 256)
    with TestClient(app) as client:
        client.app.state.model_bundle = bundle
        client.app.state.screening_service.bundle = bundle
        client.app.state.screening_service.batcher.bundle = bundle
        client.app.state.screening_service.batcher.start()
        yield client
    db_mod.reset_engine()


def token(client, email, password):
    r = client.post("/api/auth/token", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture()
def admin(app_client):
    return token(app_client, "admin@district.gov.in", "bootstrap-secret-123")


def make_user(client, admin_hdr, role, email=None, facility_id=None):
    email = email or f"{role}-{uuid.uuid4().hex[:6]}@phc.gov.in"
    r = client.post("/api/auth/users", headers=admin_hdr, json={
        "email": email, "full_name": f"Test {role}", "password": "a-long-password-1",
        "role": role, "facility_id": facility_id})
    assert r.status_code == 201, r.text
    return email, token(client, email, "a-long-password-1")


def fundus_png(severity=0.3, size=320):
    """A retina-shaped image so the FOV crop and quality gate behave."""
    img = np.zeros((size, size, 3), np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.45),
               (40, 80, int(120 + 80 * severity)), -1)
    rng = np.random.default_rng(int(severity * 1000))
    for _ in range(40):
        p = rng.integers(size // 4, 3 * size // 4, 2)
        cv2.circle(img, tuple(int(v) for v in p), 3, (30, 50, 140), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    return cv2.imencode(".png", img)[1].tobytes()


def create_patient(client, hdr, mrn=None):
    mrn = mrn or f"MRN{uuid.uuid4().hex[:8].upper()}"
    r = client.post("/api/patients", headers=hdr,
                    json={"mrn": mrn, "name": "Lakshmi Venkataraman",
                          "year_of_birth": 1968, "sex": "F"})
    assert r.status_code in (200, 201), r.text
    return r.json()


def upload(client, hdr, patient_id, eye="R", severity=0.3, client_uuid=None):
    data = {"patient_id": patient_id, "eye": eye}
    if client_uuid:
        data["client_uuid"] = client_uuid
    return client.post("/api/screenings", headers=hdr, data=data,
                       files={"file": ("eye.png", fundus_png(severity), "image/png")})


# ------------------------------------------------------------------- auth
def test_health_reports_model_state(app_client):
    r = app_client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model"]["available"] is True


def test_login_rejects_wrong_password_without_revealing_the_account(app_client):
    bad_user = app_client.post("/api/auth/token", data={
        "username": "nobody@nowhere.in", "password": "whatever-123"})
    bad_pass = app_client.post("/api/auth/token", data={
        "username": "admin@district.gov.in", "password": "wrong-password-1"})
    assert bad_user.status_code == bad_pass.status_code == 401
    # Identical responses, so the endpoint cannot enumerate programme staff.
    assert bad_user.json()["detail"] == bad_pass.json()["detail"]


def test_unauthenticated_requests_are_rejected(app_client):
    assert app_client.get("/api/patients").status_code == 401
    assert app_client.get("/api/review/queue").status_code == 401


def test_bootstrap_admin_only_created_once(app_client, admin):
    users = app_client.get("/api/auth/users", headers=admin).json()
    assert sum(u["role"] == "district_admin" for u in users) == 1


# ------------------------------------------------------------------ roles
def test_asha_cannot_sign_off_a_grade(app_client, admin):
    """The core safety rule: capture and clinical sign-off are separate."""
    _, asha = make_user(app_client, admin, "asha")
    patient = create_patient(app_client, asha)
    sc = upload(app_client, asha, patient["id"]).json()
    r = app_client.post(f"/api/review/{sc['id']}", headers=asha,
                        json={"corrected_grade": 0})
    assert r.status_code == 403
    assert "review:sign_off" in r.json()["detail"]


def test_district_admin_cannot_sign_off_either(app_client, admin):
    """Administrative seniority is not a clinical qualification."""
    _, asha = make_user(app_client, admin, "asha")
    patient = create_patient(app_client, asha)
    sc = upload(app_client, asha, patient["id"]).json()
    r = app_client.post(f"/api/review/{sc['id']}", headers=admin,
                        json={"corrected_grade": 0})
    assert r.status_code == 403


def test_only_admin_manages_users(app_client, admin):
    _, ophth = make_user(app_client, admin, "ophthalmologist")
    r = app_client.post("/api/auth/users", headers=ophth, json={
        "email": "x@y.in", "full_name": "X", "password": "a-long-password-1",
        "role": "asha"})
    assert r.status_code == 403


def test_asha_cannot_read_another_facilitys_patient(app_client, admin):
    f1 = app_client.post("/api/auth/facilities", headers=admin,
                         json={"name": "PHC Ramanathapuram", "district": "Sivaganga"}).json()
    f2 = app_client.post("/api/auth/facilities", headers=admin,
                         json={"name": "PHC Tirupattur", "district": "Sivaganga"}).json()
    _, asha1 = make_user(app_client, admin, "asha", facility_id=f1["id"])
    _, asha2 = make_user(app_client, admin, "asha", facility_id=f2["id"])
    patient = create_patient(app_client, asha1)
    r = app_client.get(f"/api/patients/{patient['id']}/history", headers=asha2)
    assert r.status_code == 403


def test_admin_cannot_deactivate_themselves(app_client, admin):
    me = app_client.get("/api/auth/me", headers=admin).json()
    r = app_client.post(f"/api/auth/users/{me['id']}/deactivate", headers=admin)
    assert r.status_code == 400


# -------------------------------------------------------------- screening
def test_screening_records_the_model_version_used(app_client, admin):
    """A grade that cannot be traced to a model version cannot be reproduced
    after an update, which makes the audit trail worthless."""
    _, asha = make_user(app_client, admin, "asha")
    patient = create_patient(app_client, asha)
    sc = upload(app_client, asha, patient["id"]).json()
    assert sc["model_version"] == "stub-v1"
    assert sc["pipeline_version"]
    assert sc["status"] in ("graded", "needs_review", "quality_failed")


def test_repeat_upload_with_same_client_uuid_is_idempotent(app_client, admin):
    """A flaky rural link makes the client retry. A retry must not create a
    second clinical record for one capture."""
    _, asha = make_user(app_client, admin, "asha")
    patient = create_patient(app_client, asha)
    cid = str(uuid.uuid4())
    a = upload(app_client, asha, patient["id"], client_uuid=cid).json()
    b = upload(app_client, asha, patient["id"], client_uuid=cid).json()
    assert a["id"] == b["id"]
    all_sc = app_client.get(f"/api/patients/{patient['id']}/screenings",
                            headers=asha).json()
    assert len(all_sc) == 1


def test_registering_a_known_mrn_returns_the_existing_patient(app_client, admin):
    """A returning patient is the normal case in annual screening; creating a
    duplicate would sever the longitudinal series."""
    _, asha = make_user(app_client, admin, "asha")
    p1 = create_patient(app_client, asha, mrn="MRN-STABLE-1")
    p2 = create_patient(app_client, asha, mrn="MRN-STABLE-1")
    assert p1["id"] == p2["id"]


def test_rejects_undecodable_image(app_client, admin):
    _, asha = make_user(app_client, admin, "asha")
    patient = create_patient(app_client, asha)
    r = app_client.post("/api/screenings", headers=asha,
                        data={"patient_id": patient["id"], "eye": "R"},
                        files={"file": ("x.png", b"not-an-image", "image/png")})
    assert r.status_code == 400


def test_rejects_invalid_eye(app_client, admin):
    _, asha = make_user(app_client, admin, "asha")
    patient = create_patient(app_client, asha)
    r = app_client.post("/api/screenings", headers=asha,
                        data={"patient_id": patient["id"], "eye": "X"},
                        files={"file": ("x.png", fundus_png(), "image/png")})
    assert r.status_code == 400


# ----------------------------------------------------------------- review
def test_sign_off_supersedes_without_erasing_the_machine_grade(app_client, admin):
    _, asha = make_user(app_client, admin, "asha")
    _, ophth = make_user(app_client, admin, "ophthalmologist")
    patient = create_patient(app_client, asha)
    sc = upload(app_client, asha, patient["id"]).json()
    machine_grade = sc["grade"]

    corrected = 0 if machine_grade != 0 else 4
    r = app_client.post(f"/api/review/{sc['id']}", headers=ophth,
                        json={"corrected_grade": corrected, "notes": "Re-read."})
    assert r.status_code == 201
    assert r.json()["agrees_with_model"] is False

    after = app_client.get(f"/api/screenings/{sc['id']}", headers=ophth).json()
    assert after["grade"] == machine_grade, "machine grade must be preserved"
    assert after["final_grade"] == corrected, "final grade must follow the human"
    assert after["status"] == "reviewed"


def test_agreement_flag_follows_the_grades_not_the_checkbox(app_client, admin):
    """A reviewer who types a different grade has disagreed, whatever they
    ticked."""
    _, asha = make_user(app_client, admin, "asha")
    _, ophth = make_user(app_client, admin, "ophthalmologist")
    patient = create_patient(app_client, asha)
    sc = upload(app_client, asha, patient["id"]).json()
    different = 0 if sc["grade"] != 0 else 4
    r = app_client.post(f"/api/review/{sc['id']}", headers=ophth,
                        json={"corrected_grade": different,
                              "agrees_with_model": True})
    assert r.json()["agrees_with_model"] is False


def test_queue_orders_emergency_ahead_of_older_routine(app_client, admin):
    """Strict FIFO would be clinically wrong."""
    from server.db import get_sessionmaker
    from server.models import Screening, ScreeningStatus, Urgency

    _, asha = make_user(app_client, admin, "asha")
    _, ophth = make_user(app_client, admin, "ophthalmologist")
    patient = create_patient(app_client, asha)
    old = upload(app_client, asha, patient["id"], severity=0.1).json()
    new = upload(app_client, asha, patient["id"], eye="L", severity=0.9).json()

    with get_sessionmaker()() as db:
        a = db.get(Screening, old["id"])
        a.status, a.urgency = ScreeningStatus.NEEDS_REVIEW, Urgency.ROUTINE
        a.captured_at = datetime.now(timezone.utc) - timedelta(days=60)
        b = db.get(Screening, new["id"])
        b.status, b.urgency = ScreeningStatus.NEEDS_REVIEW, Urgency.EMERGENCY
        b.captured_at = datetime.now(timezone.utc)
        db.commit()

    q = app_client.get("/api/review/queue", headers=ophth).json()
    assert q[0]["screening"]["id"] == new["id"], "emergency must outrank older routine"


def test_queue_flags_window_breaches(app_client, admin):
    from server.db import get_sessionmaker
    from server.models import Screening, ScreeningStatus, Urgency

    _, asha = make_user(app_client, admin, "asha")
    _, ophth = make_user(app_client, admin, "ophthalmologist")
    patient = create_patient(app_client, asha)
    sc = upload(app_client, asha, patient["id"]).json()
    with get_sessionmaker()() as db:
        row = db.get(Screening, sc["id"])
        row.status, row.urgency = ScreeningStatus.NEEDS_REVIEW, Urgency.URGENT
        row.captured_at = datetime.now(timezone.utc) - timedelta(days=45)  # window 28
        db.commit()
    item = app_client.get("/api/review/queue", headers=ophth).json()[0]
    assert item["breaching"] is True and item["waiting_days"] >= 45


def test_stats_expose_reviewer_agreement(app_client, admin):
    _, asha = make_user(app_client, admin, "asha")
    _, ophth = make_user(app_client, admin, "ophthalmologist")
    patient = create_patient(app_client, asha)
    sc = upload(app_client, asha, patient["id"]).json()
    app_client.post(f"/api/review/{sc['id']}", headers=ophth,
                    json={"corrected_grade": sc["grade"]})
    s = app_client.get("/api/review/stats/summary", headers=admin).json()
    assert s["total_screenings"] >= 1
    assert s["reviewer_agreement"] == 1.0


# ------------------------------------------------------------------- sync
def test_offline_batch_creates_patient_and_screening(app_client, admin):
    _, svc = make_user(app_client, admin, "service")
    payload = {"node_id": "phc-edge-7", "screenings": [{
        "client_uuid": str(uuid.uuid4()),
        "patient_mrn": "EDGE-0001", "patient_name": "Meenakshi Sundaram",
        "eye": "R", "captured_at": datetime.now(timezone.utc).isoformat(),
        "image_base64": base64.b64encode(fundus_png()).decode()}]}
    r = app_client.post("/api/sync/push", headers=svc, json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["created"] == 1 and r.json()["duplicates"] == 0


def test_resending_the_same_batch_is_idempotent(app_client, admin):
    """A truck carrying a laptop through patchy coverage will resend. Twice."""
    _, svc = make_user(app_client, admin, "service")
    payload = {"node_id": "phc-edge-7", "screenings": [{
        "client_uuid": str(uuid.uuid4()),
        "patient_mrn": "EDGE-0002", "patient_name": "Arivazhagan Perumal",
        "eye": "L", "captured_at": datetime.now(timezone.utc).isoformat(),
        "image_base64": base64.b64encode(fundus_png()).decode()}]}
    first = app_client.post("/api/sync/push", headers=svc, json=payload).json()
    second = app_client.post("/api/sync/push", headers=svc, json=payload).json()
    assert first["created"] == 1
    assert second["created"] == 0 and second["duplicates"] == 1


def test_sync_rejects_checksum_mismatch(app_client, admin):
    """A corrupted image must never be graded and attributed to a patient."""
    _, svc = make_user(app_client, admin, "service")
    payload = {"node_id": "phc-edge-7", "screenings": [{
        "client_uuid": str(uuid.uuid4()),
        "patient_mrn": "EDGE-0003", "patient_name": "Kalaiselvi Raman",
        "eye": "R", "captured_at": datetime.now(timezone.utc).isoformat(),
        "image_base64": base64.b64encode(fundus_png()).decode(),
        "image_sha256": "0" * 64}]}
    r = app_client.post("/api/sync/push", headers=svc, json=payload).json()
    assert r["created"] == 0
    assert any("checksum" in e for e in r["errors"])


def test_partial_batch_failure_does_not_lose_the_good_records(app_client, admin):
    _, svc = make_user(app_client, admin, "service")
    good = {"client_uuid": str(uuid.uuid4()), "patient_mrn": "EDGE-OK",
            "patient_name": "Nithya Balasubramanian", "eye": "R",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "image_base64": base64.b64encode(fundus_png()).decode()}
    bad = {"client_uuid": str(uuid.uuid4()), "patient_mrn": "EDGE-BAD",
           "patient_name": "Corrupt Record", "eye": "R",
           "captured_at": datetime.now(timezone.utc).isoformat(),
           "image_base64": "!!!not-base64!!!"}
    r = app_client.post("/api/sync/push", headers=svc,
                        json={"node_id": "phc-edge-9",
                              "screenings": [good, bad]}).json()
    assert r["created"] == 1 and len(r["errors"]) == 1


def test_asha_cannot_push_sync_batches(app_client, admin):
    _, asha = make_user(app_client, admin, "asha")
    r = app_client.post("/api/sync/push", headers=asha,
                        json={"node_id": "x", "screenings": []})
    assert r.status_code == 403


# ------------------------------------------------------------ longitudinal
def test_history_detects_progression_between_visits(app_client, admin):
    """Progression is the signal an annual programme exists to catch."""
    from server.db import get_sessionmaker
    from server.models import Screening

    _, asha = make_user(app_client, admin, "asha")
    patient = create_patient(app_client, asha)
    a = upload(app_client, asha, patient["id"], severity=0.1).json()
    b = upload(app_client, asha, patient["id"], eye="L", severity=0.9).json()

    with get_sessionmaker()() as db:
        first = db.get(Screening, a["id"])
        first.grade = 0
        first.captured_at = datetime.now(timezone.utc) - timedelta(days=400)
        second = db.get(Screening, b["id"])
        second.grade = 3
        db.commit()

    h = app_client.get(f"/api/patients/{patient['id']}/history", headers=asha).json()
    assert h["worst_grade"] == 3
    assert h["progressed"] is True
    assert "rose from 0 to 3" in h["progression_note"]


# ------------------------------------------------------------------ audit
def test_every_grade_and_review_is_audited(app_client, admin):
    from server.db import get_sessionmaker
    from server.models import AuditLog

    _, asha = make_user(app_client, admin, "asha")
    _, ophth = make_user(app_client, admin, "ophthalmologist")
    patient = create_patient(app_client, asha)
    sc = upload(app_client, asha, patient["id"]).json()
    app_client.post(f"/api/review/{sc['id']}", headers=ophth,
                    json={"corrected_grade": 1})

    with get_sessionmaker()() as db:
        actions = {a.action for a in db.query(AuditLog).all()}
    assert {"screening_graded", "screening_reviewed", "patient_created",
            "user_created", "login"} <= actions


def test_failed_login_is_audited(app_client):
    from server.db import get_sessionmaker
    from server.models import AuditLog

    app_client.post("/api/auth/token",
                    data={"username": "intruder@example.com", "password": "guessing1"})
    with get_sessionmaker()() as db:
        assert db.query(AuditLog).filter(AuditLog.action == "login_failed").count() >= 1


# ------------------------------------------------- degraded / no model
def test_missing_model_refuses_cleanly_and_keeps_the_record(app_client, admin):
    """A fresh deployment has no trained artefact. Grading must refuse with an
    actionable 503 rather than crashing -- and the capture must still be
    persisted as FAILED, because an image that never got graded is a patient
    who never got an answer and that has to stay visible."""
    from server.db import get_sessionmaker
    from server.models import Screening, ScreeningStatus

    _, asha = make_user(app_client, admin, "asha")
    patient = create_patient(app_client, asha)

    service = app_client.app.state.screening_service
    original = service.batcher.bundle
    try:
        from server.inference import ModelBundle
        service.batcher.bundle = ModelBundle(None, "absent", [0.5, 1.5, 2.5, 3.5], 256)
        r = upload(app_client, asha, patient["id"])
        assert r.status_code == 503
        assert "dr/train.py" in r.json()["detail"]
    finally:
        service.batcher.bundle = original

    with get_sessionmaker()() as db:
        rows = db.query(Screening).filter(
            Screening.patient_id == patient["id"]).all()
    assert len(rows) == 1
    assert rows[0].status == ScreeningStatus.FAILED
    assert rows[0].error


def test_quality_failure_short_circuits_before_the_grader(app_client, admin):
    """An unusable capture must cost milliseconds, not a GPU slot."""
    _, asha = make_user(app_client, admin, "asha")
    patient = create_patient(app_client, asha)
    blank = np.zeros((300, 300, 3), np.uint8)
    cv2.circle(blank, (150, 150), 130, (40, 80, 180), -1)   # flat: no structure
    png = cv2.imencode(".png", blank)[1].tobytes()
    r = app_client.post("/api/screenings", headers=asha,
                        data={"patient_id": patient["id"], "eye": "R"},
                        files={"file": ("e.png", png, "image/png")})
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "quality_failed"
    assert body["grade"] is None
    assert body["quality_guidance"]
