"""Authentication and user administration."""
from __future__ import annotations

import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..audit import record
from ..config import get_settings
from ..db import get_db
from ..models import Facility, Role, User
from ..schemas import FacilityCreate, FacilityOut, Token, UserCreate, UserOut
from ..security import (create_access_token, current_user, hash_password,
                        require, verify_password)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/token", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == form.username.lower().strip()))
    # One message and one code for both "no such user" and "wrong password", so
    # the endpoint cannot be used to enumerate who works in the programme.
    if user is None or not verify_password(form.password, user.password_hash):
        record(db, None, "login_failed", "user", form.username, {})
        db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "incorrect email or password",
                            headers={"WWW-Authenticate": "Bearer"})
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "account disabled")

    user.last_login = datetime.now(timezone.utc)
    record(db, user, "login", "user", user.id, {})
    db.commit()

    s = get_settings()
    return Token(access_token=create_access_token(user), role=user.role,
                 user_id=user.id, full_name=user.full_name,
                 expires_in_minutes=s.access_token_minutes)


@router.post("/session", response_model=Token)
def open_session(db: Session = Depends(get_db)):
    """Mint a session without credentials, when open access is enabled.

    This exists so the dashboard can drop its sign-in screen on a demo or an
    edge node. It changes nothing about authorisation: the token carries a real
    role and every endpoint keeps checking capabilities, so an open-access
    session still cannot sign off a grade unless its role is ophthalmologist.

    Settings.check() refuses to start a district node with open access on, so
    this endpoint can only ever be reachable where that was a deliberate
    choice.
    """
    s = get_settings()
    if not s.open_access:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Open access is disabled. Obtain a token from /api/auth/token with "
            "credentials, or set DRISHTI_OPEN_ACCESS=1 on a demo/edge node.")

    try:
        role = Role(s.open_access_role)
    except ValueError:
        raise HTTPException(500, f"invalid open_access_role: {s.open_access_role!r}")

    email = f"open-access@{s.node_id}.local"
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        # A real row, so audit entries and reviews attribute to something that
        # can be inspected later rather than to a null actor.
        user = User(email=email, full_name=f"Open access ({role.value})",
                    password_hash=hash_password(secrets.token_urlsafe(32)),
                    role=role)
        db.add(user)
        db.flush()
        record(db, None, "open_access_user_created", "user", user.id,
               {"role": role.value, "node_id": s.node_id})
    elif user.role != role:
        user.role = role

    user.last_login = datetime.now(timezone.utc)
    record(db, user, "open_access_session", "user", user.id, {})
    db.commit()
    return Token(access_token=create_access_token(user), role=user.role,
                 user_id=user.id, full_name=user.full_name,
                 expires_in_minutes=s.access_token_minutes)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(payload: UserCreate, db: Session = Depends(get_db),
                actor: User = Depends(require("user:manage"))):
    email = payload.email.lower().strip()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")
    user = User(email=email, full_name=payload.full_name,
                password_hash=hash_password(payload.password),
                role=payload.role, facility_id=payload.facility_id)
    db.add(user)
    db.flush()
    record(db, actor, "user_created", "user", user.id,
           {"role": user.role.value, "email": email})
    db.commit()
    return user


@router.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db),
               _: User = Depends(require("user:manage"))):
    return list(db.scalars(select(User).order_by(User.created_at)))


@router.post("/users/{user_id}/deactivate", response_model=UserOut)
def deactivate(user_id: str, db: Session = Depends(get_db),
               actor: User = Depends(require("user:manage"))):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "user not found")
    if user.id == actor.id:
        # Locking the last admin out of the district server is unrecoverable
        # without database access.
        raise HTTPException(400, "cannot deactivate your own account")
    user.is_active = False
    record(db, actor, "user_deactivated", "user", user.id, {})
    db.commit()
    return user


@router.post("/facilities", response_model=FacilityOut, status_code=201)
def create_facility(payload: FacilityCreate, db: Session = Depends(get_db),
                    actor: User = Depends(require("user:manage"))):
    f = Facility(name=payload.name, district=payload.district, kind=payload.kind)
    db.add(f)
    db.flush()
    record(db, actor, "facility_created", "facility", f.id, {"name": f.name})
    db.commit()
    return f


@router.get("/facilities", response_model=list[FacilityOut])
def list_facilities(db: Session = Depends(get_db),
                    _: User = Depends(current_user)):
    return list(db.scalars(select(Facility).order_by(Facility.name)))
