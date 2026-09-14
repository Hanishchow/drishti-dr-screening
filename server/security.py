"""Passwords, tokens and role enforcement.

Role rules encoded here, because they are clinical safety rules and not merely
access control:

  ASHA            captures images and sees only their own facility's patients.
                  Cannot sign off a grade -- that is the whole point of the
                  review queue.
  OPHTHALMOLOGIST reads the queue and signs off; may override any grade.
  DISTRICT_ADMIN  manages users and sees programme-wide statistics. NOT granted
                  sign-off: administrative seniority is not a clinical
                  qualification, and conflating the two is how unqualified
                  sign-off happens.
  SERVICE         machine-to-machine sync from an edge node. No human-facing
                  permissions at all.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .models import Role, User

ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)

# Capability -> roles allowed. Kept as data so the matrix is inspectable and
# testable rather than scattered through route decorators.
PERMISSIONS: dict[str, set[Role]] = {
    "screening:create": {Role.ASHA, Role.OPHTHALMOLOGIST, Role.DISTRICT_ADMIN, Role.SERVICE},
    "screening:read": {Role.ASHA, Role.OPHTHALMOLOGIST, Role.DISTRICT_ADMIN},
    "screening:read_all": {Role.OPHTHALMOLOGIST, Role.DISTRICT_ADMIN},
    "patient:create": {Role.ASHA, Role.OPHTHALMOLOGIST, Role.DISTRICT_ADMIN, Role.SERVICE},
    "patient:read": {Role.ASHA, Role.OPHTHALMOLOGIST, Role.DISTRICT_ADMIN},
    "review:read_queue": {Role.OPHTHALMOLOGIST, Role.DISTRICT_ADMIN},
    # Sign-off is deliberately ophthalmologist-only.
    "review:sign_off": {Role.OPHTHALMOLOGIST},
    "user:manage": {Role.DISTRICT_ADMIN},
    "stats:read": {Role.OPHTHALMOLOGIST, Role.DISTRICT_ADMIN},
    "sync:push": {Role.SERVICE, Role.DISTRICT_ADMIN},
}


def hash_password(plain: str) -> str:
    # bcrypt silently truncates at 72 bytes; reject rather than accept a
    # password whose tail is ignored.
    raw = plain.encode("utf-8")
    if len(raw) > 72:
        raise ValueError("password must be 72 bytes or fewer")
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


def create_access_token(user: User, minutes: int | None = None) -> str:
    s = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=minutes if minutes is not None else s.access_token_minutes)
    payload = {"sub": user.id, "role": user.role.value, "email": user.email,
               "facility_id": user.facility_id, "exp": expire}
    return jwt.encode(payload, s.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, get_settings().secret_key, algorithms=[ALGORITHM])
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {e}",
                            headers={"WWW-Authenticate": "Bearer"})


def current_user(token: str | None = Depends(oauth2_scheme),
                 db: Session = Depends(get_db)) -> User:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})
    payload = decode_token(token)
    user = db.get(User, payload.get("sub"))
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found or disabled")
    return user


def require(*capabilities: str):
    """Dependency factory enforcing capabilities from the matrix above."""
    for c in capabilities:
        if c not in PERMISSIONS:
            raise KeyError(f"unknown capability '{c}'")

    def dependency(user: User = Depends(current_user)) -> User:
        for c in capabilities:
            if user.role not in PERMISSIONS[c]:
                raise HTTPException(
                    status.HTTP_403_FORBIDDEN,
                    f"role '{user.role.value}' lacks capability '{c}'")
        return user

    return dependency


def can(user: User, capability: str) -> bool:
    return user.role in PERMISSIONS.get(capability, set())
