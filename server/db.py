"""Engine and session management.

SQLite and Postgres are both first-class here on purpose: the PHC edge node
runs SQLite with no server process, the district runs Postgres, and both use
the same ORM so a record written offline is byte-identical once synced.
"""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .models import Base

_engine = None
_SessionLocal = None


def _make_engine(url: str):
    kw = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        # check_same_thread=False is required because FastAPI serves requests
        # from a thread pool; SQLite's default would reject those connections.
        kw["connect_args"] = {"check_same_thread": False}
        kw.pop("pool_pre_ping")
    engine = create_engine(url, **kw)

    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            # Foreign keys are OFF by default in SQLite, which would let the
            # edge node accumulate screenings pointing at patients that do not
            # exist and only fail on sync.
            cur.execute("PRAGMA foreign_keys=ON")
            # WAL keeps reads working while a capture is being written, which
            # matters on a single-file edge database.
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    return engine


def get_engine():
    global _engine
    if _engine is None:
        _engine = _make_engine(get_settings().database_url)
    return _engine


def get_sessionmaker():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False,
                                     expire_on_commit=False, class_=Session)
    return _SessionLocal


def init_db():
    Base.metadata.create_all(bind=get_engine())


def reset_engine():
    """Drop cached engine/sessionmaker so a test can point at a new URL."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
    get_settings.cache_clear()


@contextmanager
def session_scope():
    s = get_sessionmaker()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_db():
    """FastAPI dependency."""
    s = get_sessionmaker()()
    try:
        yield s
    finally:
        s.close()
