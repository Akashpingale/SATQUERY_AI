"""Engine and session factory derived from ``DATABASE_URL``.

Local dev/tests use SQLite (no server required); Docker/production use PostgreSQL.
Only portable column types are used so the same models work on both.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from ..core.config import get_settings

_settings = get_settings()


def _ensure_sqlite_dir(database_url: str) -> None:
    """Create the parent directory for a SQLite file so the engine can open it."""
    if not database_url.startswith("sqlite"):
        return
    path = database_url.split("sqlite:///", 1)[-1]
    if path and path != ":memory:":
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)


_ensure_sqlite_dir(_settings.database_url)

_connect_args = {"check_same_thread": False} if _settings.is_sqlite else {}

engine = create_engine(
    _settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> tuple[bool, str]:
    """Return ``(connected, dialect)``. Never raises."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, engine.dialect.name
    except Exception:  # noqa: BLE001 - health check must not propagate
        return False, engine.dialect.name
