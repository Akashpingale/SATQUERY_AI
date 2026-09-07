"""SQLAlchemy declarative base.

Import ORM models through ``app.models_db`` so they are registered on this base's
metadata (needed by Alembic autogenerate and ``create_all`` in tests).
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""
