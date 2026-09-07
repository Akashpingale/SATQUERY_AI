"""Test fixtures.

Configures a throwaway SQLite database BEFORE importing the application, so tests run
with no GPU and no running PostgreSQL. Tables are created directly from the ORM
metadata for speed (the Alembic migration is exercised separately during verification).
"""

from __future__ import annotations

import os
import tempfile

import pytest

# --- Configure environment before any app import ---------------------------------
_TMP_DIR = tempfile.mkdtemp(prefix="satquery_test_")
_DB_PATH = os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["APP_ENV"] = "test"
os.environ["LOG_FORMAT"] = "console"
os.environ["LOG_LEVEL"] = "WARNING"
# Isolate uploads/previews from the developer's real storage directory.
os.environ["STORAGE_DIR"] = os.path.join(_TMP_DIR, "storage")
# Tests must never reach the network. Model weights are used only if already cached
# locally; otherwise the model-dependent tests skip (see the `models_ready` fixture).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()  # ensure our env vars are picked up

from app.db.base import Base  # noqa: E402
from app.db.session import engine  # noqa: E402
import app.models_db  # noqa: E402,F401  (registers models on Base.metadata)

Base.metadata.create_all(bind=engine)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def models_ready() -> None:
    """Skip a test unless the Phase 4 model weights are already cached locally.

    Tests never download: a clean machine without the cache skips these rather than
    pulling ~1 GB or silently passing against a model that is not there.
    """
    from ml.inference import CLIP_RS_MODEL_ID, OWLVIT_MODEL_ID, is_cached

    missing = [
        model_id for model_id in (CLIP_RS_MODEL_ID, OWLVIT_MODEL_ID) if not is_cached(model_id)
    ]
    if missing:
        pytest.skip(f"model weights not cached locally: {', '.join(missing)}")
