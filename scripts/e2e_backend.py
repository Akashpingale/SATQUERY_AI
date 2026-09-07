"""Launch an isolated backend for Playwright end-to-end tests.

Uses a throwaway SQLite database and storage directory so an E2E run never touches development
data, forces offline model use (cached weights only), and creates tables directly from the ORM
metadata (no Alembic step needed for a scratch DB). Started by Playwright's ``webServer`` config.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

_TMP = Path(tempfile.mkdtemp(prefix="satquery_e2e_"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(_TMP / 'e2e.db').as_posix()}")
os.environ.setdefault("STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("LOG_FORMAT", "console")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
os.environ.setdefault("DEVICE", "cpu")
# Use only cached model weights; never download during an E2E run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "models" / "hf-cache"))

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

import app.models_db  # noqa: E402,F401  (registers ORM models on Base.metadata)
from app.db.base import Base  # noqa: E402
from app.db.session import engine  # noqa: E402

Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, log_level="warning")
