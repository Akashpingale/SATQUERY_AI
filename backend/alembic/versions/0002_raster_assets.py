"""raster assets table

Revision ID: 0002_raster_assets
Revises: 0001_initial
Create Date: 2026-01-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_raster_assets"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "raster_assets",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("original_filename", sa.String(length=256), nullable=False),
        sa.Column("stored_filename", sa.String(length=256), nullable=False),
        sa.Column("storage_path", sa.String(length=512), nullable=False),
        sa.Column("preview_path", sa.String(length=512), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=True),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("modality", sa.String(length=32), nullable=False),
        sa.Column("detected_modality", sa.String(length=32), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("band_count", sa.Integer(), nullable=True),
        sa.Column("dtype", sa.String(length=32), nullable=True),
        sa.Column("driver", sa.String(length=32), nullable=True),
        sa.Column("crs", sa.String(length=128), nullable=True),
        sa.Column("crs_epsg", sa.Integer(), nullable=True),
        sa.Column("is_georeferenced", sa.Boolean(), nullable=False),
        sa.Column("raster_metadata", sa.JSON(), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=True),
    )
    op.create_index("ix_raster_assets_role", "raster_assets", ["role"])


def downgrade() -> None:
    op.drop_index("ix_raster_assets_role", table_name="raster_assets")
    op.drop_table("raster_assets")
