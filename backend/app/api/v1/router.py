"""Aggregates all v1 route modules under a single router."""

from __future__ import annotations

from fastapi import APIRouter

from .routes import analysis, health, models, tasks, uploads

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(models.router)
api_v1_router.include_router(tasks.router)
api_v1_router.include_router(uploads.router)
api_v1_router.include_router(analysis.router)
