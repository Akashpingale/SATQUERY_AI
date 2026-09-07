"""FastAPI application factory.

Wires configuration, structured logging, a per-request correlation id, CORS, sanitized
error handling, and the versioned API router. Import ``app`` for ASGI servers, or call
``create_app()`` in tests.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.v1.router import api_v1_router
from .core.config import get_settings
from .core.logging import configure_logging, get_logger, request_id_ctx
from .schemas.common import ErrorDetail, ErrorResponse

logger = get_logger("satquery.app")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        summary="Agentic vision-language assistant for remote-sensing image analysis.",
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_ctx.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_ctx.reset(token)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):  # type: ignore[no-untyped-def]
        # Log full detail server-side; return a sanitized envelope to the client.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        body = ErrorResponse(
            error=ErrorDetail(
                code="internal_error",
                message="An internal error occurred.",
                request_id=request_id_ctx.get(),
            )
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)

    @app.get("/", tags=["system"], summary="Service banner")
    def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": f"{settings.api_v1_prefix}/health",
        }

    logger.info(
        "%s v%s initialized (env=%s)",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )
    return app


app = create_app()
