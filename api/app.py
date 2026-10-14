"""FastAPI application factory and middleware configuration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import APP_NAME, APP_VERSION, AppConfig, ensure_directories_exist
from storage.database import init_database
from utils.logging_config import get_logger, setup_logging
from api.routes import history, models, moderation
from api.schemas import HealthResponse
from api.services import get_moderation_service

logger = get_logger("api_app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle events."""
    setup_logging()
    ensure_directories_exist()
    init_database()
    # Initialize service & enforce offline locks
    service = get_moderation_service()
    logger.info("REST API initialized. Active model: %s", service.config.active_model_id)
    yield
    logger.info("REST API shutting down.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    app = FastAPI(
        title=f"{APP_NAME} REST API",
        version=APP_VERSION,
        description=(
            "Privacy-first, 100% local on-device AI image moderation REST API. "
            "Evaluates images for NSFW, nudity, suggestive content, violence, gore, "
            "and hate symbols without any cloud calls or telemetry."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health & status endpoints
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Service Health Check",
    )
    def health_check() -> HealthResponse:
        service = get_moderation_service()
        return HealthResponse(
            status="ok",
            app_name=APP_NAME,
            version=APP_VERSION,
            active_model=service.config.active_model_id,
            offline_enforced=True,
        )

    @app.get(
        "/",
        tags=["System"],
        summary="API Root",
    )
    def root() -> dict[str, str]:
        return {
            "name": f"{APP_NAME} REST API",
            "version": APP_VERSION,
            "docs": "/docs",
            "health": "/health",
            "endpoints": "/api/v1",
        }

    # Register API v1 routes
    app.include_router(moderation.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(history.router, prefix="/api/v1")

    return app


app = create_app()
