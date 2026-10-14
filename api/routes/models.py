"""Endpoints for managing offline neural models."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import DownloadModelRequest, MessageResponse, ModelsListResponse
from api.services import ModerationService, get_moderation_service
from utils.logging_config import get_logger

logger = get_logger("api_models")

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=ModelsListResponse,
    summary="List Model Catalog",
    description="List all supported models, active default, disk sizes, and local offline readiness.",
)
def list_models(
    service: ModerationService = Depends(get_moderation_service),
) -> ModelsListResponse:
    """Retrieve catalog of available models."""
    return service.get_models_catalog()


@router.post(
    "/download",
    response_model=MessageResponse,
    summary="Download Model Weights",
    description="Download weights from Hugging Face for 100% offline local execution.",
)
def download_model(
    payload: DownloadModelRequest,
    service: ModerationService = Depends(get_moderation_service),
) -> MessageResponse:
    """Download a model to the local cache directory."""
    try:
        path = service.download_model(payload.model_id)
        return MessageResponse(
            success=True,
            message=f"Model '{payload.model_id}' installed successfully for offline use.",
            data={"local_path": path, "model_id": payload.model_id},
        )
    except Exception as err:
        logger.error("Failed to download model %s: %s", payload.model_id, err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model download failed: {str(err)}",
        )
