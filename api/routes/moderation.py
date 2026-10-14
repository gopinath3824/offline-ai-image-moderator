"""Endpoints for image content moderation."""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from api.schemas import BatchModerationResponse, ModerationResponse
from api.services import ModerationService, get_moderation_service
from utils.logging_config import get_logger

logger = get_logger("api_moderation")

router = APIRouter(prefix="/moderate", tags=["Moderation"])


@router.post(
    "",
    response_model=ModerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Moderate Single Image",
    description="Upload an image to evaluate unsafe content categories (NSFW, Nudity, Violence, etc.).",
)
async def moderate_image(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, WebP, etc.)"),
    model_id: Optional[str] = Form(
        None,
        description="Optional model override: falconsai_nsfw, clip_zero_shot, or synthetic_diagnostic",
    ),
    flag_threshold: Optional[float] = Form(
        None,
        ge=0.0,
        le=1.0,
        description="Override threshold to flag content as unsafe (default from config: 0.70)",
    ),
    review_threshold: Optional[float] = Form(
        None,
        ge=0.0,
        le=1.0,
        description="Override threshold for review uncertainty margin (default: 0.40)",
    ),
    save_history: bool = Form(
        True,
        description="Whether to persist the result in local SQLite history database",
    ),
    service: ModerationService = Depends(get_moderation_service),
) -> ModerationResponse:
    """Analyze a single uploaded image."""
    try:
        content = await file.read()
        filename = file.filename or "uploaded_image.jpg"
        return service.moderate_image_bytes(
            image_bytes=content,
            filename=filename,
            model_id=model_id,
            flag_threshold=flag_threshold,
            review_threshold=review_threshold,
            save_history=save_history,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as err:
        logger.error("Single moderation error: %s", err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(err)}",
        )


@router.post(
    "/batch",
    response_model=BatchModerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Moderate Batch of Images",
    description="Upload multiple images to receive moderation verdicts and aggregated statistics.",
)
async def moderate_batch(
    files: List[UploadFile] = File(..., description="List of images to evaluate"),
    model_id: Optional[str] = Form(None, description="Optional model ID override"),
    flag_threshold: Optional[float] = Form(None, ge=0.0, le=1.0, description="Flag threshold override"),
    review_threshold: Optional[float] = Form(None, ge=0.0, le=1.0, description="Review threshold override"),
    save_history: bool = Form(True, description="Persist all results in local SQLite history"),
    service: ModerationService = Depends(get_moderation_service),
) -> BatchModerationResponse:
    """Analyze multiple images in batch."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    payload: List[tuple[str, bytes]] = []
    for f in files:
        data = await f.read()
        payload.append((f.filename or "unnamed.jpg", data))

    return service.moderate_batch_bytes(
        files=payload,
        model_id=model_id,
        flag_threshold=flag_threshold,
        review_threshold=review_threshold,
        save_history=save_history,
    )
