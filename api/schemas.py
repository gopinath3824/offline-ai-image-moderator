"""Pydantic data schemas for request validation and response serialization."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health check and runtime status."""

    status: str = Field("ok", description="Service health state")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application semantic version")
    active_model: str = Field(..., description="Currently active model ID")
    offline_enforced: bool = Field(True, description="Strict local offline guarantee")


class CategoryScoreDetail(BaseModel):
    """Detailed score for an individual moderation category."""

    category: str = Field(..., description="Category identifier (e.g. nudity, violence)")
    display_name: str = Field(..., description="Human-friendly label")
    score: float = Field(..., ge=0.0, le=1.0, description="Raw confidence score (0.0 to 1.0)")
    score_percent: str = Field(..., description="Formatted percentage string")
    is_flagged: bool = Field(..., description="Whether this category exceeded flag threshold")


class ModerationResponse(BaseModel):
    """Result of single image content moderation."""

    filename: str = Field(..., description="Original image filename")
    verdict: str = Field(..., description="Verdict: Likely Safe, Needs Review, Flagged, or Error")
    requires_review: bool = Field(..., description="True if flagged or inside review uncertainty margin")
    highest_category: str = Field(..., description="Category with the highest confidence score")
    highest_score: float = Field(..., ge=0.0, le=1.0, description="Highest confidence score")
    highest_score_percent: str = Field(..., description="Formatted percentage of top score")
    category_scores: Dict[str, float] = Field(default_factory=dict, description="Raw category scores dict")
    category_breakdown: List[CategoryScoreDetail] = Field(
        default_factory=list, description="Ordered category breakdown"
    )
    model_id: str = Field(..., description="Identifier of the model used for inference")
    model_name: str = Field(..., description="Display name of the model")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    explanation: str = Field(..., description="Human-readable decision rationale")
    record_id: Optional[int] = Field(None, description="SQLite history record ID if persisted")
    dimensions: str = Field("", description="Image dimensions (e.g. 1920 x 1080 px)")
    file_size_display: str = Field("", description="Human-readable file size")
    timestamp: str = Field(..., description="Timestamp of evaluation")


class BatchModerationResponse(BaseModel):
    """Result of batch moderation operation."""

    total_images: int = Field(..., description="Total images evaluated")
    safe_count: int = Field(0, description="Images classified as Likely Safe")
    review_count: int = Field(0, description="Images classified as Needs Review")
    flagged_count: int = Field(0, description="Images classified as Flagged")
    error_count: int = Field(0, description="Images that failed processing")
    total_latency_ms: float = Field(0.0, description="Cumulative latency in milliseconds")
    results: List[ModerationResponse] = Field(default_factory=list, description="List of moderation results")


class ModelInfoResponse(BaseModel):
    """Metadata and local installation status for an AI model."""

    id: str = Field(..., description="Model unique identifier")
    name: str = Field(..., description="Model display title")
    description: str = Field(..., description="Architecture and moderation scope")
    approx_size_mb: int = Field(..., description="Disk size footprint in MB")
    installed: bool = Field(..., description="Whether weights are cached locally")
    local_path: str = Field(..., description="Local directory path")
    is_active: bool = Field(False, description="Whether currently set as active default")


class ModelsListResponse(BaseModel):
    """Listing of all supported models in catalog."""

    active_model_id: str = Field(..., description="Current default active model")
    models: List[ModelInfoResponse] = Field(default_factory=list, description="Available models")


class DownloadModelRequest(BaseModel):
    """Request to download model weights for offline usage."""

    model_id: str = Field(..., description="Model ID to download from Hugging Face")


class ScanRecordResponse(BaseModel):
    """Persisted history scan record."""

    id: int = Field(..., description="Unique record ID")
    filename: str = Field(..., description="Original image filename")
    verdict: str = Field(..., description="Moderation verdict")
    requires_review: bool = Field(..., description="Requires human moderator review")
    highest_category: str = Field(..., description="Highest detected category")
    highest_score: float = Field(..., description="Top confidence score")
    category_scores: Dict[str, float] = Field(default_factory=dict, description="Category scores map")
    model_id: str = Field(..., description="Model ID used")
    model_name: str = Field(..., description="Model display title")
    inference_latency_sec: float = Field(..., description="Inference latency in seconds")
    timestamp: str = Field(..., description="Scan timestamp")
    file_size_display: str = Field("", description="Formatted file size")
    dimensions_display: str = Field("", description="Image dimensions")
    error_message: str = Field("", description="Error message if failed")


class HistoryListResponse(BaseModel):
    """Paginated list of historical scan records."""

    total: int = Field(..., description="Total matching records count")
    limit: int = Field(..., description="Records limit")
    offset: int = Field(..., description="Records offset")
    records: List[ScanRecordResponse] = Field(default_factory=list, description="List of scan records")


class StatsResponse(BaseModel):
    """Aggregate moderation KPI counts."""

    total: int = Field(0, description="Total images scanned")
    safe: int = Field(0, description="Likely safe images")
    review: int = Field(0, description="Needs review images")
    flagged: int = Field(0, description="Flagged images")


class MessageResponse(BaseModel):
    """Generic status response message."""

    success: bool = Field(True, description="Operation success indicator")
    message: str = Field(..., description="Informational message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional context payload")
