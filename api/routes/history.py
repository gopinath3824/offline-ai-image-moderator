"""Endpoints for auditing scan history and statistics."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.schemas import HistoryListResponse, MessageResponse, ScanRecordResponse, StatsResponse
from api.services import ModerationService, get_moderation_service
from utils.logging_config import get_logger

logger = get_logger("api_history")

router = APIRouter(prefix="/history", tags=["History & Audit"])


@router.get(
    "",
    response_model=HistoryListResponse,
    summary="List Scan History",
    description="Query historical scan records with pagination, keyword search, and verdict filters.",
)
def list_history(
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    search: str = Query("", description="Filename keyword search query"),
    verdict: str = Query("", description="Filter by verdict: Likely Safe, Needs Review, Flagged"),
    service: ModerationService = Depends(get_moderation_service),
) -> HistoryListResponse:
    """List historical moderation scans."""
    records = service.get_history(limit=limit, offset=offset, search_query=search, verdict_filter=verdict)
    return HistoryListResponse(
        total=len(records),
        limit=limit,
        offset=offset,
        records=records,
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Moderation Statistics",
    description="Aggregate moderation counts (total, safe, needs review, flagged).",
)
def get_stats(
    service: ModerationService = Depends(get_moderation_service),
) -> StatsResponse:
    """Retrieve total summary counts."""
    return service.get_stats()


@router.get(
    "/{record_id}",
    response_model=ScanRecordResponse,
    summary="Get Scan Record Details",
    description="Retrieve full category scores and metadata for a specific record ID.",
)
def get_record(
    record_id: int,
    service: ModerationService = Depends(get_moderation_service),
) -> ScanRecordResponse:
    """Fetch record by ID."""
    rec = service.get_record(record_id)
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record #{record_id} not found.",
        )
    return rec


@router.delete(
    "/{record_id}",
    response_model=MessageResponse,
    summary="Delete Scan Record",
    description="Remove a specific moderation record from the audit trail.",
)
def delete_record(
    record_id: int,
    service: ModerationService = Depends(get_moderation_service),
) -> MessageResponse:
    """Delete a single record."""
    deleted = service.delete_record(record_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record #{record_id} not found or could not be deleted.",
        )
    return MessageResponse(success=True, message=f"Record #{record_id} successfully deleted.")


@router.delete(
    "",
    response_model=MessageResponse,
    summary="Clear All History",
    description="Permanently delete all moderation records from local SQLite database.",
)
def clear_all_history(
    service: ModerationService = Depends(get_moderation_service),
) -> MessageResponse:
    """Clear entire scan history."""
    count = service.clear_history()
    return MessageResponse(success=True, message=f"Successfully cleared {count} records from history.")
