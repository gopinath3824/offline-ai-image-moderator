"""Moderation services connecting REST endpoints to core ML and SQLite engines."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import CATEGORY_DISPLAY_NAMES, AppConfig
from moderation.classifier import ClassifierRegistry
from moderation.image_loader import LoadedImageData, load_image_from_bytes
from moderation.model_manager import ModelManager
from moderation.scoring import ModerationResult, ScoringEngine
from storage.history_repository import HistoryRepository, ScanRecord
from utils.logging_config import get_logger
from api.schemas import (
    CategoryScoreDetail,
    ModerationResponse,
    BatchModerationResponse,
    ScanRecordResponse,
    ModelInfoResponse,
    ModelsListResponse,
    StatsResponse,
)

logger = get_logger("api_services")


class ModerationService:
    """Service handling image moderation, model lifecycles, and audit history."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.load()
        self.model_manager = ModelManager(self.config)
        self.history_repo = HistoryRepository()

        # Enforce local-only offline environment
        self.model_manager.enforce_offline_environment()

    def moderate_image_bytes(
        self,
        image_bytes: bytes,
        filename: str = "upload.jpg",
        model_id: Optional[str] = None,
        flag_threshold: Optional[float] = None,
        review_threshold: Optional[float] = None,
        save_history: bool = True,
    ) -> ModerationResponse:
        """Process in-memory image bytes through the neural moderation pipeline."""
        start_time = time.perf_counter()
        target_model_id = model_id or self.config.active_model_id
        flag_th = flag_threshold if flag_threshold is not None else self.config.flag_threshold
        rev_th = review_threshold if review_threshold is not None else self.config.review_threshold

        # 1. Safely load and normalize image
        loaded: LoadedImageData = load_image_from_bytes(
            data=image_bytes,
            filename=filename,
            max_size_mb=self.config.max_file_size_mb,
            max_dimension=self.config.max_image_dimension,
        )

        # 2. Retrieve local offline classifier
        classifier = ClassifierRegistry.get_classifier(
            model_id=target_model_id,
            config=self.config,
            model_manager=self.model_manager,
        )

        # 3. Predict category scores
        scores: Dict[str, float] = classifier.predict(loaded.image)
        latency_sec = time.perf_counter() - start_time
        latency_ms = round(latency_sec * 1000, 2)

        # 4. Evaluate scores against moderation rules
        result: ModerationResult = ScoringEngine.evaluate(
            category_scores=scores,
            flag_threshold=flag_th,
            review_threshold=rev_th,
            enabled_categories=self.config.enabled_categories,
            model_id=target_model_id,
            model_name=classifier.model_name,
        )

        # 5. Persist audit trail if requested
        record_id: Optional[int] = None
        if save_history:
            record = ScanRecord(
                filename=filename,
                file_path="",
                file_size_bytes=loaded.file_size_bytes,
                dimensions=loaded.dimensions_display,
                format_name=loaded.format_name,
                verdict=result.verdict,
                requires_review=result.requires_review,
                highest_category=result.highest_category,
                highest_score=result.highest_score,
                category_scores=result.category_scores,
                model_id=target_model_id,
                model_name=classifier.model_name,
                inference_latency_sec=round(latency_sec, 4),
            )
            record_id = self.history_repo.save_record(record)

        # 6. Build ordered category breakdown
        breakdown: List[CategoryScoreDetail] = []
        for cat, sc in sorted(result.category_scores.items(), key=lambda x: x[1], reverse=True):
            display_name = CATEGORY_DISPLAY_NAMES.get(cat, cat.replace("_", " ").title())
            breakdown.append(
                CategoryScoreDetail(
                    category=cat,
                    display_name=display_name,
                    score=round(sc, 4),
                    score_percent=f"{sc * 100:.1f}%",
                    is_flagged=sc >= flag_th,
                )
            )

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return ModerationResponse(
            filename=filename,
            verdict=result.verdict,
            requires_review=result.requires_review,
            highest_category=result.highest_category_display,
            highest_score=round(result.highest_score, 4),
            highest_score_percent=result.highest_score_pct,
            category_scores={k: round(v, 4) for k, v in result.category_scores.items()},
            category_breakdown=breakdown,
            model_id=target_model_id,
            model_name=classifier.model_name,
            latency_ms=latency_ms,
            explanation=result.explanation,
            record_id=record_id,
            dimensions=loaded.dimensions_display,
            file_size_display=loaded.file_size_display,
            timestamp=now_str,
        )

    def moderate_batch_bytes(
        self,
        files: List[Tuple[str, bytes]],
        model_id: Optional[str] = None,
        flag_threshold: Optional[float] = None,
        review_threshold: Optional[float] = None,
        save_history: bool = True,
    ) -> BatchModerationResponse:
        """Process multiple images sequentially with aggregated counts."""
        start_batch = time.perf_counter()
        results: List[ModerationResponse] = []
        safe_c = 0
        review_c = 0
        flagged_c = 0
        err_c = 0

        for filename, data in files:
            try:
                res = self.moderate_image_bytes(
                    image_bytes=data,
                    filename=filename,
                    model_id=model_id,
                    flag_threshold=flag_threshold,
                    review_threshold=review_threshold,
                    save_history=save_history,
                )
                results.append(res)
                if res.verdict == "Likely Safe":
                    safe_c += 1
                elif res.verdict == "Needs Review":
                    review_c += 1
                elif res.verdict == "Flagged":
                    flagged_c += 1
                else:
                    err_c += 1
            except Exception as err:
                err_c += 1
                results.append(
                    ModerationResponse(
                        filename=filename,
                        verdict="Error",
                        requires_review=True,
                        highest_category="None",
                        highest_score=0.0,
                        highest_score_percent="0.0%",
                        category_scores={},
                        category_breakdown=[],
                        model_id=model_id or self.config.active_model_id,
                        model_name="Error",
                        latency_ms=0.0,
                        explanation=f"Processing failed: {str(err)}",
                        record_id=None,
                        dimensions="N/A",
                        file_size_display="N/A",
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                )

        total_lat_ms = round((time.perf_counter() - start_batch) * 1000, 2)

        return BatchModerationResponse(
            total_images=len(files),
            safe_count=safe_c,
            review_count=review_c,
            flagged_count=flagged_c,
            error_count=err_c,
            total_latency_ms=total_lat_ms,
            results=results,
        )

    def get_models_catalog(self) -> ModelsListResponse:
        """List all models in catalog and their local offline readiness."""
        models = self.model_manager.list_all_models_status()
        items: List[ModelInfoResponse] = []
        for m in models:
            items.append(
                ModelInfoResponse(
                    id=m["id"],
                    name=m["name"],
                    description=m.get("description", ""),
                    approx_size_mb=m.get("approx_size_mb", 0),
                    installed=m.get("installed", False),
                    local_path=str(m.get("local_path", "")),
                    is_active=(m["id"] == self.config.active_model_id),
                )
            )
        return ModelsListResponse(
            active_model_id=self.config.active_model_id,
            models=items,
        )

    def download_model(self, model_id: str) -> str:
        """Download weights for offline execution."""
        return str(self.model_manager.download_model(model_id))

    def get_history(
        self,
        limit: int = 100,
        offset: int = 0,
        search_query: str = "",
        verdict_filter: str = "",
    ) -> List[ScanRecordResponse]:
        """Fetch historical moderation records."""
        records = self.history_repo.list_records(
            limit=limit,
            offset=offset,
            search_query=search_query,
            verdict_filter=verdict_filter,
        )
        return [
            ScanRecordResponse(
                id=r.id or 0,
                filename=r.filename,
                verdict=r.verdict,
                requires_review=r.requires_review,
                highest_category=r.highest_category,
                highest_score=r.highest_score,
                category_scores=r.category_scores or {},
                model_id=r.model_id,
                model_name=r.model_name,
                inference_latency_sec=r.inference_latency_sec,
                timestamp=r.timestamp,
                file_size_display=r.file_size_display,
                dimensions_display=r.dimensions_display,
                error_message=r.error_message,
            )
            for r in records
        ]

    def get_stats(self) -> StatsResponse:
        """Aggregate summary counts."""
        stats = self.history_repo.get_summary_stats()
        return StatsResponse(
            total=stats.get("total", 0),
            safe=stats.get("safe", 0),
            review=stats.get("review", 0),
            flagged=stats.get("flagged", 0),
        )

    def get_record(self, record_id: int) -> Optional[ScanRecordResponse]:
        """Fetch a single record by ID."""
        r = self.history_repo.get_record(record_id)
        if not r:
            return None
        return ScanRecordResponse(
            id=r.id or 0,
            filename=r.filename,
            verdict=r.verdict,
            requires_review=r.requires_review,
            highest_category=r.highest_category,
            highest_score=r.highest_score,
            category_scores=r.category_scores or {},
            model_id=r.model_id,
            model_name=r.model_name,
            inference_latency_sec=r.inference_latency_sec,
            timestamp=r.timestamp,
            file_size_display=r.file_size_display,
            dimensions_display=r.dimensions_display,
            error_message=r.error_message,
        )

    def delete_record(self, record_id: int) -> bool:
        """Delete an individual scan record."""
        return self.history_repo.delete_record(record_id)

    def clear_history(self) -> int:
        """Clear all historical scan records."""
        return self.history_repo.clear_all()


# Global singleton service instance
_service_instance: Optional[ModerationService] = None


def get_moderation_service() -> ModerationService:
    """Retrieve or initialize singleton ModerationService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ModerationService()
    return _service_instance
