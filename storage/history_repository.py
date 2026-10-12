"""Repository pattern for managing scan history records in SQLite."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from storage.database import get_database_connection, init_database
from utils.logging_config import get_logger

logger = get_logger("history_repository")


@dataclass
class ScanRecord:
    """Represents a persisted image moderation scan record."""

    id: Optional[int] = None
    filename: str = ""
    file_path: str = ""
    file_size_bytes: int = 0
    dimensions: str = ""
    format_name: str = ""
    verdict: str = ""  # "Likely Safe", "Needs Review", "Flagged", "Error"
    requires_review: bool = False
    highest_category: str = ""
    highest_score: float = 0.0
    category_scores: Optional[Dict[str, float]] = None
    model_id: str = ""
    model_name: str = ""
    inference_latency_sec: float = 0.0
    timestamp: str = ""
    error_message: str = ""

    def __post_init__(self) -> None:
        if self.category_scores is None:
            self.category_scores = {}
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @property
    def file_size_display(self) -> str:
        """Formatted file size string."""
        if self.file_size_bytes < 1024:
            return f"{self.file_size_bytes} B"
        elif self.file_size_bytes < 1024 * 1024:
            return f"{self.file_size_bytes / 1024:.1f} KB"
        else:
            return f"{self.file_size_bytes / (1024 * 1024):.2f} MB"

    @property
    def dimensions_display(self) -> str:
        """Formatted dimensions."""
        return self.dimensions if self.dimensions else "N/A"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary representation for UI / table display."""
        d = asdict(self)
        d["file_size_display"] = self.file_size_display
        d["dimensions_display"] = self.dimensions_display
        return d


class HistoryRepository:
    """Data access object for scan history records."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path
        init_database(self.db_path)

    def save_record(self, record: ScanRecord) -> int:
        """Insert a scan record into the database and return its new ID."""
        conn = get_database_connection(self.db_path)
        scores_json = json.dumps(record.category_scores)
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO scan_history (
                        filename, file_path, file_size_bytes, dimensions,
                        format_name, verdict, requires_review, highest_category,
                        highest_score, category_scores_json, model_id, model_name,
                        inference_latency_sec, timestamp, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.filename,
                        record.file_path,
                        record.file_size_bytes,
                        record.dimensions,
                        record.format_name,
                        record.verdict,
                        1 if record.requires_review else 0,
                        record.highest_category,
                        record.highest_score,
                        scores_json,
                        record.model_id,
                        record.model_name,
                        record.inference_latency_sec,
                        record.timestamp,
                        record.error_message,
                    ),
                )
                record.id = cursor.lastrowid
                return cursor.lastrowid or 0
        finally:
            conn.close()

    def get_record(self, record_id: int) -> Optional[ScanRecord]:
        """Retrieve a single scan record by ID."""
        conn = get_database_connection(self.db_path)
        try:
            cursor = conn.execute("SELECT * FROM scan_history WHERE id = ?", (record_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)
        finally:
            conn.close()

    def list_records(
        self,
        limit: int = 250,
        offset: int = 0,
        verdict_filter: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> List[ScanRecord]:
        """List scan records with optional verdict filtering and filename search."""
        conn = get_database_connection(self.db_path)
        try:
            query = "SELECT * FROM scan_history WHERE 1=1"
            params: List[Any] = []

            if verdict_filter and verdict_filter.lower() != "all":
                query += " AND verdict = ?"
                params.append(verdict_filter)

            if search_query and search_query.strip():
                query += " AND (filename LIKE ? OR highest_category LIKE ?)"
                term = f"%{search_query.strip()}%"
                params.extend([term, term])

            query += " ORDER BY id DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = conn.execute(query, params)
            return [self._row_to_record(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def delete_record(self, record_id: int) -> bool:
        """Delete a record by ID."""
        conn = get_database_connection(self.db_path)
        try:
            with conn:
                cursor = conn.execute("DELETE FROM scan_history WHERE id = ?", (record_id,))
                return cursor.rowcount > 0
        finally:
            conn.close()

    def clear_all(self) -> int:
        """Delete all scan history records from the database."""
        conn = get_database_connection(self.db_path)
        try:
            with conn:
                cursor = conn.execute("DELETE FROM scan_history")
                return cursor.rowcount
        finally:
            conn.close()

    def get_summary_stats(self) -> Dict[str, int]:
        """Return counts of total, safe, flagged, and review scans."""
        conn = get_database_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN verdict = 'Flagged' THEN 1 ELSE 0 END) as flagged,
                    SUM(CASE WHEN verdict = 'Needs Review' THEN 1 ELSE 0 END) as review,
                    SUM(CASE WHEN verdict = 'Likely Safe' THEN 1 ELSE 0 END) as safe,
                    SUM(CASE WHEN verdict = 'Error' THEN 1 ELSE 0 END) as error
                FROM scan_history
                """
            )
            row = cursor.fetchone()
            if not row:
                return {"total": 0, "flagged": 0, "review": 0, "safe": 0, "error": 0}
            return {
                "total": row["total"] or 0,
                "flagged": row["flagged"] or 0,
                "review": row["review"] or 0,
                "safe": row["safe"] or 0,
                "error": row["error"] or 0,
            }
        finally:
            conn.close()

    def get_recent_activity(self, limit: int = 10) -> List[ScanRecord]:
        """Retrieve the most recent scans for the dashboard."""
        return self.list_records(limit=limit, offset=0)

    @staticmethod
    def _row_to_record(row: Any) -> ScanRecord:
        """Deserialize a database row into a ScanRecord object."""
        try:
            cat_scores = json.loads(row["category_scores_json"])
        except Exception:
            cat_scores = {}

        return ScanRecord(
            id=row["id"],
            filename=row["filename"],
            file_path=row["file_path"],
            file_size_bytes=row["file_size_bytes"] or 0,
            dimensions=row["dimensions"] or "",
            format_name=row["format_name"] or "",
            verdict=row["verdict"],
            requires_review=bool(row["requires_review"]),
            highest_category=row["highest_category"] or "None",
            highest_score=float(row["highest_score"] or 0.0),
            category_scores=cat_scores,
            model_id=row["model_id"],
            model_name=row["model_name"],
            inference_latency_sec=float(row["inference_latency_sec"] or 0.0),
            timestamp=row["timestamp"],
            error_message=row["error_message"] or "",
        )
