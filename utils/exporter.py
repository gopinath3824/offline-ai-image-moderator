"""CSV export utilities for batch results and scan history."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence


def export_batch_results_to_csv(
    results: Sequence[Dict[str, Any]],
    destination_path: str | Path,
) -> Path:
    """Export batch moderation results to CSV format.

    Each row includes: filename, file_path, status, verdict, highest_category,
    top_score, individual category scores, model_name, inference_time_sec, error.
    """
    dest = Path(destination_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Collect all unique category names present in results
    all_categories = set()
    for item in results:
        scores = item.get("category_scores", {})
        if isinstance(scores, dict):
            all_categories.update(scores.keys())
    sorted_categories = sorted(all_categories)

    fieldnames = [
        "filename",
        "verdict",
        "requires_review",
        "highest_category",
        "highest_score_pct",
        "model_name",
        "file_size",
        "dimensions",
        "inference_sec",
        "timestamp",
        "error_message",
    ]
    # Append individual category score columns
    for cat in sorted_categories:
        fieldnames.append(f"score_{cat}_pct")
    fieldnames.append("full_file_path")

    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            cat_scores = r.get("category_scores", {}) or {}
            top_score = r.get("highest_score", 0.0)
            top_score_pct = f"{top_score * 100:.1f}%" if top_score is not None else "0.0%"

            row: Dict[str, Any] = {
                "filename": r.get("filename", ""),
                "verdict": r.get("verdict", ""),
                "requires_review": "Yes" if r.get("requires_review") else "No",
                "highest_category": r.get("highest_category", "N/A"),
                "highest_score_pct": top_score_pct,
                "model_name": r.get("model_name", ""),
                "file_size": r.get("file_size_display", ""),
                "dimensions": r.get("dimensions_display", ""),
                "inference_sec": f"{r.get('inference_latency_sec', 0.0):.2f}",
                "timestamp": r.get("timestamp", datetime.now().isoformat()),
                "error_message": r.get("error_message", "") or "",
                "full_file_path": r.get("file_path", ""),
            }

            for cat in sorted_categories:
                val = cat_scores.get(cat)
                row[f"score_{cat}_pct"] = f"{val * 100:.1f}%" if val is not None else ""

            writer.writerow(row)

    return dest


def export_history_to_csv(
    records: Sequence[Dict[str, Any]],
    destination_path: str | Path,
) -> Path:
    """Export database scan history records to CSV."""
    return export_batch_results_to_csv(records, destination_path)
