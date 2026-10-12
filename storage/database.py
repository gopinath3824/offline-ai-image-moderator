"""SQLite database connection and schema management."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from config import DEFAULT_DB_PATH
from utils.logging_config import get_logger

logger = get_logger("database")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scan_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER DEFAULT 0,
    dimensions TEXT DEFAULT '',
    format_name TEXT DEFAULT '',
    verdict TEXT NOT NULL,
    requires_review INTEGER DEFAULT 0,
    highest_category TEXT DEFAULT '',
    highest_score REAL DEFAULT 0.0,
    category_scores_json TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    inference_latency_sec REAL DEFAULT 0.0,
    timestamp TEXT NOT NULL,
    error_message TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_scan_history_timestamp ON scan_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scan_history_verdict ON scan_history(verdict);
CREATE INDEX IF NOT EXISTS idx_scan_history_filename ON scan_history(filename);
"""


def get_database_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a configured sqlite3 connection.

    Enables WAL mode for safe concurrent access from background inference workers.
    """
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception as err:
        logger.debug("Failed to set PRAGMA on sqlite db: %s", err)
    return conn


def init_database(db_path: Optional[Path] = None) -> None:
    """Initialize database tables and indices if they do not exist."""
    conn = get_database_connection(db_path)
    try:
        with conn:
            conn.executescript(SCHEMA_SQL)
        logger.info("Database schema initialized successfully at %s", db_path or DEFAULT_DB_PATH)
    finally:
        conn.close()
