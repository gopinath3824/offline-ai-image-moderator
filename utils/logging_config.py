"""Logging configuration for Offline AI Image Moderator.

Ensures structured, rotating logging without logging sensitive image data or personal info.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config import DEFAULT_LOGS_DIR


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
) -> None:
    """Initialize application logging with rotating file handler and console output.

    Does not log image pixel content, EXIF personal details, or external network data.
    """
    target_dir = log_dir or DEFAULT_LOGS_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        log_file = target_dir / "app.log"
    except Exception:
        target_dir = Path.cwd() / "logs"
        target_dir.mkdir(parents=True, exist_ok=True)
        log_file = target_dir / "app.log"

    root_logger = logging.getLogger("offline_moderator")
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup is called multiple times
    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Rotating file handler
    try:
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    except Exception as err:
        root_logger.warning("Failed to initialize rotating file handler: %s", err)


def get_logger(name: str) -> logging.Logger:
    """Return a scoped logger under the application root."""
    return logging.getLogger(f"offline_moderator.{name}")
