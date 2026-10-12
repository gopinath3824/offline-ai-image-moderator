"""File and image validation utilities for Offline AI Image Moderator.

Safely checks existence, formats, dimensions, sizes, and file corruption
before attempting inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image  # type: ignore

from config import SUPPORTED_EXTENSIONS
from utils.logging_config import get_logger

logger = get_logger("validation")


@dataclass
class ValidationResult:
    """Structured result of image validation."""

    is_valid: bool
    error_message: Optional[str] = None
    file_path: Optional[Path] = None
    file_size_bytes: int = 0
    dimensions: Tuple[int, int] = (0, 0)
    format_name: str = ""
    color_mode: str = ""
    has_transparency: bool = False

    @property
    def file_size_display(self) -> str:
        """Human-readable file size string."""
        if self.file_size_bytes < 1024:
            return f"{self.file_size_bytes} B"
        elif self.file_size_bytes < 1024 * 1024:
            return f"{self.file_size_bytes / 1024:.1f} KB"
        else:
            return f"{self.file_size_bytes / (1024 * 1024):.2f} MB"

    @property
    def dimensions_display(self) -> str:
        """Human-readable dimensions string."""
        if self.dimensions == (0, 0):
            return "Unknown"
        return f"{self.dimensions[0]} x {self.dimensions[1]} px"


def validate_image_file(
    file_path: str | Path,
    max_size_mb: int = 35,
    max_dimension: int = 5000,
) -> ValidationResult:
    """Validate that the file exists, has a supported format, is not corrupted,

    and adheres to size and dimension boundaries.
    """
    path = Path(file_path).resolve()

    # 1. Check file existence
    if not path.exists():
        return ValidationResult(
            is_valid=False,
            error_message=f"File not found: {path.name}",
            file_path=path,
        )

    if not path.is_file():
        return ValidationResult(
            is_valid=False,
            error_message=f"Target is not a regular file: {path.name}",
            file_path=path,
        )

    # 2. Check extension
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
            file_path=path,
        )

    # 3. Check file size
    try:
        size_bytes = os.path.getsize(path)
    except OSError as err:
        return ValidationResult(
            is_valid=False,
            error_message=f"Unable to read file size: {err}",
            file_path=path,
        )

    if size_bytes == 0:
        return ValidationResult(
            is_valid=False,
            error_message="File is empty (0 bytes).",
            file_path=path,
            file_size_bytes=0,
        )

    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"File size ({size_bytes / (1024 * 1024):.1f} MB) exceeds limit of {max_size_mb} MB."
            ),
            file_path=path,
            file_size_bytes=size_bytes,
        )

    # 4. Open and inspect image using Pillow
    try:
        with Image.open(path) as img:
            format_name = img.format or ext.lstrip(".").upper()
            color_mode = img.mode
            dimensions = (img.width, img.height)
            has_transparency = "A" in color_mode or color_mode in ("RGBA", "LA", "PA")

            # Check dimensions limit
            if img.width > max_dimension or img.height > max_dimension:
                return ValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Image resolution ({img.width}x{img.height}) exceeds max {max_dimension}px."
                    ),
                    file_path=path,
                    file_size_bytes=size_bytes,
                    dimensions=dimensions,
                    format_name=format_name,
                    color_mode=color_mode,
                    has_transparency=has_transparency,
                )

            # Test image integrity
            img.verify()

        # Re-open briefly to ensure decoding succeed (verify() can invalidate img object)
        with Image.open(path) as img:
            img.draft(None, (64, 64))

        return ValidationResult(
            is_valid=True,
            error_message=None,
            file_path=path,
            file_size_bytes=size_bytes,
            dimensions=dimensions,
            format_name=format_name,
            color_mode=color_mode,
            has_transparency=has_transparency,
        )

    except Exception as err:
        logger.warning("Image validation failed for %s: %s", path.name, err)
        return ValidationResult(
            is_valid=False,
            error_message=f"Image file is corrupted or unreadable: {str(err)}",
            file_path=path,
            file_size_bytes=size_bytes,
        )
