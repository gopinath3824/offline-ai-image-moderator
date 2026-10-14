"""Safe image loading, EXIF orientation correction, and transparency handling.

Ensures that the original image is NEVER modified, overwritten, or deleted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps  # type: ignore

from utils.logging_config import get_logger
from utils.validation import ValidationResult, validate_image_file

logger = get_logger("image_loader")


@dataclass
class LoadedImageData:
    """Container for safely loaded PIL Image and metadata."""

    image: Image.Image
    file_path: Path
    filename: str
    dimensions: Tuple[int, int]
    file_size_bytes: int
    format_name: str
    color_mode: str
    has_transparency: bool

    @property
    def file_size_display(self) -> str:
        """Human-readable file size."""
        if self.file_size_bytes < 1024:
            return f"{self.file_size_bytes} B"
        elif self.file_size_bytes < 1024 * 1024:
            return f"{self.file_size_bytes / 1024:.1f} KB"
        else:
            return f"{self.file_size_bytes / (1024 * 1024):.2f} MB"

    @property
    def dimensions_display(self) -> str:
        """Human-readable dimensions."""
        return f"{self.dimensions[0]} x {self.dimensions[1]} px"


def load_image_for_inference(
    file_path: str | Path,
    max_size_mb: int = 35,
    max_dimension: int = 5000,
) -> LoadedImageData:
    """Safely loads an image for AI model inference.

    - Validates file existence and integrity.
    - Applies EXIF orientation transpose so rotated camera photos display and classify accurately.
    - Composites transparency over an RGB white background to prevent model artifacting.
    - Strictly preserves the original file without any disk mutations.
    """
    path = Path(file_path).resolve()

    val: ValidationResult = validate_image_file(
        path, max_size_mb=max_size_mb, max_dimension=max_dimension
    )
    if not val.is_valid:
        raise ValueError(val.error_message or f"Invalid image file: {path.name}")

    try:
        # Open image in read-only mode
        with Image.open(path) as raw_img:
            # 1. Correct orientation using EXIF metadata if present
            try:
                oriented_img = ImageOps.exif_transpose(raw_img)
            except Exception as exif_err:
                logger.debug("EXIF transpose skipped for %s: %s", path.name, exif_err)
                oriented_img = raw_img

            if oriented_img is None:
                oriented_img = raw_img

            orig_w, orig_h = oriented_img.size
            orig_mode = raw_img.mode
            orig_format = raw_img.format or path.suffix.lstrip(".").upper()
            has_transparency = "A" in orig_mode or orig_mode in ("RGBA", "LA", "PA")

            # 2. Handle transparency and normalize to 3-channel RGB
            if oriented_img.mode in ("RGBA", "LA") or (
                oriented_img.mode == "P" and "transparency" in oriented_img.info
            ):
                rgba_img = oriented_img.convert("RGBA")
                # Composite onto neutral white background
                rgb_canvas = Image.new("RGB", rgba_img.size, (255, 255, 255))
                alpha_channel = rgba_img.split()[3]
                rgb_canvas.paste(rgba_img, mask=alpha_channel)
                final_img = rgb_canvas
            elif oriented_img.mode != "RGB":
                final_img = oriented_img.convert("RGB")
            else:
                final_img = oriented_img.copy()

        return LoadedImageData(
            image=final_img,
            file_path=path,
            filename=path.name,
            dimensions=(orig_w, orig_h),
            file_size_bytes=val.file_size_bytes,
            format_name=orig_format,
            color_mode=orig_mode,
            has_transparency=has_transparency,
        )

    except Exception as err:
        logger.error("Failed to load image %s: %s", path.name, err, exc_info=True)
        raise RuntimeError(f"Failed to process image '{path.name}': {str(err)}") from err


def create_preview_thumbnail(
    image: Image.Image,
    max_size: Tuple[int, int] = (480, 480),
) -> Image.Image:
    """Create a scaled preview copy of the PIL Image maintaining aspect ratio."""
    thumb = image.copy()
    thumb.thumbnail(max_size, Image.Resampling.LANCZOS)
    return thumb


def load_image_from_bytes(
    data: bytes,
    filename: str = "upload.jpg",
    max_size_mb: int = 35,
    max_dimension: int = 5000,
) -> LoadedImageData:
    """Safely loads an in-memory image byte stream for AI model inference.

    - Validates byte size and dimensions.
    - Applies EXIF orientation transpose.
    - Composites transparency over an RGB white background.
    - Normalizes to 3-channel RGB.
    """
    import io

    size_bytes = len(data)
    if size_bytes == 0:
        raise ValueError("Image data is empty (0 bytes)")
    if size_bytes > max_size_mb * 1024 * 1024:
        raise ValueError(f"Image exceeds maximum allowed size of {max_size_mb} MB")

    try:
        with Image.open(io.BytesIO(data)) as raw_img:
            w, h = raw_img.size
            if w > max_dimension or h > max_dimension:
                raise ValueError(
                    f"Image dimensions ({w}x{h}) exceed maximum allowed dimension ({max_dimension}px)"
                )

            try:
                oriented_img = ImageOps.exif_transpose(raw_img)
            except Exception as exif_err:
                logger.debug("EXIF transpose skipped for %s: %s", filename, exif_err)
                oriented_img = raw_img

            if oriented_img is None:
                oriented_img = raw_img

            orig_w, orig_h = oriented_img.size
            orig_mode = raw_img.mode
            orig_format = raw_img.format or Path(filename).suffix.lstrip(".").upper() or "JPEG"
            has_transparency = "A" in orig_mode or orig_mode in ("RGBA", "LA", "PA")

            if oriented_img.mode in ("RGBA", "LA") or (
                oriented_img.mode == "P" and "transparency" in oriented_img.info
            ):
                rgba_img = oriented_img.convert("RGBA")
                rgb_canvas = Image.new("RGB", rgba_img.size, (255, 255, 255))
                alpha_channel = rgba_img.split()[3]
                rgb_canvas.paste(rgba_img, mask=alpha_channel)
                final_img = rgb_canvas
            elif oriented_img.mode != "RGB":
                final_img = oriented_img.convert("RGB")
            else:
                final_img = oriented_img.copy()

        return LoadedImageData(
            image=final_img,
            file_path=Path(filename),
            filename=filename,
            dimensions=(orig_w, orig_h),
            file_size_bytes=size_bytes,
            format_name=orig_format,
            color_mode=orig_mode,
            has_transparency=has_transparency,
        )

    except Exception as err:
        logger.error("Failed to load image from bytes for %s: %s", filename, err, exc_info=True)
        raise RuntimeError(f"Failed to process image '{filename}': {str(err)}") from err
