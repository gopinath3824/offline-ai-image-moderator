"""Tests for image loader, transparency handling, and immutability."""

import hashlib
from pathlib import Path
import pytest
from PIL import Image

from moderation.image_loader import (
    create_preview_thumbnail,
    load_image_for_inference,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_transparency_handling(temp_dir: Path) -> None:
    """Ensure RGBA image with alpha channel is composited to 3-channel RGB."""
    img_path = temp_dir / "transparent.png"
    # Create RGBA image with transparent pixels
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    img.save(img_path, format="PNG")

    loaded = load_image_for_inference(img_path)
    assert loaded.image.mode == "RGB"
    assert loaded.has_transparency is True
    assert loaded.dimensions == (100, 100)


def test_original_file_immutability(temp_dir: Path) -> None:
    """Ensure image loading NEVER mutates or overwrites the original file."""
    img_path = temp_dir / "immutable.jpg"
    img = Image.new("RGB", (80, 80), color=(100, 150, 200))
    img.save(img_path, format="JPEG")

    original_bytes = img_path.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()

    # Load image for inference
    loaded = load_image_for_inference(img_path)
    assert loaded.image is not None

    after_bytes = img_path.read_bytes()
    after_hash = hashlib.sha256(after_bytes).hexdigest()

    assert original_hash == after_hash, "Original image on disk was mutated!"


def test_thumbnail_creation() -> None:
    """Verify thumbnail creation scales while keeping aspect ratio."""
    img = Image.new("RGB", (800, 400), color=(50, 50, 50))
    thumb = create_preview_thumbnail(img, max_size=(200, 200))

    assert thumb.width <= 200
    assert thumb.height <= 200
    # Aspect ratio was 2:1, so width should be 200, height 100
    assert thumb.width == 200
    assert thumb.height == 100
