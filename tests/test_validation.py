"""Tests for file validation, corrupted file handling, and boundaries."""

from pathlib import Path
import pytest
from PIL import Image

from utils.validation import validate_image_file, ValidationResult


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_valid_image_formats(temp_dir: Path) -> None:
    """Test standard supported image formats (PNG, JPG, BMP, WEBP)."""
    formats = [
        ("test.png", "PNG"),
        ("test.jpg", "JPEG"),
        ("test.bmp", "BMP"),
        ("test.webp", "WEBP"),
    ]

    for fname, fmt in formats:
        p = temp_dir / fname
        img = Image.new("RGB", (100, 100), color=(73, 109, 137))
        img.save(p, format=fmt)

        res = validate_image_file(p)
        assert res.is_valid is True
        assert res.dimensions == (100, 100)
        assert res.file_size_bytes > 0
        assert res.error_message is None


def test_nonexistent_file(temp_dir: Path) -> None:
    """Test validation fails for non-existent path."""
    p = temp_dir / "does_not_exist.png"
    res = validate_image_file(p)
    assert res.is_valid is False
    assert "not found" in res.error_message.lower()


def test_unsupported_format(temp_dir: Path) -> None:
    """Test validation rejects unsupported file extensions."""
    p = temp_dir / "document.txt"
    p.write_text("Hello world", encoding="utf-8")

    res = validate_image_file(p)
    assert res.is_valid is False
    assert "unsupported" in res.error_message.lower()


def test_empty_file(temp_dir: Path) -> None:
    """Test validation rejects empty 0-byte files."""
    p = temp_dir / "empty.png"
    p.write_bytes(b"")

    res = validate_image_file(p)
    assert res.is_valid is False
    assert "empty" in res.error_message.lower()


def test_corrupted_image_file(temp_dir: Path) -> None:
    """Test validation detects corrupted image files."""
    p = temp_dir / "corrupted.jpg"
    p.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 20 + b"garbage_data_here")

    res = validate_image_file(p)
    assert res.is_valid is False
    assert "corrupted" in res.error_message.lower() or "unreadable" in res.error_message.lower()


def test_oversized_dimension_image(temp_dir: Path) -> None:
    """Test validation rejects images exceeding max dimensions."""
    p = temp_dir / "huge.png"
    img = Image.new("RGB", (200, 200), color="white")
    img.save(p, format="PNG")

    # Set max_dimension to 150
    res = validate_image_file(p, max_dimension=150)
    assert res.is_valid is False
    assert "exceeds max" in res.error_message.lower()
