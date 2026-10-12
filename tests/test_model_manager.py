"""Tests for ModelManager, offline enforcement, and model loading."""

import os
from pathlib import Path
import pytest
from PIL import Image

from config import AppConfig
from moderation.classifier import (
    ClassifierRegistry,
    SyntheticDiagnosticClassifier,
)
from moderation.model_manager import ModelManager, ModelNotInstalledError


@pytest.fixture
def mock_config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(custom_models_dir=str(tmp_path / "models"))
    return cfg


def test_missing_model_detection(mock_config: AppConfig) -> None:
    """Ensure uninstalled models are detected and raise ModelNotInstalledError."""
    mgr = ModelManager(mock_config)
    assert mgr.is_model_installed("falconsai_nsfw") is False
    assert mgr.is_model_installed("clip_zero_shot") is False
    assert mgr.is_model_installed("synthetic_diagnostic") is True


def test_offline_environment_flags(mock_config: AppConfig) -> None:
    """Verify offline environment variables are strictly applied."""
    mgr = ModelManager(mock_config)
    mgr.enforce_offline_environment()

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_missing_model_raises_clean_error(mock_config: AppConfig) -> None:
    """Attempting to load an uninstalled neural model raises ModelNotInstalledError."""
    ClassifierRegistry.unload_active()
    mgr = ModelManager(mock_config)
    classifier = ClassifierRegistry.get_classifier(
        "falconsai_nsfw", config=mock_config, model_manager=mgr
    )

    with pytest.raises(ModelNotInstalledError):
        classifier.load()


def test_synthetic_diagnostic_offline_classifier() -> None:
    """Diagnostic classifier operates completely offline without network or files."""
    classifier = SyntheticDiagnosticClassifier()
    img = Image.new("RGB", (120, 120), color=(200, 140, 100))

    scores = classifier.predict(img)
    assert isinstance(scores, dict)
    assert "nsfw" in scores
    assert "nudity" in scores
    assert "violence" in scores
    assert "gore" in scores
    assert "hate_symbols" in scores

    for cat, score in scores.items():
        assert 0.0 <= score <= 1.0


def test_falconsai_real_installed_offline_inference() -> None:
    """If falconsai_nsfw is installed locally, verify real neural loading and prediction."""
    default_config = AppConfig.load()
    mgr = ModelManager(default_config)
    if not mgr.is_model_installed("falconsai_nsfw"):
        pytest.skip("Falconsai model not downloaded locally")

    mgr.enforce_offline_environment()
    classifier = ClassifierRegistry.get_classifier(
        "falconsai_nsfw", config=default_config, model_manager=mgr
    )
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    scores = classifier.predict(img)
    assert "nsfw" in scores
    assert 0.0 <= scores["nsfw"] <= 1.0
    assert classifier.is_loaded is True

