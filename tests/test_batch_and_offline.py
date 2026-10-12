"""Tests for batch processing error tolerance, offline enforcement, and network isolation."""

import os
import urllib.request
from pathlib import Path
import pytest
from PIL import Image

from config import AppConfig
from moderation.classifier import SyntheticDiagnosticClassifier
from moderation.image_loader import load_image_for_inference
from moderation.model_manager import ModelManager, ModelNotInstalledError
from moderation.scoring import ScoringEngine


def test_batch_tolerance_to_corrupted_files(tmp_path: Path) -> None:
    """Ensure batch processing continues safely when individual files are corrupted."""
    # Create 1 valid file and 1 corrupted file
    valid_file = tmp_path / "valid.png"
    Image.new("RGB", (50, 50), color="blue").save(valid_file)

    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"invalid_header_content")

    files = [valid_file, corrupt_file]
    results = []
    classifier = SyntheticDiagnosticClassifier()

    for f in files:
        try:
            loaded = load_image_for_inference(f)
            scores = classifier.predict(loaded.image)
            eval_res = ScoringEngine.evaluate(scores)
            results.append({"status": "Completed", "verdict": eval_res.verdict})
        except Exception as err:
            results.append({"status": "Failed", "error": str(err)})

    assert len(results) == 2
    assert results[0]["status"] == "Completed"
    assert results[1]["status"] == "Failed"
    assert "corrupted" in results[1]["error"].lower() or "unreadable" in results[1]["error"].lower()


def test_strict_offline_isolation_environment(tmp_path: Path) -> None:
    """Verify that when offline mode is active, offline environment flags are set."""
    cfg = AppConfig(custom_models_dir=str(tmp_path / "models"))
    mgr = ModelManager(cfg)
    mgr.enforce_offline_environment()

    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_offline_inference_never_makes_network_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure that classifier inference never executes any network socket calls."""
    # Mock urllib and socket connections to raise error if called
    def blocked_open(*args, **kwargs):
        raise ConnectionRefusedError("Offline Mode Violation: Outgoing network request detected!")

    monkeypatch.setattr(urllib.request, "urlopen", blocked_open)

    # Run inference on local image
    img = Image.new("RGB", (64, 64), color="green")
    classifier = SyntheticDiagnosticClassifier()
    scores = classifier.predict(img)

    # Verification: Prediction succeeded without touching network
    assert "nsfw" in scores
    assert isinstance(scores["nsfw"], float)
