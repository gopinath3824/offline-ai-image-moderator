"""Model manager for offline AI image moderation models.

Handles local directory storage, offline-only enforcement (HF_HUB_OFFLINE, local_files_only),
model installation status, downloading with progress callbacks, and disk cleanup.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional

from config import MODEL_CATALOG, AppConfig, ModelMetadata
from utils.logging_config import get_logger

logger = get_logger("model_manager")


class ModelNotInstalledError(FileNotFoundError):
    """Raised when an operation requires an offline model that is not yet downloaded."""

    pass


class ModelManager:
    """Manages local storage, status verification, and downloads for moderation models."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.load()
        self.models_dir = self.config.get_models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def enforce_offline_environment(self) -> None:
        """Set environment flags to prevent Hugging Face Hub from attempting network connections."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info("Enforced strict offline environment: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1")

    def relax_offline_environment_for_download(self) -> None:
        """Temporarily clear offline environment flags during explicit user-initiated downloads."""
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        logger.info("Relaxed offline flags for user-requested model download.")

    def get_model_path(self, model_id: str) -> Path:
        """Return the local directory path where the model weights are stored."""
        return self.models_dir / model_id

    def is_model_installed(self, model_id: str) -> bool:
        """Check whether the model weights and configuration are installed locally."""
        if model_id == "synthetic_diagnostic":
            return True  # Built-in offline diagnostic engine is always ready

        model_path = self.get_model_path(model_id)
        if not model_path.exists() or not model_path.is_dir():
            return False

        # Check for essential configuration and weight markers
        has_config = (model_path / "config.json").exists()
        has_weights = (
            (model_path / "model.safetensors").exists()
            or (model_path / "pytorch_model.bin").exists()
            or any(model_path.glob("*.safetensors"))
            or any(model_path.glob("*.bin"))
        )
        return has_config and has_weights

    def get_installed_size_bytes(self, model_id: str) -> int:
        """Calculate total disk space in bytes consumed by a local model."""
        model_path = self.get_model_path(model_id)
        if not model_path.exists():
            return 0

        total_bytes = 0
        for entry in model_path.rglob("*"):
            if entry.is_file():
                try:
                    total_bytes += entry.stat().st_size
                except OSError:
                    pass
        return total_bytes

    def get_installed_size_display(self, model_id: str) -> str:
        """Human-readable string of local model disk usage."""
        size_bytes = self.get_installed_size_bytes(model_id)
        if size_bytes == 0:
            return "Not Installed"
        return f"{size_bytes / (1024 * 1024):.1f} MB"

    def list_all_models_status(self) -> List[Dict[str, Any]]:
        """Return status, metadata, and local paths for all models in catalog."""
        result = []
        for model_id, meta in MODEL_CATALOG.items():
            installed = self.is_model_installed(model_id)
            result.append(
                {
                    "id": model_id,
                    "name": meta.name,
                    "repo_id": meta.repo_id,
                    "type": meta.model_type,
                    "description": meta.description,
                    "categories": meta.supported_categories,
                    "approx_size_mb": meta.size_mb,
                    "installed": installed,
                    "installed_size_display": self.get_installed_size_display(model_id),
                    "local_path": str(self.get_model_path(model_id)),
                    "license": meta.license,
                }
            )
        return result

    def download_model(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """Explicitly download a model from Hugging Face Hub to the local directory.

        Must ONLY be called when user triggers download or during explicit setup.
        """
        if model_id not in MODEL_CATALOG:
            raise ValueError(f"Unknown model identifier: '{model_id}'")

        meta = MODEL_CATALOG[model_id]
        if meta.model_type == "synthetic":
            if progress_callback:
                progress_callback(1.0, "Offline Diagnostic Engine is already built-in and ready.")
            return self.get_model_path(model_id)

        target_dir = self.get_model_path(model_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.relax_offline_environment_for_download()
            if progress_callback:
                progress_callback(0.1, f"Connecting to Hugging Face Hub for {meta.name}...")

            # Use huggingface_hub snapshot_download
            from huggingface_hub import snapshot_download  # type: ignore

            if progress_callback:
                progress_callback(0.3, f"Downloading model files to local storage ({meta.size_mb} MB)...")

            downloaded_folder = snapshot_download(
                repo_id=meta.repo_id,
                local_dir=str(target_dir),
                revision=meta.default_revision,
                local_dir_use_symlinks=False,
            )

            if progress_callback:
                progress_callback(0.9, "Verifying downloaded model files...")

            if not self.is_model_installed(model_id):
                raise RuntimeError(
                    f"Download completed but configuration or weights verification failed in {target_dir}"
                )

            if progress_callback:
                progress_callback(1.0, f"Successfully installed {meta.name} for offline use.")

            logger.info("Model '%s' downloaded and validated at %s", model_id, target_dir)
            return Path(downloaded_folder)

        except Exception as err:
            logger.error("Failed to download model '%s': %s", model_id, err, exc_info=True)
            raise RuntimeError(
                f"Failed to download model '{meta.name}': {str(err)}. "
                "Verify your internet connection and try again."
            ) from err

        finally:
            if self.config.offline_enforce_strictly:
                self.enforce_offline_environment()

    def delete_local_model(self, model_id: str) -> bool:
        """Remove local model directory to free up disk space."""
        if model_id == "synthetic_diagnostic":
            return False  # Built-in synthetic model has no directory to delete

        target_dir = self.get_model_path(model_id)
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
            logger.info("Removed local weights for model '%s' at %s", model_id, target_dir)
            return True
        return False
