"""Configuration and application settings for Offline AI Image Moderator.

Handles application paths, persistent user settings, threshold configurations,
and catalog of supported offline moderation models.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

APP_NAME = "Offline AI Image Moderator"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "100% Local, Privacy-First Desktop Image Content Moderation"

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Default application storage directory in user home
DEFAULT_APP_DIR = Path.home() / ".offline_image_moderator"
DEFAULT_MODELS_DIR = DEFAULT_APP_DIR / "models"
DEFAULT_DB_PATH = DEFAULT_APP_DIR / "history.db"
DEFAULT_CONFIG_PATH = DEFAULT_APP_DIR / "config.json"
DEFAULT_LOGS_DIR = DEFAULT_APP_DIR / "logs"

# Available moderation categories
CATEGORY_DISPLAY_NAMES = {
    "nsfw": "NSFW Content",
    "nudity": "Nudity & Sexually Explicit",
    "buttocks_exposed": "Exposed Buttocks",
    "partial_nudity": "Partial Nudity",
    "suggestive": "Suggestive / Erotic",
    "violence": "Violence & Physical Harm",
    "gore": "Gore & Disturbing Imagery",
    "hate_symbols": "Hate Symbols & Extremism",
}

CATEGORY_DESCRIPTIONS = {
    "nsfw": "General explicit or adult imagery detected by baseline models.",
    "nudity": "Sexually explicit content, exposed anatomy, or sexual acts.",
    "buttocks_exposed": "Bare, exposed buttocks without clothing coverage across rear, side, or front angles.",
    "partial_nudity": "Partially exposed anatomy, revealing swimwear/cutouts, cheeky bottoms, or micro-coverage.",
    "suggestive": "Sexually themed, provocative poses, or revealing attire.",
    "violence": "Graphic physical assault, weapons in aggressive use, or bodily trauma.",
    "gore": "Mutilation, open wounds, severe injury, or gruesome imagery.",
    "hate_symbols": "Known hate group iconography, extremist badges, or hateful visual insignia.",
}


@dataclass
class ModelMetadata:
    """Metadata specification for supported moderation models."""

    id: str
    name: str
    repo_id: str
    model_type: str  # "binary_nsfw", "clip_zero_shot", "multi_class_nsfw", "synthetic"
    description: str
    supported_categories: List[str]
    size_mb: int
    license: str
    default_revision: str = "main"


# Catalog of recognized models
MODEL_CATALOG: Dict[str, ModelMetadata] = {
    "falconsai_nsfw": ModelMetadata(
        id="falconsai_nsfw",
        name="Falconsai NSFW Classifier (Baseline)",
        repo_id="Falconsai/nsfw_image_detection",
        model_type="binary_nsfw",
        description=(
            "Vision Transformer baseline for binary normal vs NSFW classification. "
            "Note: Binary classifier (normal vs nsfw); paired with body-region detection "
            "for anatomical buttocks exposure analysis."
        ),
        supported_categories=["nsfw", "nudity", "buttocks_exposed", "partial_nudity"],
        size_mb=340,
        license="Apache 2.0",
    ),
    "clip_zero_shot": ModelMetadata(
        id="clip_zero_shot",
        name="CLIP Multi-Category Zero-Shot Engine",
        repo_id="openai/clip-vit-base-patch32",
        model_type="clip_zero_shot",
        description=(
            "Comprehensive zero-shot visual matcher covering Nudity, Exposed Buttocks, "
            "Partial Nudity, Suggestive, Violence, Gore, and Hate Symbols entirely offline via multi-angle text prompts."
        ),
        supported_categories=[
            "nudity",
            "buttocks_exposed",
            "partial_nudity",
            "suggestive",
            "violence",
            "gore",
            "hate_symbols",
        ],
        size_mb=605,
        license="MIT",
    ),
    "adamcodd_nsfw": ModelMetadata(
        id="adamcodd_nsfw",
        name="AdamCodd Fine-Grained NSFW Classifier",
        repo_id="AdamCodd/vit-base-nsfw-detector",
        model_type="multi_class_nsfw",
        description=(
            "Fine-grained multi-class detector categorizing neutral, drawings, "
            "hentai, sexy, and pornographic content."
        ),
        supported_categories=["nudity", "suggestive"],
        size_mb=343,
        license="MIT",
    ),
    "synthetic_diagnostic": ModelMetadata(
        id="synthetic_diagnostic",
        name="Offline Diagnostic Engine (Test Mode)",
        repo_id="offline/diagnostic-engine",
        model_type="synthetic",
        description=(
            "Local simulated engine with spatial body-region detection for offline testing, UI verification, "
            "and multi-angle buttocks exposure analysis without downloading large neural weight files."
        ),
        supported_categories=[
            "nsfw",
            "nudity",
            "buttocks_exposed",
            "partial_nudity",
            "suggestive",
            "violence",
            "gore",
            "hate_symbols",
        ],
        size_mb=1,
        license="Internal / Open Source",
    ),
}


@dataclass
class AppConfig:
    """User-configurable application settings with persistent JSON storage."""

    flag_threshold: float = 0.70
    review_threshold: float = 0.40
    enabled_categories: List[str] = field(
        default_factory=lambda: [
            "nsfw",
            "nudity",
            "buttocks_exposed",
            "partial_nudity",
            "suggestive",
            "violence",
            "gore",
            "hate_symbols",
        ]
    )
    active_model_id: str = "falconsai_nsfw"
    device_preference: str = "cpu"  # "auto", "cpu", "cuda"
    enable_history: bool = True
    max_file_size_mb: int = 35
    max_image_dimension: int = 5000
    theme_mode: str = "dark"  # "dark", "light", "system"
    custom_models_dir: str = ""
    offline_enforce_strictly: bool = True

    def get_models_dir(self) -> Path:
        """Return the resolved path to local models storage directory."""
        if self.custom_models_dir:
            p = Path(self.custom_models_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p
        DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return DEFAULT_MODELS_DIR

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppConfig:
        """Create AppConfig from dictionary with safe defaults for unknown/missing keys."""
        valid_keys = cls.__dataclass_fields__.keys()  # type: ignore
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, config_path: Path | None = None) -> None:
        """Persist settings to disk in JSON format."""
        target_path = config_path or DEFAULT_CONFIG_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, config_path: Path | None = None) -> AppConfig:
        """Load settings from disk or return defaults if absent/corrupt."""
        target_path = config_path or DEFAULT_CONFIG_PATH
        if not target_path.exists():
            config = cls()
            try:
                config.save(target_path)
            except Exception:
                pass
            return config

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception:
            # Fallback to default if configuration corrupted
            return cls()


def ensure_directories_exist() -> None:
    """Ensure standard storage and log folders exist."""
    DEFAULT_APP_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
