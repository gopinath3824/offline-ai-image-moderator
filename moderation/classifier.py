"""Classification engines and inference adapters for offline moderation models.

Supports Falconsai NSFW baseline, CLIP zero-shot multi-category moderator,
AdamCodd fine-grained classifier, and the built-in Synthetic Diagnostic Engine.
All neural models strictly load with `local_files_only=True` to guarantee zero network traffic.
"""

from __future__ import annotations

import gc
import hashlib
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image  # type: ignore

try:
    import torch  # type: ignore
    from transformers import (  # type: ignore
        AutoImageProcessor,
        AutoModelForImageClassification,
        CLIPModel,
        CLIPProcessor,
    )
except ImportError:
    torch = None  # type: ignore
    AutoImageProcessor = None  # type: ignore
    AutoModelForImageClassification = None  # type: ignore
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore

from config import MODEL_CATALOG, AppConfig
from moderation.model_manager import ModelManager, ModelNotInstalledError
from utils.logging_config import get_logger

logger = get_logger("classifier")


def resolve_device(preference: str = "auto") -> str:
    """Resolve compute device based on preference and hardware availability."""
    pref = preference.lower().strip()
    if pref == "cpu":
        return "cpu"

    try:
        if torch is not None and torch.cuda.is_available():
            if pref in ("auto", "cuda"):
                return "cuda"
    except Exception:
        pass

    return "cpu"


class BaseModerationClassifier(ABC):
    """Abstract base class for offline moderation classifiers."""

    def __init__(self, model_id: str, model_path: Path, device: str = "cpu") -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.device = device
        self.meta = MODEL_CATALOG.get(model_id)
        self.model_name = self.meta.name if self.meta else model_id
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor from local disk."""
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, float]:
        """Perform inference on loaded PIL Image and return category confidence scores (0.0 - 1.0)."""
        pass

    def unload(self) -> None:
        """Release weights and free memory."""
        self._is_loaded = False
        gc.collect()
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class FalconsaiNSFWClassifier(BaseModerationClassifier):
    """Classifier adapter for Falconsai/nsfw_image_detection (ViT binary classifier).

    Note: Falconsai ViT outputs binary classes ('normal' vs 'nsfw') without anatomical
    region labels. We pair it with the BodyRegionDetector to evaluate anatomical
    gluteal exposure and view angles rather than arbitrarily remapping output labels.
    """

    def __init__(self, model_id: str, model_path: Path, device: str = "cpu") -> None:
        super().__init__(model_id, model_path, device)
        self.model = None
        self.processor = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise ModelNotInstalledError(
                f"Model weights not found at {self.model_path}. "
                "Please download the model via Settings or run diagnostic test mode."
            )

        if torch is None or AutoImageProcessor is None or AutoModelForImageClassification is None:
            raise RuntimeError("PyTorch and Hugging Face Transformers are required for neural model loading.")

        logger.info("Loading Falconsai NSFW model from local path: %s on %s", self.model_path, self.device)
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
        except Exception as err:
            logger.error("Failed to load Falconsai model: %s", err, exc_info=True)
            raise RuntimeError(f"Error loading offline Falconsai model: {err}") from err

    def predict(self, image: Image.Image) -> Dict[str, float]:
        if not self._is_loaded or self.model is None or self.processor is None:
            self.load()

        if torch is None:
            raise RuntimeError("PyTorch is required for inference.")

        # Preprocess PIL image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().tolist()

        # Map binary labels (0: normal, 1: nsfw)
        id2label = self.model.config.id2label
        nsfw_score = 0.0

        for idx, prob in enumerate(probs):
            label = str(id2label.get(idx, "")).lower()
            if "nsfw" in label:
                nsfw_score = float(prob)

        # Body-region anatomical analysis to isolate gluteal exposure and view angles
        from moderation.body_region_detector import BodyRegionDetector

        region = BodyRegionDetector.analyze_image(image)

        # Rule evaluation:
        # If model detects NSFW and gluteal region has exposed skin, buttocks_exposed is verified
        if region.coverage_category == "exposed":
            buttocks_score = round(max(nsfw_score, region.buttocks_exposed_score), 4)
            partial_score = round(region.partial_nudity_score, 4)
            nudity_score = round(max(nsfw_score, buttocks_score), 4)
        elif region.coverage_category == "partial_covered":
            buttocks_score = round(min(nsfw_score, region.buttocks_exposed_score), 4)
            partial_score = round(max(nsfw_score * 0.8, region.partial_nudity_score), 4)
            nudity_score = round(max(nsfw_score * 0.6, partial_score * 0.8), 4)
        elif region.coverage_category == "swimwear_covered":
            # Swimwear should not automatically be classified as nudity
            buttocks_score = round(region.buttocks_exposed_score, 4)
            partial_score = round(region.partial_nudity_score, 4)
            nudity_score = round(min(nsfw_score * 0.25, 0.20), 4)
        else:
            # Fully covered buttocks
            buttocks_score = round(region.buttocks_exposed_score, 4)
            partial_score = round(region.partial_nudity_score, 4)
            nudity_score = round(min(nsfw_score * 0.2, 0.15), 4)

        return {
            "nsfw": round(nsfw_score, 4),
            "nudity": round(nudity_score, 4),
            "buttocks_exposed": round(buttocks_score, 4),
            "partial_nudity": round(partial_score, 4),
            "suggestive": round(nsfw_score * 0.75, 4),
        }

    def unload(self) -> None:
        self.model = None
        self.processor = None
        super().unload()


class CLIPZeroShotClassifier(BaseModerationClassifier):
    """Multi-category zero-shot moderation classifier using local CLIP model.

    Evaluates multi-angle visual prompts (front, side, rear) across exposure levels:
    - Exposed buttocks (rear, side, bare skin)
    - Partial exposure (cheeky cuts, thongs, minimal fabric)
    - Fully covered clothing (rear-view jeans, skirts, trousers)
    - Swimwear / Underwear (full coverage, beachwear - distinct from nudity)
    - General explicit nudity, suggestive, violence, gore, and hate symbols.
    """

    # Curated offline reference descriptions for multi-angle, multi-category moderation
    PROMPTS: List[Tuple[str, str]] = [
        # Dedicated exposed buttocks (rear, side, and direct back views)
        ("buttocks_exposed", "a rear-view photo of a person with bare, exposed, uncovered buttocks without clothing"),
        ("buttocks_exposed", "a photograph of naked exposed buttocks seen from behind or from the side"),
        ("buttocks_exposed", "a rear-view photograph of a person completely nude from the waist down"),
        # Partial nudity (cheeky cutouts, thong, micro-swimwear)
        ("partial_nudity", "a rear-view photo of a person wearing cheeky underwear, a thong, or micro bikini bottoms with partially exposed buttocks"),
        ("partial_nudity", "a photograph of a person in revealing cutouts or partially covered buttocks"),
        # General explicit nudity
        ("nudity", "a photograph showing explicit frontal nudity, bare genitals, or sexual acts"),
        ("nudity", "a photograph of a completely nude person with bare exposed private anatomy"),
        # Suggestive / lingerie
        ("suggestive", "a photograph of a person in provocative, revealing lingerie, sensual posing, or erotic sleepwear"),
        # Covered swimwear & underwear (contrastive category: prevents false positive nudity!)
        ("swimwear_covered", "a rear-view photo of a person wearing standard full coverage swimwear, swim trunks, or bikini bottoms at a beach or pool"),
        ("swimwear_covered", "a photograph of a person wearing ordinary covered underwear or swimsuit bottoms with covered buttocks"),
        # Fully clothed (contrastive category: prevents false positive nudity!)
        ("clothed_rear", "a rear-view photo of a person wearing ordinary pants, jeans, trousers, shorts, or a skirt with fully covered buttocks"),
        ("clothed_safe", "a safe photograph of a fully clothed person walking or standing seen from behind or front"),
        # Violence, gore, hate symbols, safe
        ("violence", "a photograph of violent assault, physical combat, bleeding wounds, or brutal attack"),
        ("gore", "a photograph depicting gruesome gore, mutilation, visceral trauma, or human carnage"),
        ("hate_symbols", "a photograph depicting hate symbols, swastika, klan insignia, or extremist hate emblems"),
        ("safe", "a safe, harmless, ordinary photograph of everyday scenes, objects, landscapes, nature, or architecture"),
    ]

    def __init__(self, model_id: str, model_path: Path, device: str = "cpu") -> None:
        super().__init__(model_id, model_path, device)
        self.model = None
        self.processor = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise ModelNotInstalledError(
                f"CLIP model weights not found at {self.model_path}. "
                "Download CLIP via the Model Manager tab to enable multi-category offline detection."
            )

        if torch is None or CLIPModel is None or CLIPProcessor is None:
            raise RuntimeError("PyTorch and Hugging Face Transformers are required for CLIP inference.")

        logger.info("Loading CLIP model from %s on %s", self.model_path, self.device)
        try:
            self.processor = CLIPProcessor.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self.model = CLIPModel.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
        except Exception as err:
            logger.error("Failed to load CLIP model: %s", err, exc_info=True)
            raise RuntimeError(f"Error loading offline CLIP model: {err}") from err

    def predict(self, image: Image.Image) -> Dict[str, float]:
        if not self._is_loaded or self.model is None or self.processor is None:
            self.load()

        if torch is None:
            raise RuntimeError("PyTorch is required for inference.")

        text_labels = [p[0] for p in self.PROMPTS]
        text_prompts = [p[1] for p in self.PROMPTS]

        inputs = self.processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # shape (1, num_prompts)
            probs = torch.nn.functional.softmax(logits_per_image, dim=-1)[0].cpu().tolist()

        raw_category_probs: Dict[str, float] = {}
        for (cat, _), prob in zip(self.PROMPTS, probs):
            raw_category_probs[cat] = raw_category_probs.get(cat, 0.0) + float(prob)

        # Rule evaluation:
        # Exposed buttocks directly counts as nudity when clothing coverage is absent
        buttocks_score = raw_category_probs.get("buttocks_exposed", 0.0)
        partial_score = raw_category_probs.get("partial_nudity", 0.0)
        explicit_nudity = raw_category_probs.get("nudity", 0.0)
        nudity_score = max(explicit_nudity, buttocks_score)
        suggestive_score = raw_category_probs.get("suggestive", 0.0)
        nsfw_score = max(nudity_score, suggestive_score)

        return {
            "nsfw": round(nsfw_score, 4),
            "nudity": round(nudity_score, 4),
            "buttocks_exposed": round(buttocks_score, 4),
            "partial_nudity": round(partial_score, 4),
            "suggestive": round(suggestive_score, 4),
            "violence": round(raw_category_probs.get("violence", 0.0), 4),
            "gore": round(raw_category_probs.get("gore", 0.0), 4),
            "hate_symbols": round(raw_category_probs.get("hate_symbols", 0.0), 4),
        }

    def unload(self) -> None:
        self.model = None
        self.processor = None
        super().unload()


class AdamCoddNSFWClassifier(BaseModerationClassifier):
    """Adapter for AdamCodd/vit-base-nsfw-detector fine-grained classifier."""

    def __init__(self, model_id: str, model_path: Path, device: str = "cpu") -> None:
        super().__init__(model_id, model_path, device)
        self.model = None
        self.processor = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise ModelNotInstalledError(
                f"Model weights not found at {self.model_path}. Download via Settings."
            )

        if torch is None or AutoImageProcessor is None or AutoModelForImageClassification is None:
            raise RuntimeError("PyTorch and Hugging Face Transformers are required.")

        try:
            self.processor = AutoImageProcessor.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
        except Exception as err:
            raise RuntimeError(f"Error loading AdamCodd model: {err}") from err

    def predict(self, image: Image.Image) -> Dict[str, float]:
        if not self._is_loaded or self.model is None or self.processor is None:
            self.load()

        if torch is None:
            raise RuntimeError("PyTorch is required for inference.")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

        id2label = self.model.config.id2label
        raw_probs = {id2label[i].lower(): p for i, p in enumerate(probs)}

        porn = raw_probs.get("porn", 0.0)
        hentai = raw_probs.get("hentai", 0.0)
        sexy = raw_probs.get("sexy", 0.0)

        nudity_score = min(1.0, porn + hentai)
        suggestive_score = sexy
        nsfw_score = max(nudity_score, suggestive_score)

        return {
            "nsfw": round(nsfw_score, 4),
            "nudity": round(nudity_score, 4),
            "suggestive": round(suggestive_score, 4),
        }

    def unload(self) -> None:
        self.model = None
        self.processor = None
        super().unload()


class SyntheticDiagnosticClassifier(BaseModerationClassifier):
    """Built-in offline diagnostic engine with anatomical body-region detection.

    Performs image heuristic feature extraction:
    - Anatomical gluteal/pelvic region spatial localization (middle-lower body)
    - Viewpoint estimation (front, side, rear)
    - Skin-tone vs fabric coverage analysis (detecting jeans/clothing vs swimwear vs bare skin)
    - Red-channel injury index and deterministic visual variance.

    Enables instant offline testing, metric validation, and UI verification without neural weights.
    """

    def __init__(self, model_id: str = "synthetic_diagnostic", model_path: Path = Path("."), device: str = "cpu") -> None:
        super().__init__(model_id, model_path, device)
        self._is_loaded = True

    def load(self) -> None:
        self._is_loaded = True

    def predict(self, image: Image.Image) -> Dict[str, float]:
        # 1. Anatomical body region & view-angle analysis
        from moderation.body_region_detector import BodyRegionDetector

        region = BodyRegionDetector.analyze_image(image)

        # Fast thumbnail downsampling for global statistical extraction
        thumb = image.resize((64, 64)).convert("RGB")
        if hasattr(thumb, "get_flattened_data"):
            pixels = list(thumb.get_flattened_data())
        else:
            pixels = list(thumb.getdata())
        total_pixels = len(pixels)

        # 2. Global skin and blood-red ratios
        skin_count = 0
        blood_red_count = 0
        for r, g, b in pixels:
            if BodyRegionDetector._is_skin_pixel_rgb(r, g, b):
                skin_count += 1
            if r > 160 and g < 60 and b < 60:
                blood_red_count += 1

        skin_ratio = skin_count / total_pixels
        red_ratio = blood_red_count / total_pixels

        # 3. Deterministic hash variance
        img_bytes = thumb.tobytes()
        digest = hashlib.sha256(img_bytes).hexdigest()
        hash_val = int(digest[:8], 16) / 0xFFFFFFFF  # 0.0 - 1.0

        # Category scores incorporating detection rules:
        buttocks_exposed = region.buttocks_exposed_score
        partial_nudity = region.partial_nudity_score

        # Detection rule 2:
        # - Fully covered buttocks: classify according to existing clothing categories (safe)
        # - Buttocks exposed without clothing coverage: classify as nudity
        # - Buttocks partially covered: classify as partial_nudity
        # - Swimwear and underwear coverage: should NOT automatically be classified as nudity
        base_nudity = min(0.95, round(skin_ratio * 1.1 + (hash_val * 0.10 - 0.03), 4))
        base_nudity = max(0.01, base_nudity)

        if region.coverage_category == "fully_covered":
            # Ordinary clothing (jeans, skirts, pants) covering buttocks
            nudity = min(0.10, base_nudity * 0.15)
            buttocks_exposed = min(0.08, buttocks_exposed)
            suggestive = min(0.12, round(skin_ratio * 0.3, 4))
        elif region.coverage_category == "swimwear_covered":
            # Swimwear and underwear coverage should NOT automatically be classified as nudity
            nudity = min(0.18, base_nudity * 0.25)
            buttocks_exposed = min(0.12, buttocks_exposed)
            suggestive = min(0.35, round(0.20 + skin_ratio * 0.3, 4))
        elif region.coverage_category == "partial_covered":
            # Partial coverage (cheeky, thong, cutouts)
            nudity = round(max(0.40, partial_nudity * 0.7), 4)
            suggestive = round(max(0.55, partial_nudity * 0.85), 4)
        else:
            # Exposed buttocks without clothing coverage -> classify as nudity
            nudity = round(max(base_nudity, buttocks_exposed), 4)
            suggestive = min(0.95, round(max(0.60, buttocks_exposed * 0.8), 4))

        violence = min(0.95, round(red_ratio * 2.5 + (hash_val * 0.12 - 0.06), 4))
        violence = max(0.01, violence)

        gore = min(0.95, round(red_ratio * 2.8 + (hash_val * 0.08 - 0.04), 4))
        gore = max(0.01, gore)

        hate_symbols = min(0.95, round(hash_val * 0.08, 4))
        hate_symbols = max(0.01, hate_symbols)

        nsfw = max(nudity, suggestive, buttocks_exposed)

        return {
            "nsfw": round(nsfw, 4),
            "nudity": round(nudity, 4),
            "buttocks_exposed": round(buttocks_exposed, 4),
            "partial_nudity": round(partial_nudity, 4),
            "suggestive": round(suggestive, 4),
            "violence": round(violence, 4),
            "gore": round(gore, 4),
            "hate_symbols": round(hate_symbols, 4),
        }


class ClassifierRegistry:
    """Singleton registry and cache for loaded classifiers."""

    _active_classifier: Optional[BaseModerationClassifier] = None

    @classmethod
    def get_classifier(
        cls,
        model_id: str,
        config: Optional[AppConfig] = None,
        model_manager: Optional[ModelManager] = None,
    ) -> BaseModerationClassifier:
        """Retrieve or instantiate classifier for given model id."""
        app_config = config or AppConfig.load()
        manager = model_manager or ModelManager(app_config)
        device = resolve_device(app_config.device_preference)

        model_path = manager.get_model_path(model_id)

        # Check if already active and matches
        if (
            cls._active_classifier is not None
            and cls._active_classifier.model_id == model_id
            and cls._active_classifier.model_path == model_path
            and cls._active_classifier.device == device
        ):
            return cls._active_classifier

        # Unload previous model
        if cls._active_classifier is not None:
            logger.info("Unloading previous model: %s", cls._active_classifier.model_id)
            cls._active_classifier.unload()
            cls._active_classifier = None

        classifier: BaseModerationClassifier
        if model_id == "falconsai_nsfw":
            classifier = FalconsaiNSFWClassifier(model_id, model_path, device=device)
        elif model_id == "clip_zero_shot":
            classifier = CLIPZeroShotClassifier(model_id, model_path, device=device)
        elif model_id == "adamcodd_nsfw":
            classifier = AdamCoddNSFWClassifier(model_id, model_path, device=device)
        elif model_id == "synthetic_diagnostic":
            classifier = SyntheticDiagnosticClassifier(model_id, model_path, device=device)
        else:
            raise ValueError(f"Unknown classifier model id: '{model_id}'")

        cls._active_classifier = classifier
        return classifier

    @classmethod
    def unload_active(cls) -> None:
        """Explicitly unload current model from memory."""
        if cls._active_classifier is not None:
            cls._active_classifier.unload()
            cls._active_classifier = None
