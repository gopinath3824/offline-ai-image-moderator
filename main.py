"""Main entry point for Offline AI Image Moderator.

Supports launching the graphical desktop application as well as CLI utility commands
for headless image scanning and explicit model downloads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import APP_NAME, APP_VERSION, AppConfig, ensure_directories_exist
from moderation.classifier import ClassifierRegistry
from moderation.image_loader import load_image_for_inference
from moderation.model_manager import ModelManager
from moderation.scoring import ScoringEngine
from storage.database import init_database
from utils.logging_config import get_logger, setup_logging

logger = get_logger("main")


def run_cli_moderation(image_path: str, model_id: str, flag_th: float, rev_th: float) -> int:
    """Run headless image moderation on an image file from command line."""
    config = AppConfig.load()
    manager = ModelManager(config)
    manager.enforce_offline_environment()

    print(f"\n=======================================================")
    print(f"  {APP_NAME} v{APP_VERSION} (CLI Mode)")
    print(f"=======================================================")
    print(f"Target Image:    {image_path}")
    print(f"Model ID:        {model_id}")
    print(f"Flag Threshold:  {flag_th * 100:.0f}%")
    print(f"Review Margins:  [{rev_th * 100:.0f}%, {flag_th * 100:.0f}%)")
    print("-------------------------------------------------------")

    try:
        loaded = load_image_for_inference(image_path)
        print(f"Resolution:      {loaded.dimensions_display}")
        print(f"File Size:       {loaded.file_size_display} ({loaded.format_name})")

        classifier = ClassifierRegistry.get_classifier(
            model_id=model_id,
            config=config,
            model_manager=manager,
        )

        scores = classifier.predict(loaded.image)
        result = ScoringEngine.evaluate(
            category_scores=scores,
            flag_threshold=flag_th,
            review_threshold=rev_th,
            enabled_categories=config.enabled_categories,
            model_id=model_id,
            model_name=classifier.model_name,
        )

        print("\nCATEGORY BREAKDOWN:")
        for cat, sc in sorted(result.category_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  * {cat.ljust(15)} : {sc * 100:.1f}%")

        print("\nMODERATION VERDICT:")
        print(f"  >>> {result.verdict.upper()} <<<")
        print(f"  Requires Review: {'Yes' if result.requires_review else 'No'}")
        print(f"  Top Category:    {result.highest_category_display} ({result.highest_score_pct})")
        print(f"  Explanation:     {result.explanation}")
        print("=======================================================\n")
        return 0 if result.verdict == "Likely Safe" else 1

    except Exception as err:
        print(f"\n[ERROR] Analysis failed: {err}\n", file=sys.stderr)
        return 2


def run_model_download_cli(model_id: str) -> int:
    """Download model weights via CLI."""
    config = AppConfig.load()
    manager = ModelManager(config)
    print(f"Starting explicit download for '{model_id}'...")

    def progress(fraction: float, msg: str) -> None:
        print(f"[{fraction * 100:3.0f}%] {msg}")

    try:
        path = manager.download_model(model_id, progress_callback=progress)
        print(f"\n[SUCCESS] Model installed at: {path}\n")
        return 0
    except Exception as err:
        print(f"\n[ERROR] Download failed: {err}\n", file=sys.stderr)
        return 1


def main() -> int:
    """Application main entry point."""
    setup_logging()
    ensure_directories_exist()
    init_database()

    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} - Privacy-first offline desktop image content moderation."
    )
    parser.add_argument(
        "--cli",
        metavar="IMAGE_PATH",
        type=str,
        help="Moderate an image directly from the terminal without launching GUI.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="falconsai_nsfw",
        help="Model ID to use for inference (falconsai_nsfw, clip_zero_shot, synthetic_diagnostic).",
    )
    parser.add_argument(
        "--flag-threshold",
        type=float,
        default=0.70,
        help="Threshold to flag image as unsafe (0.0 to 1.0).",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.40,
        help="Threshold for manual review uncertainty range (0.0 to 1.0).",
    )
    parser.add_argument(
        "--download-model",
        metavar="MODEL_ID",
        type=str,
        help="Download model weights from Hugging Face for offline use and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and local installation status.",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Launch the FastAPI REST API server instead of desktop GUI.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for REST API server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for REST API server (default: 8000).",
    )

    args = parser.parse_args()

    if args.list_models:
        manager = ModelManager()
        models = manager.list_all_models_status()
        print(f"\nAvailable Models in Catalog:")
        print("-" * 65)
        for m in models:
            status = "Installed (Ready)" if m["installed"] else "Missing"
            print(f"[{status.ljust(17)}] {m['id']}: {m['name']}")
            print(f"  Size: {m['approx_size_mb']} MB | Path: {m['local_path']}")
        print("-" * 65 + "\n")
        return 0

    if args.download_model:
        return run_model_download_cli(args.download_model)

    if args.cli:
        return run_cli_moderation(
            image_path=args.cli,
            model_id=args.model,
            flag_th=args.flag_threshold,
            rev_th=args.review_threshold,
        )

    if args.api:
        import uvicorn
        print("\n" + "=" * 60)
        print(f"  {APP_NAME} v{APP_VERSION} — REST API Server")
        print("=" * 60)
        print(f"  Swagger Docs: http://{args.host}:{args.port}/docs")
        print(f"  Endpoints:    http://{args.host}:{args.port}/api/v1")
        print("=" * 60 + "\n")
        uvicorn.run("api.app:app", host=args.host, port=args.port, log_level="info")
        return 0

    # Launch desktop GUI
    from ui.app import ModeratorApp

    app = ModeratorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
