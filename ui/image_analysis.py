"""Single-image moderation analysis view for Offline AI Image Moderator.

Provides image selection, large preview, metadata inspection, background AI inference,
verdict presentation, category score meters, and safety explanations.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Allow running directly
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import customtkinter as ctk  # type: ignore

from config import CATEGORY_DISPLAY_NAMES, MODEL_CATALOG, SUPPORTED_EXTENSIONS
from moderation.classifier import ClassifierRegistry
from moderation.image_loader import LoadedImageData, load_image_for_inference
from moderation.model_manager import ModelNotInstalledError
from moderation.scoring import ModerationResult, ScoringEngine, Verdict
from storage.history_repository import ScanRecord
from ui.components import (
    COLORS,
    CardFrame,
    ImagePreviewBox,
    ScoreProgressBar,
    VerdictBadge,
)
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from ui.app import ModeratorApp

logger = get_logger("image_analysis_view")


class ImageAnalysisView(ctk.CTkScrollableFrame):
    """Interactive single-image moderation workspace."""

    def __init__(
        self,
        master: Any,
        app: ModeratorApp,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app
        self.config = app.config
        self.history_repo = app.history_repo
        self.model_manager = app.model_manager

        self.current_loaded_image: Optional[LoadedImageData] = None
        self.last_moderation_result: Optional[ModerationResult] = None
        self.is_analyzing = False

        self._build_layout()

    def _build_layout(self) -> None:
        """Create split two-column workspace: Preview & Metadata on left, Analysis & Scores on right."""
        # Top toolbar
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", padx=16, pady=(16, 12))

        title_lbl = ctk.CTkLabel(
            toolbar,
            text="Single Image Moderation",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_main"],
        )
        title_lbl.pack(side="left")

        btn_box = ctk.CTkFrame(toolbar, fg_color="transparent")
        btn_box.pack(side="right")

        self.select_btn = ctk.CTkButton(
            btn_box,
            text="📁 Select Image...",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            height=34,
            command=self.select_image_dialog,
        )
        self.select_btn.pack(side="left", padx=(0, 8))

        self.clear_btn = ctk.CTkButton(
            btn_box,
            text="Clear",
            font=ctk.CTkFont(size=12),
            fg_color="#334155",
            hover_color="#475569",
            width=70,
            height=34,
            command=self.reset_view,
        )
        self.clear_btn.pack(side="left")

        # Two-column container
        columns_frame = ctk.CTkFrame(self, fg_color="transparent")
        columns_frame.pack(fill="both", expand=True, padx=16, pady=(0, 20))
        columns_frame.grid_columnconfigure(0, weight=1)
        columns_frame.grid_columnconfigure(1, weight=1)

        # Left Column: Image Preview and Metadata
        left_col = ctk.CTkFrame(columns_frame, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.preview_box = ImagePreviewBox(left_col, width=440, height=360)
        self.preview_box.pack(fill="x", pady=(0, 12))

        # Metadata Card
        self.meta_card = CardFrame(left_col)
        self.meta_card.pack(fill="x")

        meta_title = ctk.CTkLabel(
            self.meta_card,
            text="IMAGE PROPERTIES",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        meta_title.pack(anchor="w", padx=16, pady=(12, 6))

        self.filename_label = ctk.CTkLabel(
            self.meta_card,
            text="No file selected",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_main"],
        )
        self.filename_label.pack(anchor="w", padx=16, pady=2)

        self.props_label = ctk.CTkLabel(
            self.meta_card,
            text="Resolution: — • File size: — • Format: —",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        )
        self.props_label.pack(anchor="w", padx=16, pady=(0, 12))

        # Right Column: Moderation Verdict & Category Scores
        right_col = ctk.CTkFrame(columns_frame, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Verdict Card
        self.verdict_card = CardFrame(right_col)
        self.verdict_card.pack(fill="x", pady=(0, 12))

        verdict_top = ctk.CTkFrame(self.verdict_card, fg_color="transparent")
        verdict_top.pack(fill="x", padx=16, pady=(14, 8))

        verdict_title = ctk.CTkLabel(
            verdict_top,
            text="MODERATION VERDICT",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        verdict_title.pack(side="left")

        self.verdict_badge = VerdictBadge(verdict_top, verdict="Pending")
        self.verdict_badge.pack(side="right")

        self.explanation_label = ctk.CTkLabel(
            self.verdict_card,
            text="Ready to inspect. Click 'Analyze Image' below to start local offline inference.",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_main"],
            wraplength=400,
            justify="left",
        )
        self.explanation_label.pack(anchor="w", padx=16, pady=(0, 8))

        self.runtime_meta_label = ctk.CTkLabel(
            self.verdict_card,
            text="Model: — • Latency: — • Device: —",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        )
        self.runtime_meta_label.pack(anchor="w", padx=16, pady=(0, 14))

        # Category Scores Card
        self.scores_card = CardFrame(right_col)
        self.scores_card.pack(fill="x", pady=(0, 12))

        scores_title = ctk.CTkLabel(
            self.scores_card,
            text="CATEGORY CONFIDENCE SCORES",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        scores_title.pack(anchor="w", padx=16, pady=(14, 8))

        self.score_bars_frame = ctk.CTkFrame(self.scores_card, fg_color="transparent")
        self.score_bars_frame.pack(fill="x", padx=16, pady=(0, 12))

        self.score_bars: Dict[str, ScoreProgressBar] = {}
        self._init_score_bars()

        # Action Buttons & Progress Bar
        self.action_card = CardFrame(right_col)
        self.action_card.pack(fill="x")

        self.analyze_btn = ctk.CTkButton(
            self.action_card,
            text="▶  Analyze Image",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            height=40,
            command=self.run_analysis_async,
            state="disabled",
        )
        self.analyze_btn.pack(fill="x", padx=16, pady=(16, 10))

        self.progress_bar = ctk.CTkProgressBar(self.action_card, mode="indeterminate", height=6)
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 12))
        self.progress_bar.pack_forget()

        # Responsible AI Disclaimer
        disclaimer_label = ctk.CTkLabel(
            self.action_card,
            text="⚠️ AI predictions are advisory recommendations. Verify ambiguous cases manually.",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_muted"],
            wraplength=400,
            justify="center",
        )
        disclaimer_label.pack(anchor="center", padx=16, pady=(0, 12))

    def _init_score_bars(self) -> None:
        """Create category progress meters based on configuration."""
        for child in self.score_bars_frame.winfo_children():
            child.destroy()
        self.score_bars.clear()

        # Standard categories
        order = [
            "nudity",
            "buttocks_exposed",
            "partial_nudity",
            "suggestive",
            "violence",
            "gore",
            "hate_symbols",
            "nsfw",
        ]
        for cat in order:
            disp_name = CATEGORY_DISPLAY_NAMES.get(cat, cat.title())
            bar = ScoreProgressBar(
                self.score_bars_frame,
                category_name=cat,
                display_name=disp_name,
                score=0.0,
                flag_thresh=self.config.flag_threshold,
                review_thresh=self.config.review_threshold,
            )
            bar.pack(fill="x", pady=2)
            self.score_bars[cat] = bar

    def select_image_dialog(self) -> None:
        """Open file dialog for choosing single image."""
        ext_filter = " ".join([f"*{ext}" for ext in SUPPORTED_EXTENSIONS])
        file_path = filedialog.askopenfilename(
            title="Select Image for Moderation",
            filetypes=[
                ("Supported Images", ext_filter),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.load_image_file(file_path)

    def load_image_file(self, file_path: str) -> None:
        """Safely load and preview the specified image file."""
        try:
            loaded = load_image_for_inference(
                file_path,
                max_size_mb=self.config.max_file_size_mb,
                max_dimension=self.config.max_image_dimension,
            )
            self.current_loaded_image = loaded

            # Update preview box
            self.preview_box.display_image(loaded.image)

            # Update metadata labels
            self.filename_label.configure(text=loaded.filename)
            self.props_label.configure(
                text=(
                    f"Resolution: {loaded.dimensions_display} • "
                    f"File size: {loaded.file_size_display} • "
                    f"Format: {loaded.format_name} ({loaded.color_mode})"
                )
            )

            # Enable analyze button and reset verdict card
            self.analyze_btn.configure(state="normal")
            self.verdict_badge.set_verdict("Pending")
            self.explanation_label.configure(
                text="Image loaded successfully. Click 'Analyze Image' to begin offline evaluation."
            )
            self.runtime_meta_label.configure(text="Model: — • Latency: — • Device: —")

            # Zero out score bars
            for bar in self.score_bars.values():
                bar.set_score(0.0)

        except Exception as err:
            logger.error("Failed to load image: %s", err)
            messagebox.showerror("Image Loading Error", str(err))

    def run_analysis_async(self) -> None:
        """Trigger AI inference in a background thread to maintain GUI responsiveness."""
        if not self.current_loaded_image or self.is_analyzing:
            return

        self.is_analyzing = True
        self.analyze_btn.configure(state="disabled", text="Analyzing...")
        self.select_btn.configure(state="disabled")
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 12))
        self.progress_bar.start()
        self.verdict_badge.set_verdict("Processing...")

        thread = threading.Thread(target=self._inference_worker, daemon=True)
        thread.start()

    def _inference_worker(self) -> None:
        """Worker thread executing model inference and score evaluation."""
        loaded = self.current_loaded_image
        if loaded is None:
            return

        start_time = time.time()
        try:
            # Enforce offline flags
            self.model_manager.enforce_offline_environment()

            model_id = self.config.active_model_id
            classifier = ClassifierRegistry.get_classifier(
                model_id=model_id,
                config=self.config,
                model_manager=self.model_manager,
            )

            # Run prediction
            raw_scores = classifier.predict(loaded.image)
            latency = time.time() - start_time

            # Evaluate with ScoringEngine
            result = ScoringEngine.evaluate(
                category_scores=raw_scores,
                flag_threshold=self.config.flag_threshold,
                review_threshold=self.config.review_threshold,
                enabled_categories=self.config.enabled_categories,
                model_id=model_id,
                model_name=classifier.model_name,
                inference_latency_sec=latency,
            )

            # Dispatch UI update on main thread
            self.after(0, lambda: self._on_analysis_success(result, classifier.device))

        except ModelNotInstalledError as not_installed_err:
            logger.warning("Selected model not downloaded: %s", not_installed_err)
            self.after(0, lambda: self._on_model_missing_error(str(not_installed_err)))

        except Exception as err:
            logger.error("Inference execution failed: %s", err, exc_info=True)
            self.after(0, lambda: self._on_analysis_failure(str(err)))

    def _on_analysis_success(self, result: ModerationResult, device: str) -> None:
        """Handle successful analysis completion on UI thread."""
        self._finish_analysis_state()
        self.last_moderation_result = result

        # 1. Update Verdict Card
        self.verdict_badge.set_verdict(result.verdict)
        self.explanation_label.configure(text=result.explanation)
        self.runtime_meta_label.configure(
            text=f"Model: {result.model_name} • Latency: {result.inference_latency_sec:.2f}s • Device: {device.upper()}"
        )

        # 2. Update Category Score bars
        for cat, bar in self.score_bars.items():
            score = result.category_scores.get(cat, 0.0)
            bar.set_score(score)

        # 3. Save to database if history enabled
        if self.config.enable_history and self.current_loaded_image:
            try:
                record = ScanRecord(
                    filename=self.current_loaded_image.filename,
                    file_path=str(self.current_loaded_image.file_path),
                    file_size_bytes=self.current_loaded_image.file_size_bytes,
                    dimensions=self.current_loaded_image.dimensions_display,
                    format_name=self.current_loaded_image.format_name,
                    verdict=result.verdict,
                    requires_review=result.requires_review,
                    highest_category=result.highest_category_display,
                    highest_score=result.highest_score,
                    category_scores=result.category_scores,
                    model_id=result.model_id,
                    model_name=result.model_name,
                    inference_latency_sec=result.inference_latency_sec,
                )
                self.history_repo.save_record(record)
            except Exception as db_err:
                logger.warning("Failed to save history record: %s", db_err)

    def _on_model_missing_error(self, err_msg: str) -> None:
        """Handle situation where model weights are not found offline."""
        self._finish_analysis_state()
        self.verdict_badge.set_verdict("Error")
        self.explanation_label.configure(
            text=(
                f"Model weights are not downloaded for offline use.\n\n"
                f"{err_msg}\n\n"
                f"Go to Settings to download model weights, or select the Diagnostic Test Engine."
            )
        )
        messagebox.showwarning(
            "Model Not Installed",
            f"The selected model is not installed locally.\n\n{err_msg}\n\n"
            "You can download it in Settings, or switch to the Diagnostic Test Engine.",
        )

    def _on_analysis_failure(self, err_msg: str) -> None:
        """Handle unexpected failure during inference."""
        self._finish_analysis_state()
        self.verdict_badge.set_verdict("Error")
        self.explanation_label.configure(text=f"Analysis failed: {err_msg}")
        messagebox.showerror("Moderation Error", f"Analysis failed:\n{err_msg}")

    def _finish_analysis_state(self) -> None:
        """Reset buttons and stop progress animation."""
        self.is_analyzing = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.analyze_btn.configure(state="normal", text="▶  Re-Analyze Image")
        self.select_btn.configure(state="normal")

    def reset_view(self) -> None:
        """Clear the current preview, metadata, and verdict."""
        self.current_loaded_image = None
        self.last_moderation_result = None
        self.preview_box.clear()
        self.filename_label.configure(text="No file selected")
        self.props_label.configure(text="Resolution: — • File size: — • Format: —")
        self.verdict_badge.set_verdict("Pending")
        self.explanation_label.configure(
            text="Ready to inspect. Choose an image to begin offline evaluation."
        )
        self.runtime_meta_label.configure(text="Model: — • Latency: — • Device: —")
        self.analyze_btn.configure(state="disabled", text="▶  Analyze Image")
        for bar in self.score_bars.values():
            bar.set_score(0.0)


if __name__ == "__main__":
    from ui.app import ModeratorApp
    from utils.logging_config import setup_logging

    setup_logging()
    app = ModeratorApp()
    app.navigate_to("single")
    app.mainloop()

