"""Batch moderation view for processing multiple images or entire directories.

Features queue management, real-time progress indicators, sortable/filterable results table,
CSV report export, and automatic retry of failed scans.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import customtkinter as ctk  # type: ignore

from config import MODEL_CATALOG, SUPPORTED_EXTENSIONS
from moderation.classifier import ClassifierRegistry
from moderation.image_loader import load_image_for_inference
from moderation.model_manager import ModelNotInstalledError
from moderation.scoring import ModerationResult, ScoringEngine, Verdict
from storage.history_repository import ScanRecord
from ui.components import COLORS, CardFrame, VerdictBadge
from utils.exporter import export_batch_results_to_csv
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from ui.app import ModeratorApp

logger = get_logger("batch_analysis_view")


@dataclass
class BatchItem:
    """An individual file in the batch moderation queue."""

    id: int
    file_path: Path
    filename: str
    status: str = "Queued"  # "Queued", "Processing", "Completed", "Failed"
    verdict: str = "Pending"
    highest_category: str = "—"
    highest_score: float = 0.0
    category_scores: Dict[str, float] = None  # type: ignore
    inference_latency_sec: float = 0.0
    error_message: str = ""
    file_size_display: str = ""
    dimensions_display: str = ""
    requires_review: bool = False

    def __post_init__(self) -> None:
        if self.category_scores is None:
            self.category_scores = {}


class BatchAnalysisView(ctk.CTkFrame):
    """Batch analysis interface."""

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

        self.queue_items: List[BatchItem] = []
        self.is_running = False
        self.stop_requested = False
        self._next_id = 1

        self._build_layout()

    def _build_layout(self) -> None:
        """Create controls, progress bar, filters, and results treeview."""
        # Top toolbar
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", padx=16, pady=(16, 10))

        title_lbl = ctk.CTkLabel(
            toolbar,
            text="Batch Image Moderation",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_main"],
        )
        title_lbl.pack(side="left")

        actions_box = ctk.CTkFrame(toolbar, fg_color="transparent")
        actions_box.pack(side="right")

        self.add_files_btn = ctk.CTkButton(
            actions_box,
            text="+ Add Files",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            height=32,
            command=self.select_files_dialog,
        )
        self.add_files_btn.pack(side="left", padx=4)

        self.add_folder_btn = ctk.CTkButton(
            actions_box,
            text="📁 Add Folder",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#334155",
            hover_color="#475569",
            height=32,
            command=self.select_folder_dialog,
        )
        self.add_folder_btn.pack(side="left", padx=4)

        self.clear_btn = ctk.CTkButton(
            actions_box,
            text="Clear Queue",
            font=ctk.CTkFont(size=12),
            fg_color="transparent",
            border_width=1,
            border_color=COLORS["border_dark"],
            hover_color=COLORS["card_hover"],
            height=32,
            command=self.clear_queue,
        )
        self.clear_btn.pack(side="left", padx=4)

        # Execution and Progress Card
        self.exec_card = CardFrame(self)
        self.exec_card.pack(fill="x", padx=16, pady=(0, 10))

        exec_row = ctk.CTkFrame(self.exec_card, fg_color="transparent")
        exec_row.pack(fill="x", padx=16, pady=12)

        self.start_btn = ctk.CTkButton(
            exec_row,
            text="▶  Start Batch Analysis",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["safe"],
            hover_color="#059669",
            height=36,
            command=self.start_batch_processing,
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = ctk.CTkButton(
            exec_row,
            text="⏹ Stop",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["flagged"],
            hover_color="#dc2626",
            height=36,
            command=self.stop_batch_processing,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=(0, 16))

        self.retry_btn = ctk.CTkButton(
            exec_row,
            text="🔄 Retry Failed",
            font=ctk.CTkFont(size=12),
            fg_color="#475569",
            hover_color="#334155",
            height=36,
            command=self.retry_failed,
        )
        self.retry_btn.pack(side="left", padx=(0, 16))

        self.export_btn = ctk.CTkButton(
            exec_row,
            text="📥 Export CSV Report",
            font=ctk.CTkFont(size=12),
            fg_color="#334155",
            hover_color="#475569",
            height=36,
            command=self.export_csv_report,
        )
        self.export_btn.pack(side="right")

        # Progress bar & Counters
        prog_row = ctk.CTkFrame(self.exec_card, fg_color="transparent")
        prog_row.pack(fill="x", padx=16, pady=(0, 12))

        self.progress_bar = ctk.CTkProgressBar(prog_row, height=10, corner_radius=5)
        self.progress_bar.pack(fill="x", pady=(0, 6))
        self.progress_bar.set(0.0)

        counter_row = ctk.CTkFrame(prog_row, fg_color="transparent")
        counter_row.pack(fill="x")

        self.status_label = ctk.CTkLabel(
            counter_row,
            text="0 images in queue",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_main"],
        )
        self.status_label.pack(side="left")

        self.breakdown_label = ctk.CTkLabel(
            counter_row,
            text="Safe: 0 • Needs Review: 0 • Flagged: 0 • Failed: 0",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
        )
        self.breakdown_label.pack(side="right")

        # Table Filter Row
        filter_row = ctk.CTkFrame(self, fg_color="transparent")
        filter_row.pack(fill="x", padx=16, pady=(0, 8))

        filter_lbl = ctk.CTkLabel(
            filter_row,
            text="Filter Table:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        filter_lbl.pack(side="left", padx=(0, 8))

        self.filter_var = ctk.StringVar(value="All")
        self.filter_menu = ctk.CTkSegmentedButton(
            filter_row,
            values=["All", "Flagged", "Needs Review", "Likely Safe", "Failed"],
            variable=self.filter_var,
            command=self._apply_filter,
            selected_color=COLORS["primary"],
        )
        self.filter_menu.pack(side="left")

        # Results Table Frame
        table_card = CardFrame(self)
        table_card.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Styled TTK Treeview
        columns = ("id", "filename", "status", "verdict", "highest_cat", "top_score", "latency", "size")
        self.tree = ttk.Treeview(table_card, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("id", text="#")
        self.tree.heading("filename", text="Filename")
        self.tree.heading("status", text="Status")
        self.tree.heading("verdict", text="Verdict")
        self.tree.heading("highest_cat", text="Top Category")
        self.tree.heading("top_score", text="Score")
        self.tree.heading("latency", text="Latency")
        self.tree.heading("size", text="Size")

        self.tree.column("id", width=40, anchor="center")
        self.tree.column("filename", width=260, anchor="w")
        self.tree.column("status", width=90, anchor="center")
        self.tree.column("verdict", width=110, anchor="center")
        self.tree.column("highest_cat", width=130, anchor="center")
        self.tree.column("top_score", width=70, anchor="center")
        self.tree.column("latency", width=70, anchor="center")
        self.tree.column("size", width=80, anchor="center")

        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        tree_scroll.pack(side="right", fill="y", padx=(0, 10), pady=10)

        # Style Treeview dark theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#1e293b",
            foreground="#f8fafc",
            fieldbackground="#1e293b",
            rowheight=26,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#0f172a",
            foreground="#94a3b8",
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Treeview", background=[("selected", "#3b82f6")])

    def select_files_dialog(self) -> None:
        """Add multiple individual image files to queue."""
        ext_filter = " ".join([f"*{ext}" for ext in SUPPORTED_EXTENSIONS])
        file_paths = filedialog.askopenfilenames(
            title="Select Images for Batch Moderation",
            filetypes=[("Images", ext_filter), ("All files", "*.*")],
        )
        if file_paths:
            self._add_paths_to_queue([Path(p) for p in file_paths])

    def select_folder_dialog(self) -> None:
        """Scan folder for supported images and add them to queue."""
        folder = filedialog.askdirectory(title="Select Folder to Moderate")
        if folder:
            folder_path = Path(folder)
            found: List[Path] = []
            for ext in SUPPORTED_EXTENSIONS:
                found.extend(folder_path.glob(f"*{ext}"))
                found.extend(folder_path.glob(f"*{ext.upper()}"))
            # Sort uniquely
            unique_paths = sorted(list(set(found)))
            if not unique_paths:
                messagebox.showinfo("No Images Found", f"No supported images found in '{folder_path.name}'.")
                return
            self._add_paths_to_queue(unique_paths)

    def _add_paths_to_queue(self, paths: List[Path]) -> None:
        """Create BatchItem entries and insert into treeview."""
        added_count = 0
        for p in paths:
            # Avoid duplicate paths already queued
            if any(item.file_path == p for item in self.queue_items):
                continue
            item = BatchItem(
                id=self._next_id,
                file_path=p,
                filename=p.name,
                file_size_display=f"{os.path.getsize(p) / 1024:.0f} KB" if p.exists() else "—",
            )
            self._next_id += 1
            self.queue_items.append(item)
            added_count += 1

        self._refresh_table()
        self._update_progress_summary()

    def clear_queue(self) -> None:
        """Clear all items from queue if not running."""
        if self.is_running:
            messagebox.showwarning("Busy", "Cannot clear queue while batch analysis is in progress.")
            return
        self.queue_items.clear()
        self._next_id = 1
        self._refresh_table()
        self._update_progress_summary()

    def start_batch_processing(self) -> None:
        """Initiate background batch worker thread."""
        if self.is_running:
            return
        if not self.queue_items:
            messagebox.showinfo("Queue Empty", "Please add files or a folder to the batch queue first.")
            return

        self.is_running = True
        self.stop_requested = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.add_files_btn.configure(state="disabled")
        self.add_folder_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")

        thread = threading.Thread(target=self._batch_worker, daemon=True)
        thread.start()

    def stop_batch_processing(self) -> None:
        """Signal worker to halt after current image."""
        self.stop_requested = True
        self.stop_btn.configure(state="disabled", text="Stopping...")

    def _batch_worker(self) -> None:
        """Background thread executing sequential image moderation."""
        try:
            self.model_manager.enforce_offline_environment()
            model_id = self.config.active_model_id
            classifier = ClassifierRegistry.get_classifier(
                model_id=model_id,
                config=self.config,
                model_manager=self.model_manager,
            )

            total = len(self.queue_items)

            for index, item in enumerate(self.queue_items):
                if self.stop_requested:
                    logger.info("Batch execution stopped by user request.")
                    break

                # Skip items that are already completed
                if item.status == "Completed":
                    continue

                item.status = "Processing"
                self.after(0, self._refresh_item_row, item)

                start_t = time.time()
                try:
                    # Load image safely
                    loaded = load_image_for_inference(
                        item.file_path,
                        max_size_mb=self.config.max_file_size_mb,
                        max_dimension=self.config.max_image_dimension,
                    )
                    item.dimensions_display = loaded.dimensions_display
                    item.file_size_display = loaded.file_size_display

                    # Run inference
                    raw_scores = classifier.predict(loaded.image)
                    latency = time.time() - start_t

                    # Score evaluation
                    eval_result = ScoringEngine.evaluate(
                        category_scores=raw_scores,
                        flag_threshold=self.config.flag_threshold,
                        review_threshold=self.config.review_threshold,
                        enabled_categories=self.config.enabled_categories,
                        model_id=model_id,
                        model_name=classifier.model_name,
                        inference_latency_sec=latency,
                    )

                    item.status = "Completed"
                    item.verdict = eval_result.verdict
                    item.highest_category = eval_result.highest_category_display
                    item.highest_score = eval_result.highest_score
                    item.category_scores = eval_result.category_scores
                    item.inference_latency_sec = latency
                    item.requires_review = eval_result.requires_review

                    # Persist to database if history enabled
                    if self.config.enable_history:
                        try:
                            record = ScanRecord(
                                filename=item.filename,
                                file_path=str(item.file_path),
                                file_size_bytes=loaded.file_size_bytes,
                                dimensions=item.dimensions_display,
                                format_name=loaded.format_name,
                                verdict=item.verdict,
                                requires_review=item.requires_review,
                                highest_category=item.highest_category,
                                highest_score=item.highest_score,
                                category_scores=item.category_scores,
                                model_id=model_id,
                                model_name=classifier.model_name,
                                inference_latency_sec=latency,
                            )
                            self.history_repo.save_record(record)
                        except Exception as dberr:
                            logger.warning("Batch save to history failed: %s", dberr)

                except ModelNotInstalledError as mnerr:
                    item.status = "Failed"
                    item.error_message = f"Model missing: {mnerr}"
                    self.after(0, lambda: messagebox.showwarning("Model Missing", str(mnerr)))
                    break

                except Exception as err:
                    item.status = "Failed"
                    item.error_message = str(err)
                    item.verdict = "Error"
                    logger.warning("Failed to moderate batch item %s: %s", item.filename, err)

                self.after(0, self._refresh_item_row, item)
                self.after(0, self._update_progress_summary)

        finally:
            self.after(0, self._on_batch_complete)

    def _on_batch_complete(self) -> None:
        """Reset buttons after worker finishes or stops."""
        self.is_running = False
        self.stop_requested = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled", text="⏹ Stop")
        self.add_files_btn.configure(state="normal")
        self.add_folder_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")
        self._update_progress_summary()

    def retry_failed(self) -> None:
        """Reset status of failed items so they can be re-run."""
        if self.is_running:
            return
        failed_count = 0
        for item in self.queue_items:
            if item.status == "Failed":
                item.status = "Queued"
                item.verdict = "Pending"
                item.error_message = ""
                failed_count += 1
        if failed_count > 0:
            self._refresh_table()
            self._update_progress_summary()
            self.start_batch_processing()
        else:
            messagebox.showinfo("Retry", "No failed scans in the current queue.")

    def export_csv_report(self) -> None:
        """Export current batch results to CSV file."""
        if not self.queue_items:
            messagebox.showinfo("Export", "No queue items available to export.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Batch Moderation Report",
            defaultextension=".csv",
            filetypes=[("CSV Spreadsheet", "*.csv")],
            initialfile="moderation_batch_report.csv",
        )
        if not save_path:
            return

        try:
            records = [
                {
                    "filename": it.filename,
                    "file_path": str(it.file_path),
                    "verdict": it.verdict,
                    "requires_review": it.requires_review,
                    "highest_category": it.highest_category,
                    "highest_score": it.highest_score,
                    "category_scores": it.category_scores,
                    "model_name": self.config.active_model_id,
                    "file_size_display": it.file_size_display,
                    "dimensions_display": it.dimensions_display,
                    "inference_latency_sec": it.inference_latency_sec,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error_message": it.error_message,
                }
                for it in self.queue_items
            ]
            export_batch_results_to_csv(records, save_path)
            messagebox.showinfo("Export Successful", f"Report successfully saved to:\n{save_path}")
        except Exception as err:
            logger.error("Failed to export batch CSV: %s", err)
            messagebox.showerror("Export Failed", str(err))

    def _apply_filter(self, selected_filter: str) -> None:
        """Filter visible items in table based on verdict or failure."""
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Re-render treeview rows respecting the active filter."""
        # Clear existing rows
        for row_id in self.tree.get_children():
            self.tree.delete(row_id)

        target_filter = self.filter_var.get()

        for item in self.queue_items:
            if target_filter == "All":
                pass
            elif target_filter == "Failed" and item.status != "Failed":
                continue
            elif target_filter != "Failed" and item.verdict != target_filter:
                continue

            score_str = f"{item.highest_score * 100:.1f}%" if item.highest_score > 0 else "—"
            latency_str = f"{item.inference_latency_sec:.2f}s" if item.inference_latency_sec > 0 else "—"

            self.tree.insert(
                "",
                "end",
                iid=str(item.id),
                values=(
                    item.id,
                    item.filename,
                    item.status,
                    item.verdict,
                    item.highest_category,
                    score_str,
                    latency_str,
                    item.file_size_display,
                ),
            )

    def _refresh_item_row(self, item: BatchItem) -> None:
        """Update or re-insert single row in treeview."""
        if self.tree.exists(str(item.id)):
            score_str = f"{item.highest_score * 100:.1f}%" if item.highest_score > 0 else "—"
            latency_str = f"{item.inference_latency_sec:.2f}s" if item.inference_latency_sec > 0 else "—"
            self.tree.item(
                str(item.id),
                values=(
                    item.id,
                    item.filename,
                    item.status,
                    item.verdict,
                    item.highest_category,
                    score_str,
                    latency_str,
                    item.file_size_display,
                ),
            )
        else:
            self._refresh_table()

    def _update_progress_summary(self) -> None:
        """Update counter labels and progress meter."""
        total = len(self.queue_items)
        if total == 0:
            self.progress_bar.set(0.0)
            self.status_label.configure(text="0 images in queue")
            self.breakdown_label.configure(text="Safe: 0 • Needs Review: 0 • Flagged: 0 • Failed: 0")
            return

        completed = sum(1 for it in self.queue_items if it.status == "Completed")
        failed = sum(1 for it in self.queue_items if it.status == "Failed")
        processed = completed + failed

        pct = processed / total
        self.progress_bar.set(pct)
        self.status_label.configure(text=f"{processed}/{total} images analyzed ({pct * 100:.0f}%)")

        safe_cnt = sum(1 for it in self.queue_items if it.verdict == Verdict.LIKELY_SAFE.value)
        review_cnt = sum(1 for it in self.queue_items if it.verdict == Verdict.NEEDS_REVIEW.value)
        flagged_cnt = sum(1 for it in self.queue_items if it.verdict == Verdict.FLAGGED.value)

        self.breakdown_label.configure(
            text=f"Safe: {safe_cnt} • Needs Review: {review_cnt} • Flagged: {flagged_cnt} • Failed: {failed}"
        )
