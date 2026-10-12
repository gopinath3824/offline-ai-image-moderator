"""Scan history view for Offline AI Image Moderator.

Provides search, filtering, detailed score inspection, deletion, and CSV export
for locally persisted scan records.
"""

from __future__ import annotations

import json
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import customtkinter as ctk  # type: ignore

from config import CATEGORY_DISPLAY_NAMES
from moderation.scoring import Verdict
from storage.history_repository import HistoryRepository, ScanRecord
from ui.components import COLORS, CardFrame, VerdictBadge
from utils.exporter import export_history_to_csv
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from ui.app import ModeratorApp

logger = get_logger("history_view")


class HistoryView(ctk.CTkFrame):
    """Scan history manager interface."""

    def __init__(
        self,
        master: Any,
        app: ModeratorApp,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app
        self.history_repo = app.history_repo
        self.config = app.config

        self.current_records: List[ScanRecord] = []
        self.selected_record: Optional[ScanRecord] = None

        self._build_layout()
        self.refresh()

    def _build_layout(self) -> None:
        """Construct search bar, action buttons, table view, and details inspection panel."""
        # Top toolbar
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", padx=16, pady=(16, 10))

        title_lbl = ctk.CTkLabel(
            toolbar,
            text="Scan History",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_main"],
        )
        title_lbl.pack(side="left")

        actions_box = ctk.CTkFrame(toolbar, fg_color="transparent")
        actions_box.pack(side="right")

        self.export_btn = ctk.CTkButton(
            actions_box,
            text="📥 Export to CSV",
            font=ctk.CTkFont(size=12),
            fg_color="#334155",
            hover_color="#475569",
            height=32,
            command=self.export_csv,
        )
        self.export_btn.pack(side="left", padx=4)

        self.delete_btn = ctk.CTkButton(
            actions_box,
            text="Delete Selected",
            font=ctk.CTkFont(size=12),
            fg_color="#475569",
            hover_color="#334155",
            height=32,
            command=self.delete_selected,
        )
        self.delete_btn.pack(side="left", padx=4)

        self.clear_all_btn = ctk.CTkButton(
            actions_box,
            text="Clear All",
            font=ctk.CTkFont(size=12),
            fg_color=COLORS["flagged"],
            hover_color="#dc2626",
            height=32,
            command=self.clear_all_history,
        )
        self.clear_all_btn.pack(side="left", padx=4)

        # Search and Filter FilterBar
        filter_card = CardFrame(self)
        filter_card.pack(fill="x", padx=16, pady=(0, 10))

        f_inner = ctk.CTkFrame(filter_card, fg_color="transparent")
        f_inner.pack(fill="x", padx=16, pady=10)

        # Search box
        self.search_entry = ctk.CTkEntry(
            f_inner,
            placeholder_text="🔍 Search by filename or category...",
            width=280,
            height=32,
        )
        self.search_entry.pack(side="left", padx=(0, 12))
        self.search_entry.bind("<Return>", lambda e: self.refresh())

        search_btn = ctk.CTkButton(
            f_inner,
            text="Search",
            width=80,
            height=32,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            command=self.refresh,
        )
        search_btn.pack(side="left", padx=(0, 16))

        # Verdict filter menu
        self.verdict_filter_var = ctk.StringVar(value="All")
        self.verdict_filter = ctk.CTkSegmentedButton(
            f_inner,
            values=["All", "Flagged", "Needs Review", "Likely Safe"],
            variable=self.verdict_filter_var,
            command=lambda v: self.refresh(),
            selected_color=COLORS["primary"],
        )
        self.verdict_filter.pack(side="left")

        # Main Split Content: Table on Left (60%), Details on Right (40%)
        split_frame = ctk.CTkFrame(self, fg_color="transparent")
        split_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        split_frame.grid_columnconfigure(0, weight=3)
        split_frame.grid_columnconfigure(1, weight=2)

        # Left Table Card
        table_card = CardFrame(split_frame)
        table_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        columns = ("id", "filename", "verdict", "top_cat", "top_score", "timestamp")
        self.tree = ttk.Treeview(table_card, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("id", text="#")
        self.tree.heading("filename", text="Filename")
        self.tree.heading("verdict", text="Verdict")
        self.tree.heading("top_cat", text="Top Category")
        self.tree.heading("top_score", text="Score")
        self.tree.heading("timestamp", text="Timestamp")

        self.tree.column("id", width=40, anchor="center")
        self.tree.column("filename", width=200, anchor="w")
        self.tree.column("verdict", width=100, anchor="center")
        self.tree.column("top_cat", width=120, anchor="center")
        self.tree.column("top_score", width=70, anchor="center")
        self.tree.column("timestamp", width=130, anchor="center")

        tree_scroll = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        tree_scroll.pack(side="right", fill="y", padx=(0, 10), pady=10)

        self.tree.bind("<<TreeviewSelect>>", self._on_row_selected)

        # Right Detail Inspection Card
        self.detail_card = CardFrame(split_frame)
        self.detail_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        detail_title = ctk.CTkLabel(
            self.detail_card,
            text="RECORD DETAILS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        detail_title.pack(anchor="w", padx=16, pady=(14, 6))

        self.detail_name = ctk.CTkLabel(
            self.detail_card,
            text="Select a scan to inspect details",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_main"],
            wraplength=300,
            justify="left",
        )
        self.detail_name.pack(anchor="w", padx=16, pady=2)

        self.detail_badge = VerdictBadge(self.detail_card, verdict="Pending")
        self.detail_badge.pack(anchor="w", padx=16, pady=6)

        self.detail_meta = ctk.CTkLabel(
            self.detail_card,
            text="Path: —\nModel: —\nTime: —\nDimensions: —",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
            justify="left",
            wraplength=300,
        )
        self.detail_meta.pack(anchor="w", padx=16, pady=6)

        # Detail scores breakdown container
        self.detail_scores_box = ctk.CTkScrollableFrame(self.detail_card, fg_color="#182234", height=200)
        self.detail_scores_box.pack(fill="both", expand=True, padx=16, pady=(6, 16))

    def refresh(self) -> None:
        """Query repository and reload records into treeview."""
        query = self.search_entry.get().strip()
        v_filter = self.verdict_filter_var.get()

        records = self.history_repo.list_records(
            limit=300,
            verdict_filter=v_filter if v_filter != "All" else None,
            search_query=query if query else None,
        )
        self.current_records = records

        # Clear existing rows
        for row in self.tree.get_children():
            self.tree.delete(row)

        for r in records:
            score_str = f"{r.highest_score * 100:.1f}%" if r.highest_score > 0 else "0.0%"
            self.tree.insert(
                "",
                "end",
                iid=str(r.id),
                values=(
                    r.id,
                    r.filename,
                    r.verdict,
                    r.highest_category,
                    score_str,
                    r.timestamp,
                ),
            )

        if not records:
            self._clear_detail_panel()

    def _on_row_selected(self, event: Any) -> None:
        """Display full scores and metadata for selected record."""
        selected_items = self.tree.selection()
        if not selected_items:
            return

        rec_id = int(selected_items[0])
        record = next((r for r in self.current_records if r.id == rec_id), None)
        if not record:
            return

        self.selected_record = record
        self.detail_name.configure(text=record.filename)
        self.detail_badge.set_verdict(record.verdict)
        self.detail_meta.configure(
            text=(
                f"File: {record.file_path}\n"
                f"Model: {record.model_name}\n"
                f"Timestamp: {record.timestamp}\n"
                f"Resolution: {record.dimensions_display} • Size: {record.file_size_display}\n"
                f"Inference Time: {record.inference_latency_sec:.2f}s"
            )
        )

        # Populate individual category scores
        for child in self.detail_scores_box.winfo_children():
            child.destroy()

        if record.category_scores:
            for cat, score in sorted(record.category_scores.items(), key=lambda x: x[1], reverse=True):
                disp_name = CATEGORY_DISPLAY_NAMES.get(cat, cat.title())
                row = ctk.CTkFrame(self.detail_scores_box, fg_color="transparent")
                row.pack(fill="x", pady=2)

                lbl = ctk.CTkLabel(
                    row,
                    text=disp_name,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=COLORS["text_main"],
                )
                lbl.pack(side="left")

                score_lbl = ctk.CTkLabel(
                    row,
                    text=f"{score * 100:.1f}%",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=COLORS["flagged"] if score >= self.config.flag_threshold else (
                        COLORS["review"] if score >= self.config.review_threshold else COLORS["safe"]
                    ),
                )
                score_lbl.pack(side="right")
        else:
            lbl = ctk.CTkLabel(
                self.detail_scores_box,
                text="No breakdown scores recorded.",
                font=ctk.CTkFont(size=11),
                text_color=COLORS["text_muted"],
            )
            lbl.pack()

    def _clear_detail_panel(self) -> None:
        """Reset detail card to placeholder state."""
        self.selected_record = None
        self.detail_name.configure(text="No record selected")
        self.detail_badge.set_verdict("Pending")
        self.detail_meta.configure(text="Path: —\nModel: —\nTime: —\nDimensions: —")
        for child in self.detail_scores_box.winfo_children():
            child.destroy()

    def delete_selected(self) -> None:
        """Delete currently highlighted record."""
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showinfo("Selection Required", "Please select a scan record to delete.")
            return

        rec_id = int(selected_items[0])
        if messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete this scan record?"):
            self.history_repo.delete_record(rec_id)
            self.refresh()
            self._clear_detail_panel()

    def clear_all_history(self) -> None:
        """Delete all history entries after user confirmation."""
        if not self.current_records:
            return
        if messagebox.askyesno(
            "Clear Entire History",
            "This will permanently delete all scan records from your local database.\n\nContinue?",
        ):
            self.history_repo.clear_all()
            self.refresh()
            self._clear_detail_panel()
            messagebox.showinfo("History Cleared", "All scan history has been cleared.")

    def export_csv(self) -> None:
        """Export current filtered history records to CSV."""
        if not self.current_records:
            messagebox.showinfo("Export", "No records found to export.")
            return

        dest = filedialog.asksaveasfilename(
            title="Export History to CSV",
            defaultextension=".csv",
            filetypes=[("CSV Spreadsheet", "*.csv")],
            initialfile="moderator_scan_history.csv",
        )
        if not dest:
            return

        try:
            records_dict = [r.to_dict() for r in self.current_records]
            export_history_to_csv(records_dict, dest)
            messagebox.showinfo("Export Successful", f"Saved {len(records_dict)} records to:\n{dest}")
        except Exception as err:
            logger.error("Failed to export history: %s", err)
            messagebox.showerror("Export Error", str(err))
