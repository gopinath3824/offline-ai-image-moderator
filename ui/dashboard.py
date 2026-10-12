"""Dashboard view for Offline AI Image Moderator.

Displays high-level moderation statistics, privacy badge, model readiness,
quick action triggers, and recent scan activity feed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import customtkinter as ctk  # type: ignore

from config import APP_DESCRIPTION, APP_NAME, APP_VERSION, MODEL_CATALOG, AppConfig
from moderation.model_manager import ModelManager
from storage.history_repository import HistoryRepository, ScanRecord
from ui.components import COLORS, CardFrame, StatCard, VerdictBadge

if TYPE_CHECKING:
    from ui.app import ModeratorApp


class DashboardView(ctk.CTkScrollableFrame):
    """Primary application dashboard."""

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
        self.model_manager = app.model_manager

        self._build_header()
        self._build_kpi_row()
        self._build_status_and_actions()
        self._build_recent_activity_section()
        self.refresh()

    def _build_header(self) -> None:
        """Top banner card with branding and privacy indicator."""
        header_card = CardFrame(self, fg_color=COLORS["card_dark"])
        header_card.pack(fill="x", padx=16, pady=(16, 12))

        top_row = ctk.CTkFrame(header_card, fg_color="transparent")
        top_row.pack(fill="x", padx=20, pady=(16, 4))

        title_label = ctk.CTkLabel(
            top_row,
            text=f"🛡️ {APP_NAME}",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLORS["text_main"],
        )
        title_label.pack(side="left")

        privacy_pill = ctk.CTkLabel(
            top_row,
            text="🔒 100% Offline • Zero Cloud Telemetry",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#34d399",
            fg_color="#064e3b",
            corner_radius=12,
            padx=12,
            pady=4,
        )
        privacy_pill.pack(side="right")

        desc_label = ctk.CTkLabel(
            header_card,
            text=f"{APP_DESCRIPTION} (v{APP_VERSION}). Securely inspect images locally on Windows.",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_muted"],
        )
        desc_label.pack(anchor="w", padx=20, pady=(0, 16))

    def _build_kpi_row(self) -> None:
        """Metric counter cards for scan volume and verdicts."""
        kpi_frame = ctk.CTkFrame(self, fg_color="transparent")
        kpi_frame.pack(fill="x", padx=16, pady=(0, 12))
        kpi_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.stat_total = StatCard(
            kpi_frame,
            title="Total Scanned",
            value="0",
            subtitle="Lifetime image scans",
            accent_color=COLORS["primary"],
        )
        self.stat_total.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        self.stat_flagged = StatCard(
            kpi_frame,
            title="Flagged Unsafe",
            value="0",
            subtitle="Exceeded flag threshold",
            accent_color=COLORS["flagged"],
        )
        self.stat_flagged.grid(row=0, column=1, padx=4, sticky="ew")

        self.stat_review = StatCard(
            kpi_frame,
            title="Needs Review",
            value="0",
            subtitle="Uncertain / borderline",
            accent_color=COLORS["review"],
        )
        self.stat_review.grid(row=0, column=2, padx=4, sticky="ew")

        self.stat_safe = StatCard(
            kpi_frame,
            title="Likely Safe",
            value="0",
            subtitle="Below review threshold",
            accent_color=COLORS["safe"],
        )
        self.stat_safe.grid(row=0, column=3, padx=(8, 0), sticky="ew")

    def _build_status_and_actions(self) -> None:
        """Two-column section: Model readiness status on left, quick actions on right."""
        row_frame = ctk.CTkFrame(self, fg_color="transparent")
        row_frame.pack(fill="x", padx=16, pady=(0, 12))
        row_frame.grid_columnconfigure(0, weight=3)
        row_frame.grid_columnconfigure(1, weight=2)

        # Left: Model & System Status card
        self.model_status_card = CardFrame(row_frame)
        self.model_status_card.grid(row=0, column=0, padx=(0, 8), sticky="nsew")

        status_header = ctk.CTkLabel(
            self.model_status_card,
            text="⚡ ACTIVE AI ENGINE & RUNTIME",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        status_header.pack(anchor="w", padx=20, pady=(16, 6))

        self.model_name_label = ctk.CTkLabel(
            self.model_status_card,
            text="",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=COLORS["text_main"],
        )
        self.model_name_label.pack(anchor="w", padx=20, pady=2)

        self.model_desc_label = ctk.CTkLabel(
            self.model_status_card,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
            wraplength=440,
            justify="left",
        )
        self.model_desc_label.pack(anchor="w", padx=20, pady=(2, 8))

        # Runtime tags
        self.tags_frame = ctk.CTkFrame(self.model_status_card, fg_color="transparent")
        self.tags_frame.pack(fill="x", padx=20, pady=(0, 16))

        self.installed_badge = ctk.CTkLabel(
            self.tags_frame,
            text="Status: Checking...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#ffffff",
            fg_color="#334155",
            corner_radius=8,
            padx=10,
            pady=3,
        )
        self.installed_badge.pack(side="left", padx=(0, 8))

        self.device_badge = ctk.CTkLabel(
            self.tags_frame,
            text="Device: CPU",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#93c5fd",
            fg_color="#1e3a8a",
            corner_radius=8,
            padx=10,
            pady=3,
        )
        self.device_badge.pack(side="left")

        # Right: Quick Action triggers
        action_card = CardFrame(row_frame)
        action_card.grid(row=0, column=1, padx=(8, 0), sticky="nsew")

        action_header = ctk.CTkLabel(
            action_card,
            text="🚀 QUICK ACTIONS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        action_header.pack(anchor="w", padx=20, pady=(16, 12))

        btn_single = ctk.CTkButton(
            action_card,
            text="📷 Scan Single Image",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            height=36,
            command=lambda: self.app.navigate_to("single"),
        )
        btn_single.pack(fill="x", padx=20, pady=(0, 8))

        btn_batch = ctk.CTkButton(
            action_card,
            text="📚 Batch Folder / Multi-Image",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#475569",
            hover_color="#334155",
            height=36,
            command=lambda: self.app.navigate_to("batch"),
        )
        btn_batch.pack(fill="x", padx=20, pady=(0, 8))

        btn_models = ctk.CTkButton(
            action_card,
            text="⚙️ Manage Models & Thresholds",
            font=ctk.CTkFont(size=13),
            fg_color="transparent",
            border_width=1,
            border_color=COLORS["border_dark"],
            hover_color=COLORS["card_hover"],
            height=34,
            command=lambda: self.app.navigate_to("settings"),
        )
        btn_models.pack(fill="x", padx=20, pady=(0, 16))

    def _build_recent_activity_section(self) -> None:
        """Feed of recent image moderation scans."""
        self.activity_card = CardFrame(self)
        self.activity_card.pack(fill="x", padx=16, pady=(0, 20))

        header_row = ctk.CTkFrame(self.activity_card, fg_color="transparent")
        header_row.pack(fill="x", padx=20, pady=(16, 10))

        title = ctk.CTkLabel(
            header_row,
            text="🕒 RECENT MODERATION ACTIVITY",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        title.pack(side="left")

        view_all_btn = ctk.CTkButton(
            header_row,
            text="View Full History →",
            font=ctk.CTkFont(size=12),
            fg_color="transparent",
            text_color=COLORS["primary"],
            hover_color=COLORS["card_hover"],
            width=130,
            height=28,
            command=lambda: self.app.navigate_to("history"),
        )
        view_all_btn.pack(side="right")

        self.recent_items_frame = ctk.CTkFrame(self.activity_card, fg_color="transparent")
        self.recent_items_frame.pack(fill="x", padx=20, pady=(0, 16))

    def refresh(self) -> None:
        """Refresh dashboard statistics, model status, and recent activity feed."""
        # 1. Update KPI numbers
        stats = self.history_repo.get_summary_stats()
        self.stat_total.update_value(stats["total"], f"{stats['safe']} safe • {stats['flagged']} flagged")
        self.stat_flagged.update_value(stats["flagged"])
        self.stat_review.update_value(stats["review"])
        self.stat_safe.update_value(stats["safe"])

        # 2. Update Model Status card
        meta = MODEL_CATALOG.get(self.config.active_model_id)
        if meta:
            self.model_name_label.configure(text=meta.name)
            self.model_desc_label.configure(text=meta.description)
            is_installed = self.model_manager.is_model_installed(meta.id)
            if is_installed:
                size_str = self.model_manager.get_installed_size_display(meta.id)
                self.installed_badge.configure(
                    text=f"✓ Ready Offline ({size_str})",
                    fg_color="#064e3b",
                    text_color="#6ee7b7",
                )
            else:
                self.installed_badge.configure(
                    text=f"⚠️ Not Downloaded ({meta.size_mb} MB required)",
                    fg_color="#78350f",
                    text_color="#fcd34d",
                )
        self.device_badge.configure(text=f"Device: {self.config.device_preference.upper()}")

        # 3. Populate recent activity items
        for child in self.recent_items_frame.winfo_children():
            child.destroy()

        recent: List[ScanRecord] = self.history_repo.get_recent_activity(limit=6)
        if not recent:
            empty_lbl = ctk.CTkLabel(
                self.recent_items_frame,
                text="No scans recorded yet. Select an image above to run your first offline analysis.",
                font=ctk.CTkFont(size=13),
                text_color=COLORS["text_muted"],
                pady=16,
            )
            empty_lbl.pack()
            return

        for record in recent:
            row = ctk.CTkFrame(self.recent_items_frame, fg_color="#182234", corner_radius=8)
            row.pack(fill="x", pady=3)

            # Filename & time
            left_box = ctk.CTkFrame(row, fg_color="transparent")
            left_box.pack(side="left", padx=12, pady=8)

            fname_lbl = ctk.CTkLabel(
                left_box,
                text=record.filename,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=COLORS["text_main"],
            )
            fname_lbl.pack(anchor="w")

            time_lbl = ctk.CTkLabel(
                left_box,
                text=f"{record.timestamp} • {record.model_name}",
                font=ctk.CTkFont(size=11),
                text_color=COLORS["text_muted"],
            )
            time_lbl.pack(anchor="w")

            # Right: Top category score and Verdict Badge
            right_box = ctk.CTkFrame(row, fg_color="transparent")
            right_box.pack(side="right", padx=12, pady=8)

            score_txt = (
                f"{record.highest_category}: {record.highest_score * 100:.1f}%"
                if record.highest_category and record.highest_category != "None"
                else "Clean"
            )
            score_lbl = ctk.CTkLabel(
                right_box,
                text=score_txt,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=COLORS["text_muted"],
            )
            score_lbl.pack(side="left", padx=(0, 12))

            badge = VerdictBadge(right_box, verdict=record.verdict, font_size=11)
            badge.pack(side="left")
