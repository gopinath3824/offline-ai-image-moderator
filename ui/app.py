"""Main application window and navigation controller for Offline AI Image Moderator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import customtkinter as ctk  # type: ignore

from config import APP_NAME, APP_VERSION, AppConfig, ensure_directories_exist
from moderation.classifier import ClassifierRegistry
from moderation.model_manager import ModelManager
from storage.history_repository import HistoryRepository
from ui.batch_analysis import BatchAnalysisView
from ui.components import COLORS
from ui.dashboard import DashboardView
from ui.history import HistoryView
from ui.image_analysis import ImageAnalysisView
from ui.settings import SettingsView
from utils.logging_config import get_logger, setup_logging

logger = get_logger("app")


class ModeratorApp(ctk.CTk):
    """Root application window with sidebar navigation."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__()

        # Initialize configurations and storage
        ensure_directories_exist()
        self.config = config or AppConfig.load()
        self.history_repo = HistoryRepository()
        self.model_manager = ModelManager(self.config)

        # Enforce strict local-only offline environment on startup
        self.model_manager.enforce_offline_environment()

        # Appearance configuration
        ctk.set_appearance_mode(self.config.theme_mode)
        ctk.set_default_color_theme("blue")

        # Window properties
        self.title(f"{APP_NAME} — v{APP_VERSION} (100% Offline)")
        self.geometry("1140x740")
        self.minsize(960, 620)
        self.configure(fg_color=COLORS["bg_dark"])

        self._build_sidebar()
        self._build_content_area()

        # Protocol for clean shutdown
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Default start view
        self.navigate_to("dashboard")

    def _build_sidebar(self) -> None:
        """Create sleek left navigation sidebar."""
        self.sidebar = ctk.CTkFrame(
            self,
            width=240,
            corner_radius=0,
            fg_color="#0b1120",
            border_width=0,
        )
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Branding Header
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=16, pady=(24, 20))

        logo_lbl = ctk.CTkLabel(
            brand_frame,
            text="🛡️ AI MODERATOR",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=COLORS["text_main"],
        )
        logo_lbl.pack(anchor="w")

        sub_logo = ctk.CTkLabel(
            brand_frame,
            text="Offline Privacy-First",
            font=ctk.CTkFont(size=11),
            text_color="#34d399",
        )
        sub_logo.pack(anchor="w", pady=(2, 0))

        # Nav Buttons Container
        nav_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        nav_container.pack(fill="both", expand=True, padx=12, pady=10)

        self.nav_buttons: Dict[str, ctk.CTkButton] = {}

        nav_items = [
            ("dashboard", "📊  Dashboard"),
            ("single", "📷  Single Image"),
            ("batch", "📚  Batch Moderation"),
            ("history", "🕒  Scan History"),
            ("settings", "⚙️  Settings & Models"),
        ]

        for key, text in nav_items:
            btn = ctk.CTkButton(
                nav_container,
                text=text,
                font=ctk.CTkFont(size=13, weight="bold"),
                anchor="w",
                height=40,
                corner_radius=8,
                fg_color="transparent",
                text_color=COLORS["text_muted"],
                hover_color="#1e293b",
                command=lambda k=key: self.navigate_to(k),
            )
            btn.pack(fill="x", pady=4)
            self.nav_buttons[key] = btn

        # Sidebar Footer: System Status
        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=20)

        offline_tag = ctk.CTkLabel(
            footer,
            text="🔒 100% Offline Mode",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#10b981",
        )
        offline_tag.pack(anchor="w")

        ver_tag = ctk.CTkLabel(
            footer,
            text=f"v{APP_VERSION} • Windows Local",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_muted"],
        )
        ver_tag.pack(anchor="w")

    def _build_content_area(self) -> None:
        """Create main container for dynamically switched views."""
        self.content_container = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.content_container.pack(side="right", fill="both", expand=True)

        # Lazy instantiated views
        self.dashboard_view = DashboardView(self.content_container, app=self)
        self.single_view = ImageAnalysisView(self.content_container, app=self)
        self.batch_view = BatchAnalysisView(self.content_container, app=self)
        self.history_view = HistoryView(self.content_container, app=self)
        self.settings_view = SettingsView(self.content_container, app=self)

        self.views = {
            "dashboard": self.dashboard_view,
            "single": self.single_view,
            "batch": self.batch_view,
            "history": self.history_view,
            "settings": self.settings_view,
        }

    def navigate_to(self, view_name: str) -> None:
        """Switch active view and update sidebar button styling."""
        if view_name not in self.views:
            return

        # Hide all views
        for v in self.views.values():
            v.pack_forget()

        # Show target view
        target_view = self.views[view_name]
        target_view.pack(fill="both", expand=True)

        # Update button highlights
        for key, btn in self.nav_buttons.items():
            if key == view_name:
                btn.configure(
                    fg_color=COLORS["primary"],
                    text_color="#ffffff",
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=COLORS["text_muted"],
                )

        # Trigger view-specific refreshes
        if view_name == "dashboard":
            self.dashboard_view.refresh()
        elif view_name == "history":
            self.history_view.refresh()
        elif view_name == "settings":
            self.settings_view.refresh_models_list()

    def on_close(self) -> None:
        """Safely unload models and exit application."""
        logger.info("Application shutting down...")
        try:
            ClassifierRegistry.unload_active()
        except Exception:
            pass
        self.destroy()
