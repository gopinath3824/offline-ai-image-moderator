"""Settings and Model Management view for Offline AI Image Moderator.

Configures thresholds, category toggles, model selections, hardware devices,
file limits, scan history storage, and provides model download/removal tools.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import customtkinter as ctk  # type: ignore
from tkinter import messagebox

from config import CATEGORY_DISPLAY_NAMES, MODEL_CATALOG, AppConfig
from moderation.model_manager import ModelManager
from ui.components import COLORS, CardFrame
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from ui.app import ModeratorApp

logger = get_logger("settings_view")


class SettingsView(ctk.CTkScrollableFrame):
    """Configuration and model management screen."""

    def __init__(
        self,
        master: Any,
        app: ModeratorApp,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.app = app
        self.config = app.config
        self.model_manager = app.model_manager
        self.is_downloading = False

        self._build_layout()

    def _build_layout(self) -> None:
        """Construct settings sections."""
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(16, 12))

        title_lbl = ctk.CTkLabel(
            header,
            text="Application Settings & Model Management",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_main"],
        )
        title_lbl.pack(side="left")

        save_btn = ctk.CTkButton(
            header,
            text="💾 Save Settings",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            height=32,
            command=self.save_settings,
        )
        save_btn.pack(side="right")

        # Section 1: Thresholds Configuration
        self._build_thresholds_section()

        # Section 2: Active Model & Inference Hardware
        self._build_model_and_hardware_section()

        # Section 3: Enabled Categories
        self._build_categories_section()

        # Section 4: Privacy, History & File Limits
        self._build_privacy_limits_section()

        # Section 5: Offline Model Storage & Download Manager
        self._build_model_manager_section()

    def _build_thresholds_section(self) -> None:
        """Sliders for configuring flag and manual review thresholds."""
        card = CardFrame(self)
        card.pack(fill="x", padx=16, pady=(0, 12))

        title = ctk.CTkLabel(
            card,
            text="🎯 MODERATION THRESHOLDS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        title.pack(anchor="w", padx=20, pady=(16, 4))

        desc = ctk.CTkLabel(
            card,
            text=(
                "Scores below the Review Threshold are marked 'Likely Safe'. "
                "Scores between Review and Flag Threshold are marked 'Needs Review'. "
                "Scores equal to or exceeding Flag Threshold are marked 'Flagged Unsafe'."
            ),
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
            wraplength=700,
            justify="left",
        )
        desc.pack(anchor="w", padx=20, pady=(0, 12))

        # Flag threshold slider
        f_row = ctk.CTkFrame(card, fg_color="transparent")
        f_row.pack(fill="x", padx=20, pady=4)

        f_label_frame = ctk.CTkFrame(f_row, fg_color="transparent")
        f_label_frame.pack(fill="x")

        ctk.CTkLabel(
            f_label_frame,
            text="Flag Threshold (Unsafe Trigger)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_main"],
        ).pack(side="left")

        self.flag_val_lbl = ctk.CTkLabel(
            f_label_frame,
            text=f"{self.config.flag_threshold * 100:.0f}%",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["flagged"],
        )
        self.flag_val_lbl.pack(side="right")

        self.flag_slider = ctk.CTkSlider(
            f_row,
            from_=0.10,
            to=1.00,
            number_of_steps=90,
            command=self._on_flag_slider_changed,
        )
        self.flag_slider.set(self.config.flag_threshold)
        self.flag_slider.pack(fill="x", pady=(4, 12))

        # Review threshold slider
        r_row = ctk.CTkFrame(card, fg_color="transparent")
        r_row.pack(fill="x", padx=20, pady=4)

        r_label_frame = ctk.CTkFrame(r_row, fg_color="transparent")
        r_label_frame.pack(fill="x")

        ctk.CTkLabel(
            r_label_frame,
            text="Review Threshold (Uncertainty Margin)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_main"],
        ).pack(side="left")

        self.rev_val_lbl = ctk.CTkLabel(
            r_label_frame,
            text=f"{self.config.review_threshold * 100:.0f}%",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["review"],
        )
        self.rev_val_lbl.pack(side="right")

        self.rev_slider = ctk.CTkSlider(
            r_row,
            from_=0.05,
            to=0.90,
            number_of_steps=85,
            command=self._on_rev_slider_changed,
        )
        self.rev_slider.set(self.config.review_threshold)
        self.rev_slider.pack(fill="x", pady=(4, 16))

    def _on_flag_slider_changed(self, val: float) -> None:
        """Handle flag slider movement."""
        self.flag_val_lbl.configure(text=f"{val * 100:.0f}%")
        # Ensure flag slider stays >= review slider
        if val < self.rev_slider.get():
            self.rev_slider.set(val)
            self.rev_val_lbl.configure(text=f"{val * 100:.0f}%")

    def _on_rev_slider_changed(self, val: float) -> None:
        """Handle review slider movement."""
        self.rev_val_lbl.configure(text=f"{val * 100:.0f}%")
        # Ensure review slider stays <= flag slider
        if val > self.flag_slider.get():
            self.flag_slider.set(val)
            self.flag_val_lbl.configure(text=f"{val * 100:.0f}%")

    def _build_model_and_hardware_section(self) -> None:
        """Model selection and CPU/GPU hardware preference."""
        card = CardFrame(self)
        card.pack(fill="x", padx=16, pady=(0, 12))

        title = ctk.CTkLabel(
            card,
            text="🧠 MODEL SELECTION & INFERENCE HARDWARE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        title.pack(anchor="w", padx=20, pady=(16, 10))

        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(0, 16))
        row.grid_columnconfigure((0, 1), weight=1)

        # Model dropdown
        m_frame = ctk.CTkFrame(row, fg_color="transparent")
        m_frame.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        ctk.CTkLabel(
            m_frame,
            text="Active Moderation Model",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_main"],
        ).pack(anchor="w", pady=(0, 4))

        model_names = [f"{m.id}: {m.name}" for m in MODEL_CATALOG.values()]
        active_display = next(
            (f"{m.id}: {m.name}" for m in MODEL_CATALOG.values() if m.id == self.config.active_model_id),
            model_names[0],
        )

        self.model_combo = ctk.CTkComboBox(
            m_frame,
            values=model_names,
            height=34,
            command=self._on_model_selected,
        )
        self.model_combo.set(active_display)
        self.model_combo.pack(fill="x")

        # Device selection
        d_frame = ctk.CTkFrame(row, fg_color="transparent")
        d_frame.grid(row=0, column=1, sticky="ew", padx=(10, 0))

        ctk.CTkLabel(
            d_frame,
            text="Inference Device Preference",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_main"],
        ).pack(anchor="w", pady=(0, 4))

        self.device_combo = ctk.CTkComboBox(
            d_frame,
            values=["cpu", "cuda", "auto"],
            height=34,
        )
        self.device_combo.set(self.config.device_preference)
        self.device_combo.pack(fill="x")

    def _on_model_selected(self, choice: str) -> None:
        """Update active model ID when dropdown choice changes."""
        model_id = choice.split(":")[0].strip()
        self.config.active_model_id = model_id

    def _build_categories_section(self) -> None:
        """Toggles for activating specific moderation categories."""
        card = CardFrame(self)
        card.pack(fill="x", padx=16, pady=(0, 12))

        title = ctk.CTkLabel(
            card,
            text="🏷️ ACTIVE MODERATION CATEGORIES",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        title.pack(anchor="w", padx=20, pady=(16, 6))

        desc = ctk.CTkLabel(
            card,
            text="Disabled categories will be ignored when calculating final verdicts.",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
        )
        desc.pack(anchor="w", padx=20, pady=(0, 10))

        grid_frame = ctk.CTkFrame(card, fg_color="transparent")
        grid_frame.pack(fill="x", padx=20, pady=(0, 16))
        grid_frame.grid_columnconfigure((0, 1), weight=1)

        self.cat_vars: Dict[str, ctk.BooleanVar] = {}
        for idx, (cat_id, disp_name) in enumerate(CATEGORY_DISPLAY_NAMES.items()):
            var = ctk.BooleanVar(value=(cat_id in self.config.enabled_categories))
            self.cat_vars[cat_id] = var

            chk = ctk.CTkCheckBox(
                grid_frame,
                text=disp_name,
                variable=var,
                font=ctk.CTkFont(size=12),
                fg_color=COLORS["primary"],
                hover_color=COLORS["primary_hover"],
            )
            r = idx // 2
            c = idx % 2
            chk.grid(row=r, column=c, sticky="w", pady=6, padx=8)

    def _build_privacy_limits_section(self) -> None:
        """Scan history toggle, maximum dimensions, and max file size limits."""
        card = CardFrame(self)
        card.pack(fill="x", padx=16, pady=(0, 12))

        title = ctk.CTkLabel(
            card,
            text="⚙️ STORAGE, LIMITS & PRIVACY",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        title.pack(anchor="w", padx=20, pady=(16, 10))

        # History toggle switch
        self.history_var = ctk.BooleanVar(value=self.config.enable_history)
        h_switch = ctk.CTkSwitch(
            card,
            text="Enable Local Scan History (Stores metadata and scores in local SQLite database)",
            variable=self.history_var,
            font=ctk.CTkFont(size=13),
            fg_color="#334155",
            progress_color=COLORS["primary"],
        )
        h_switch.pack(anchor="w", padx=20, pady=(0, 12))

        # Limits row
        lim_row = ctk.CTkFrame(card, fg_color="transparent")
        lim_row.pack(fill="x", padx=20, pady=(0, 16))
        lim_row.grid_columnconfigure((0, 1), weight=1)

        # Max file size
        fsize_box = ctk.CTkFrame(lim_row, fg_color="transparent")
        fsize_box.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        ctk.CTkLabel(
            fsize_box,
            text="Max File Size (MB)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_main"],
        ).pack(anchor="w", pady=(0, 2))
        self.max_size_entry = ctk.CTkEntry(fsize_box, height=32)
        self.max_size_entry.insert(0, str(self.config.max_file_size_mb))
        self.max_size_entry.pack(fill="x")

        # Max dimension
        dim_box = ctk.CTkFrame(lim_row, fg_color="transparent")
        dim_box.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        ctk.CTkLabel(
            dim_box,
            text="Max Image Dimension (px)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_main"],
        ).pack(anchor="w", pady=(0, 2))
        self.max_dim_entry = ctk.CTkEntry(dim_box, height=32)
        self.max_dim_entry.insert(0, str(self.config.max_image_dimension))
        self.max_dim_entry.pack(fill="x")

    def _build_model_manager_section(self) -> None:
        """Catalog of local models, offline status, and explicit download controls."""
        card = CardFrame(self)
        card.pack(fill="x", padx=16, pady=(0, 20))

        title = ctk.CTkLabel(
            card,
            text="📦 OFFLINE MODEL MANAGER",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        title.pack(anchor="w", padx=20, pady=(16, 4))

        notice = ctk.CTkLabel(
            card,
            text=(
                "Models run 100% offline. An active internet connection is ONLY required once "
                "during the initial download from Hugging Face Hub. After downloading, no external "
                "network calls are ever made."
            ),
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
            wraplength=700,
            justify="left",
        )
        notice.pack(anchor="w", padx=20, pady=(0, 12))

        # Model cards container
        self.models_list_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.models_list_frame.pack(fill="x", padx=20, pady=(0, 16))

        self.refresh_models_list()

    def refresh_models_list(self) -> None:
        """Render all models in catalog with install status, download and delete buttons."""
        for child in self.models_list_frame.winfo_children():
            child.destroy()

        models_status = self.model_manager.list_all_models_status()

        for info in models_status:
            m_id = info["id"]
            row_card = ctk.CTkFrame(self.models_list_frame, fg_color="#182234", corner_radius=8)
            row_card.pack(fill="x", pady=4)

            # Left side: Name, description, size
            info_box = ctk.CTkFrame(row_card, fg_color="transparent")
            info_box.pack(side="left", fill="both", expand=True, padx=14, pady=10)

            name_lbl = ctk.CTkLabel(
                info_box,
                text=info["name"],
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=COLORS["text_main"],
            )
            name_lbl.pack(anchor="w")

            desc_lbl = ctk.CTkLabel(
                info_box,
                text=info["description"],
                font=ctk.CTkFont(size=11),
                text_color=COLORS["text_muted"],
                wraplength=480,
                justify="left",
            )
            desc_lbl.pack(anchor="w", pady=(2, 4))

            # Status pill
            status_txt = "✓ Installed & Ready" if info["installed"] else f"⚠️ Missing ({info['approx_size_mb']} MB)"
            status_fg = "#064e3b" if info["installed"] else "#78350f"
            status_text_color = "#6ee7b7" if info["installed"] else "#fde68a"

            status_pill = ctk.CTkLabel(
                info_box,
                text=f"{status_txt} • Disk: {info['installed_size_display']} • License: {info['license']}",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=status_text_color,
                fg_color=status_fg,
                corner_radius=6,
                padx=8,
                pady=2,
            )
            status_pill.pack(anchor="w")

            # Right side: Action buttons
            actions_box = ctk.CTkFrame(row_card, fg_color="transparent")
            actions_box.pack(side="right", padx=14, pady=10)

            if m_id != "synthetic_diagnostic":
                if not info["installed"]:
                    dl_btn = ctk.CTkButton(
                        actions_box,
                        text="⬇ Download Model",
                        font=ctk.CTkFont(size=11, weight="bold"),
                        fg_color=COLORS["primary"],
                        hover_color=COLORS["primary_hover"],
                        width=120,
                        height=30,
                        command=lambda mid=m_id: self.download_model_async(mid),
                    )
                    dl_btn.pack(side="right")
                else:
                    del_btn = ctk.CTkButton(
                        actions_box,
                        text="🗑 Remove Weights",
                        font=ctk.CTkFont(size=11),
                        fg_color="#475569",
                        hover_color="#dc2626",
                        width=120,
                        height=30,
                        command=lambda mid=m_id: self.delete_model(mid),
                    )
                    del_btn.pack(side="right")
            else:
                diag_lbl = ctk.CTkLabel(
                    actions_box,
                    text="Built-in Engine",
                    font=ctk.CTkFont(size=11),
                    text_color=COLORS["text_muted"],
                )
                diag_lbl.pack(side="right", padx=10)

    def download_model_async(self, model_id: str) -> None:
        """Start model download thread with progress notification."""
        if self.is_downloading:
            messagebox.showinfo("Download in Progress", "Another model download is currently running.")
            return

        meta = MODEL_CATALOG.get(model_id)
        if not meta:
            return

        model_name = meta.name

        confirm = messagebox.askyesno(
            "Confirm Model Download",
            f"Download '{model_name}' (~{meta.size_mb} MB)?\n\n"
            "This will connect to Hugging Face Hub to download model weights. "
            "Once downloaded, inference runs 100% offline.\n\nProceed?",
        )
        if not confirm:
            return

        self.is_downloading = True

        # Progress window dialog
        prog_dialog = ctk.CTkToplevel(self)
        prog_dialog.title(f"Downloading {model_name}")
        prog_dialog.geometry("460x180")
        prog_dialog.attributes("-topmost", True)

        dlg_title = ctk.CTkLabel(
            prog_dialog,
            text=f"Downloading {model_name}",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        dlg_title.pack(pady=(20, 8))

        dlg_status = ctk.CTkLabel(
            prog_dialog,
            text="Connecting to Hugging Face Hub...",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
        )
        dlg_status.pack(pady=(0, 12))

        prog_meter = ctk.CTkProgressBar(prog_dialog, width=380, height=10)
        prog_meter.pack(pady=(0, 20))
        prog_meter.set(0.1)

        def callback(fraction: float, message: str) -> None:
            self.after(
                0,
                lambda: (prog_meter.set(fraction), dlg_status.configure(text=message)),
            )

        def worker() -> None:
            try:
                self.model_manager.download_model(model_id, progress_callback=callback)
                self.after(
                    0,
                    lambda: (
                        prog_dialog.destroy(),
                        messagebox.showinfo("Success", f"{model_name} downloaded successfully and ready for offline use!"),
                        self.refresh_models_list(),
                        self.app.dashboard_view.refresh(),
                    ),
                )
            except Exception as err:
                logger.error("Download failed: %s", err)
                self.after(
                    0,
                    lambda: (
                        prog_dialog.destroy(),
                        messagebox.showerror("Download Error", f"Failed to download model:\n{err}"),
                        self.refresh_models_list(),
                    ),
                )
            finally:
                self.is_downloading = False

        threading.Thread(target=worker, daemon=True).start()

    def delete_model(self, model_id: str) -> None:
        """Remove model weights from disk to reclaim space."""
        meta = MODEL_CATALOG.get(model_id)
        if not meta:
            return
        if messagebox.askyesno(
            "Confirm Removal",
            f"Delete local weights for '{meta.name}' to reclaim disk space?",
        ):
            self.model_manager.delete_local_model(model_id)
            self.refresh_models_list()
            self.app.dashboard_view.refresh()
            messagebox.showinfo("Deleted", f"Local weights for '{meta.name}' have been removed.")

    def save_settings(self) -> None:
        """Commit user settings to JSON config on disk."""
        try:
            self.config.flag_threshold = round(float(self.flag_slider.get()), 2)
            self.config.review_threshold = round(float(self.rev_slider.get()), 2)

            # Selected model
            choice = self.model_combo.get()
            self.config.active_model_id = choice.split(":")[0].strip()

            # Device
            self.config.device_preference = self.device_combo.get().strip().lower()

            # Enabled categories
            enabled = [cat for cat, var in self.cat_vars.items() if var.get()]
            if not enabled:
                messagebox.showwarning("Warning", "At least one category should be enabled.")
                return
            self.config.enabled_categories = enabled

            # History
            self.config.enable_history = self.history_var.get()

            # Limits
            try:
                self.config.max_file_size_mb = max(1, int(self.max_size_entry.get().strip()))
                self.config.max_image_dimension = max(500, int(self.max_dim_entry.get().strip()))
            except ValueError:
                messagebox.showwarning("Invalid Limits", "File size and dimension limits must be positive integers.")
                return

            self.config.save()
            messagebox.showinfo("Settings Saved", "Your settings have been saved successfully.")
            self.app.dashboard_view.refresh()

        except Exception as err:
            logger.error("Failed to save settings: %s", err)
            messagebox.showerror("Save Error", f"Failed to save settings: {err}")
