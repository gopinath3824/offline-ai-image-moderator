"""Reusable UI widgets and styling components for Offline AI Image Moderator.

Provides modern cards, stat counters, color-coded verdict badges, category score bars,
and responsive preview containers built on CustomTkinter.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any, Callable, Dict, List, Optional, Tuple

import customtkinter as ctk
from PIL import Image  # type: ignore

from moderation.scoring import Verdict


# Premium Modern Theme Color Tokens
COLORS = {
    "bg_dark": "#0f172a",       # Slate 900
    "card_dark": "#1e293b",     # Slate 800
    "card_hover": "#334155",    # Slate 700
    "border_dark": "#334155",   # Slate 700
    "primary": "#3b82f6",       # Blue 500
    "primary_hover": "#2563eb", # Blue 600
    "accent": "#6366f1",        # Indigo 500
    "safe": "#10b981",          # Emerald 500
    "safe_soft": "#064e3b",     # Emerald 900
    "safe_text": "#a7f3d0",     # Emerald 200
    "review": "#f59e0b",        # Amber 500
    "review_soft": "#78350f",   # Amber 900
    "review_text": "#fde68a",   # Amber 200
    "flagged": "#ef4444",       # Red 500
    "flagged_soft": "#7f1d1d",  # Red 900
    "flagged_text": "#fecaca",  # Red 200
    "neutral": "#64748b",       # Slate 500
    "text_main": "#f8fafc",     # Slate 50
    "text_muted": "#94a3b8",    # Slate 400
}


class CardFrame(ctk.CTkFrame):
    """Modern rounded container card with subtle border and elevation."""

    def __init__(
        self,
        master: Any,
        fg_color: Optional[str] = None,
        border_color: Optional[str] = None,
        corner_radius: int = 12,
        border_width: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            master,
            fg_color=fg_color or COLORS["card_dark"],
            border_color=border_color or COLORS["border_dark"],
            corner_radius=corner_radius,
            border_width=border_width,
            **kwargs,
        )


class StatCard(CardFrame):
    """KPI summary widget displaying a prominent metric, icon/title, and subtitle."""

    def __init__(
        self,
        master: Any,
        title: str,
        value: str | int = "0",
        subtitle: str = "",
        accent_color: str = COLORS["primary"],
        width: int = 180,
        height: int = 110,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, width=width, height=height, **kwargs)
        self.pack_propagate(False)

        self.title_label = ctk.CTkLabel(
            self,
            text=title.upper(),
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        self.title_label.pack(anchor="w", padx=16, pady=(14, 2))

        self.value_label = ctk.CTkLabel(
            self,
            text=str(value),
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=accent_color,
        )
        self.value_label.pack(anchor="w", padx=16, pady=0)

        self.sub_label = ctk.CTkLabel(
            self,
            text=subtitle,
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        )
        self.sub_label.pack(anchor="w", padx=16, pady=(2, 10))

    def update_value(self, value: str | int, subtitle: Optional[str] = None) -> None:
        """Update stat value and optional subtext."""
        self.value_label.configure(text=str(value))
        if subtitle is not None:
            self.sub_label.configure(text=subtitle)


class VerdictBadge(ctk.CTkFrame):
    """Pill badge showing color-coded moderation status."""

    def __init__(
        self,
        master: Any,
        verdict: str = Verdict.LIKELY_SAFE.value,
        font_size: int = 13,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, corner_radius=16, border_width=1, **kwargs)
        self.font_size = font_size
        self.label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=font_size, weight="bold"),
        )
        self.label.pack(padx=14, pady=4)
        self.set_verdict(verdict)

    def set_verdict(self, verdict: str) -> None:
        """Update badge appearance according to moderation outcome."""
        v = (verdict or "").strip()
        if v == Verdict.LIKELY_SAFE.value or v.lower() == "safe":
            bg = COLORS["safe_soft"]
            fg = COLORS["safe_text"]
            border = COLORS["safe"]
            icon = "✓  "
            text = "Likely Safe"
        elif v == Verdict.NEEDS_REVIEW.value or "review" in v.lower():
            bg = COLORS["review_soft"]
            fg = COLORS["review_text"]
            border = COLORS["review"]
            icon = "⚠  "
            text = "Needs Review"
        elif v == Verdict.FLAGGED.value or "flag" in v.lower():
            bg = COLORS["flagged_soft"]
            fg = COLORS["flagged_text"]
            border = COLORS["flagged"]
            icon = "✕  "
            text = "Flagged Unsafe"
        elif v == "Processing...":
            bg = COLORS["card_dark"]
            fg = COLORS["primary"]
            border = COLORS["primary"]
            icon = "⏳  "
            text = "Processing..."
        else:
            bg = COLORS["card_dark"]
            fg = COLORS["text_muted"]
            border = COLORS["border_dark"]
            icon = "•  "
            text = v or "Pending"

        self.configure(fg_color=bg, border_color=border)
        self.label.configure(text=f"{icon}{text}", text_color=fg)


class ScoreProgressBar(ctk.CTkFrame):
    """Category confidence bar displaying name, percentage, and color-coded meter."""

    def __init__(
        self,
        master: Any,
        category_name: str,
        display_name: str,
        score: float = 0.0,
        flag_thresh: float = 0.70,
        review_thresh: float = 0.40,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self.category_name = category_name
        self.display_name = display_name
        self.flag_thresh = flag_thresh
        self.review_thresh = review_thresh

        # Header row
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(2, 3))

        self.name_label = ctk.CTkLabel(
            self.header_frame,
            text=display_name,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_main"],
        )
        self.name_label.pack(side="left")

        self.score_label = ctk.CTkLabel(
            self.header_frame,
            text="0.0%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        self.score_label.pack(side="right")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            self,
            height=10,
            corner_radius=5,
            fg_color="#334155",
            progress_color=COLORS["safe"],
        )
        self.progress_bar.pack(fill="x", pady=(0, 6))
        self.set_score(score)

    def set_score(self, score: float) -> None:
        """Update progress value and color code based on threshold."""
        clamped = max(0.0, min(1.0, float(score)))
        self.score_label.configure(text=f"{clamped * 100:.1f}%")
        self.progress_bar.set(clamped)

        if clamped >= self.flag_thresh:
            self.progress_bar.configure(progress_color=COLORS["flagged"])
            self.score_label.configure(text_color=COLORS["flagged"])
        elif clamped >= self.review_thresh:
            self.progress_bar.configure(progress_color=COLORS["review"])
            self.score_label.configure(text_color=COLORS["review"])
        else:
            self.progress_bar.configure(progress_color=COLORS["safe"])
            self.score_label.configure(text_color=COLORS["safe"])


class ImagePreviewBox(CardFrame):
    """Dedicated responsive image preview container with letterbox fit and placeholder."""

    def __init__(
        self,
        master: Any,
        width: int = 420,
        height: int = 380,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, width=width, height=height, **kwargs)
        self.pack_propagate(False)
        self.max_preview_width = width - 24
        self.max_preview_height = height - 24

        self.placeholder_label = ctk.CTkLabel(
            self,
            text="📁 No Image Selected\n\nChoose an image or drop a file to begin moderation",
            font=ctk.CTkFont(size=14),
            text_color=COLORS["text_muted"],
            justify="center",
        )
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor="center")

        self.image_label = ctk.CTkLabel(self, text="")
        self._ctk_image: Optional[ctk.CTkImage] = None

    def display_image(self, pil_image: Image.Image) -> None:
        """Display PIL Image scaled while preserving aspect ratio."""
        self.placeholder_label.place_forget()

        # Compute aspect ratio fit
        w, h = pil_image.size
        ratio = min(self.max_preview_width / max(1, w), self.max_preview_height / max(1, h))
        target_w = max(1, int(w * ratio))
        target_h = max(1, int(h * ratio))

        self._ctk_image = ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=(target_w, target_h),
        )
        self.image_label.configure(image=self._ctk_image)
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

    def clear(self) -> None:
        """Clear image and restore placeholder."""
        self.image_label.place_forget()
        self._ctk_image = None
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
