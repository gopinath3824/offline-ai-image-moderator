"""Body region and camera view angle detection engine for content moderation.

Analyzes human anatomical regions, viewpoint orientations (front, side, rear),
and gluteal coverage levels (fully clothed, covered swimwear/underwear,
partially covered, fully exposed) to provide granular, explainable nudity moderation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from utils.logging_config import get_logger

logger = get_logger("body_region_detector")


class ViewAngle(str, Enum):
    """Camera viewpoint relative to subject."""

    FRONT = "front"
    SIDE = "side"
    REAR = "rear"
    UNKNOWN = "unknown"


class CoverageCategory(str, Enum):
    """Anatomical clothing coverage level for gluteal / pelvic region."""

    FULLY_COVERED = "fully_covered"
    SWIMWEAR_COVERED = "swimwear_covered"
    PARTIAL_COVERED = "partial_covered"
    EXPOSED = "exposed"


@dataclass
class BodyRegionAnalysis:
    """Detailed anatomical and view-angle analysis results."""

    view_angle: str
    angle_confidence: float
    coverage_category: str
    gluteal_skin_ratio: float
    gluteal_fabric_ratio: float
    buttocks_exposed_score: float
    partial_nudity_score: float
    swimwear_score: float
    clothed_score: float
    explanation: str

    def to_dict(self) -> Dict[str, float]:
        """Return score dictionary suitable for moderation scoring."""
        return {
            "buttocks_exposed": round(self.buttocks_exposed_score, 4),
            "partial_nudity": round(self.partial_nudity_score, 4),
            "swimwear": round(self.swimwear_score, 4),
            "clothed": round(self.clothed_score, 4),
        }


class BodyRegionDetector:
    """Analyzes spatial anatomy, orientation, and clothing coverage in images."""

    @staticmethod
    def _is_skin_pixel_rgb(r: int, g: int, b: int) -> bool:
        """Heuristic skin-tone classifier supporting diverse human complexions.

        Covers fair, olive, warm, tan, and dark skin tones using multi-condition
        RGB and chrominance boundary rules.
        """
        # Rule 1: General Caucasian / East Asian / Fair-to-Medium complexions
        c1 = (
            r > 95
            and g > 40
            and b > 20
            and (max(r, g, b) - min(r, g, b) > 15)
            and abs(r - g) > 15
            and r > g
            and r > b
        )
        if c1:
            return True

        # Rule 2: Deep / Dark / Rich melanin complexions (lower overall brightness, warm balance)
        c2 = (
            r > 50
            and g > 25
            and b > 15
            and r >= g
            and g >= b
            and (r - g >= 8)
            and (r - b >= 15)
            and (max(r, g, b) - min(r, g, b) > 12)
        )
        if c2:
            return True

        # Rule 3: High-key / Sunlit warm skin (beach, direct sunlight)
        c3 = (
            r > 160
            and g > 110
            and b > 80
            and r > g
            and g > b
            and (r - b > 25)
            and abs(r - g) < 70
        )
        return c3

    @staticmethod
    def _is_denim_or_fabric_pixel(r: int, g: int, b: int) -> bool:
        """Identify typical clothing fabrics such as denim jeans, black trousers, or neutral pants."""
        # Denim blue tones
        is_blue_denim = b > r + 15 and b > g + 5 and b > 40
        # Dark black / charcoal / gray fabrics (pants, leggings, skirts)
        is_dark_neutral = max(r, g, b) < 65 and (max(r, g, b) - min(r, g, b) < 18)
        # Khaki / olive / denim wash
        is_khaki_olive = (
            g > b + 15 and r > b + 15 and abs(r - g) < 25 and (r + g + b) < 380
        )
        # White / light gray cotton fabric
        is_white_fabric = (
            min(r, g, b) > 180 and (max(r, g, b) - min(r, g, b) < 15)
        )
        return is_blue_denim or is_dark_neutral or is_khaki_olive or is_white_fabric

    @staticmethod
    def _is_swimwear_pixel(r: int, g: int, b: int) -> bool:
        """Detect synthetic, vibrant, or high-contrast swimwear textile colors."""
        # Bright lycra colors: cyan, magenta, red, bright yellow, neon green
        is_vibrant = (
            (r > 180 and g < 60 and b > 120)  # Magenta / Fuchsia
            or (r < 60 and g > 160 and b > 160)  # Bright Cyan / Aqua
            or (r > 180 and g > 180 and b < 60)  # Bright Yellow
            or (r > 190 and g < 70 and b < 70)  # Bright Red swimwear
            or (r < 60 and g > 180 and b < 80)  # Neon Green
            or (b > 160 and r < 80 and g < 120)  # Deep Royal Blue swimwear
        )
        return is_vibrant

    @classmethod
    def analyze_image(cls, image: Image.Image) -> BodyRegionAnalysis:
        """Perform spatial anatomical evaluation of the image.

        Evaluates:
        1. Vertical region division (Torso: 0-40%, Gluteal/Pelvis: 40-75%, Legs: 75-100%).
        2. Viewpoint angle estimation (rear vs side vs front).
        3. Gluteal region skin ratio vs fabric coverage ratio.
        4. Category scores for exposed buttocks, partial nudity, swimwear, and clothed.
        """
        # Downsample to standardized grid for robust spatial analysis
        target_size = (128, 128)
        thumb = image.resize(target_size, Image.Resampling.BILINEAR).convert("RGB")
        w, h = thumb.size

        # Define vertical anatomical bands
        # Upper body (torso/chest/face): 0 to 40%
        # Gluteal / pelvic region: 40% to 75%
        # Lower extremities (legs): 75% to 100%
        y_gluteal_start = int(h * 0.40)
        y_gluteal_end = int(h * 0.75)

        # Pixel buffers
        pixels = thumb.load()
        if pixels is None:
            # Fallback if image load failed
            return BodyRegionAnalysis(
                view_angle=ViewAngle.UNKNOWN.value,
                angle_confidence=0.5,
                coverage_category=CoverageCategory.FULLY_COVERED.value,
                gluteal_skin_ratio=0.0,
                gluteal_fabric_ratio=1.0,
                buttocks_exposed_score=0.02,
                partial_nudity_score=0.02,
                swimwear_score=0.10,
                clothed_score=0.90,
                explanation="Default fallback: unable to read image pixels.",
            )

        # 1. Evaluate Gluteal Region (Middle-Lower body)
        gluteal_total = 0
        gluteal_skin = 0
        gluteal_fabric = 0
        gluteal_swimwear = 0

        # Sub-bands: Left half vs Right half for bilateral symmetry
        left_skin = 0
        right_skin = 0

        # Central vertical strip (cleft/crease region: 42% to 58% horizontal)
        center_x_start = int(w * 0.42)
        center_x_end = int(w * 0.58)
        center_skin = 0
        center_total = 0

        for y in range(y_gluteal_start, y_gluteal_end):
            for x in range(w):
                r, g, b = pixels[x, y]
                gluteal_total += 1

                is_skin = cls._is_skin_pixel_rgb(r, g, b)
                is_fab = cls._is_denim_or_fabric_pixel(r, g, b)
                is_swim = cls._is_swimwear_pixel(r, g, b)

                if is_skin:
                    gluteal_skin += 1
                    if x < w // 2:
                        left_skin += 1
                    else:
                        right_skin += 1

                if is_fab:
                    gluteal_fabric += 1
                if is_swim:
                    gluteal_swimwear += 1

                if center_x_start <= x <= center_x_end:
                    center_total += 1
                    if is_skin:
                        center_skin += 1

        gluteal_skin_ratio = gluteal_skin / max(1, gluteal_total)
        gluteal_fabric_ratio = gluteal_fabric / max(1, gluteal_total)
        gluteal_swimwear_ratio = gluteal_swimwear / max(1, gluteal_total)

        # 2. Evaluate Upper Body (Torso / Face: 0 to 40%)
        upper_total = 0
        upper_skin = 0
        for y in range(0, y_gluteal_start):
            for x in range(w):
                r, g, b = pixels[x, y]
                upper_total += 1
                if cls._is_skin_pixel_rgb(r, g, b):
                    upper_skin += 1
        upper_skin_ratio = upper_skin / max(1, upper_total)

        # 3. View Angle Estimation
        # Rear-view characteristics:
        # - Bilateral gluteal lobe symmetry (left_skin ≈ right_skin)
        # - High skin in center/lower-mid body, or back curvature
        # - Absence of high frontal facial features (upper skin often uniform back skin or covered hair)
        # Side-view characteristics:
        # - Pronounced horizontal asymmetry (left_skin significantly different from right_skin)
        # Front-view characteristics:
        # - Frontal torso / face features

        diff_lr = abs(left_skin - right_skin) / max(1, (left_skin + right_skin))
        if diff_lr > 0.55:
            view_angle = ViewAngle.SIDE
            angle_conf = min(0.92, 0.60 + diff_lr * 0.35)
        elif upper_skin_ratio < 0.30 and gluteal_skin_ratio > 0.40:
            view_angle = ViewAngle.REAR
            angle_conf = min(0.90, 0.65 + (1.0 - diff_lr) * 0.25)
        elif gluteal_skin_ratio > 0.35 and diff_lr < 0.30:
            view_angle = ViewAngle.REAR
            angle_conf = min(0.88, 0.60 + (1.0 - diff_lr) * 0.25)
        elif upper_skin_ratio > 0.50 and gluteal_skin_ratio < 0.20:
            view_angle = ViewAngle.FRONT
            angle_conf = 0.75
        else:
            view_angle = ViewAngle.REAR if (diff_lr < 0.40) else ViewAngle.SIDE
            angle_conf = 0.65

        # 4. Coverage Classification & Scoring
        # Rule A: Fully Covered Buttocks
        # Fabric ratio is dominant in gluteal region, or skin ratio is very low (< 0.15)
        if gluteal_fabric_ratio > 0.45 or gluteal_skin_ratio < 0.18:
            coverage = CoverageCategory.FULLY_COVERED
            clothed_score = min(0.98, max(0.70, gluteal_fabric_ratio * 1.2 + (1.0 - gluteal_skin_ratio) * 0.4))
            swimwear_score = min(0.35, gluteal_swimwear_ratio * 0.8)
            exposed_score = min(0.12, max(0.01, gluteal_skin_ratio * 0.3))
            partial_score = min(0.15, max(0.02, gluteal_skin_ratio * 0.4))
            explanation = (
                f"Gluteal region is fully covered by clothing fabric "
                f"(Fabric: {gluteal_fabric_ratio * 100:.1f}%, Skin: {gluteal_skin_ratio * 100:.1f}%). "
                f"Angle: {view_angle.value}."
            )

        # Rule B: Swimwear / Underwear Coverage
        # High swimwear synthetic color ratio OR moderate fabric with beach/pool context without bare gluteal exposure
        elif gluteal_swimwear_ratio > 0.15 and gluteal_skin_ratio < 0.55:
            coverage = CoverageCategory.SWIMWEAR_COVERED
            swimwear_score = min(0.95, max(0.65, gluteal_swimwear_ratio * 2.5 + 0.30))
            clothed_score = min(0.70, 0.40 + gluteal_fabric_ratio)
            exposed_score = min(0.20, max(0.02, gluteal_skin_ratio * 0.25))
            partial_score = min(0.35, max(0.05, gluteal_skin_ratio * 0.4))
            explanation = (
                f"Gluteal region covered by swimwear or underwear "
                f"(Swimwear textile: {gluteal_swimwear_ratio * 100:.1f}%, Skin: {gluteal_skin_ratio * 100:.1f}%). "
                f"Classified as covered swimwear (non-nudity)."
            )

        # Rule C: Partial Exposure (Cheeky bottoms, thong, revealing cutout)
        # Moderate skin ratio (0.35 to 0.65) with small fabric strip detected
        elif (0.30 <= gluteal_skin_ratio <= 0.65) and (gluteal_fabric_ratio > 0.08 or gluteal_swimwear_ratio > 0.05):
            coverage = CoverageCategory.PARTIAL_COVERED
            partial_score = min(0.92, max(0.55, gluteal_skin_ratio * 1.1))
            exposed_score = min(0.65, max(0.35, gluteal_skin_ratio * 0.9))
            swimwear_score = min(0.50, max(0.20, (gluteal_swimwear_ratio + gluteal_fabric_ratio) * 1.2))
            clothed_score = 0.15
            explanation = (
                f"Partially covered buttocks detected (cheeky cut / minimal coverage). "
                f"Skin ratio: {gluteal_skin_ratio * 100:.1f}%, Angle: {view_angle.value}."
            )

        # Rule D: Fully Exposed Buttocks
        # High skin ratio in gluteal region (> 0.60) without opaque clothing fabric
        else:
            coverage = CoverageCategory.EXPOSED
            exposed_score = min(0.98, max(0.72, gluteal_skin_ratio * 1.15))
            partial_score = min(0.60, 0.30 + (1.0 - gluteal_skin_ratio) * 0.5)
            swimwear_score = min(0.15, gluteal_swimwear_ratio * 0.5)
            clothed_score = min(0.10, gluteal_fabric_ratio * 0.3)
            explanation = (
                f"Bare, exposed buttocks detected without clothing coverage "
                f"(Gluteal skin: {gluteal_skin_ratio * 100:.1f}%). "
                f"Angle: {view_angle.value}."
            )

        return BodyRegionAnalysis(
            view_angle=view_angle.value,
            angle_confidence=round(angle_conf, 3),
            coverage_category=coverage.value,
            gluteal_skin_ratio=round(gluteal_skin_ratio, 4),
            gluteal_fabric_ratio=round(gluteal_fabric_ratio, 4),
            buttocks_exposed_score=round(exposed_score, 4),
            partial_nudity_score=round(partial_score, 4),
            swimwear_score=round(swimwear_score, 4),
            clothed_score=round(clothed_score, 4),
            explanation=explanation,
        )
