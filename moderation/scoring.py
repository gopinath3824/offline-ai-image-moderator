"""Scoring engine and verdict evaluation for moderation results.

Calculates category scores, determines overall verdict (Likely Safe, Needs Review, Flagged),
and formats explanations with responsible AI disclaimers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from config import CATEGORY_DISPLAY_NAMES


class Verdict(str, Enum):
    """Primary moderation verdict levels."""

    LIKELY_SAFE = "Likely Safe"
    NEEDS_REVIEW = "Needs Review"
    FLAGGED = "Flagged"
    ERROR = "Error"


STANDARD_DISCLAIMER = (
    "AI predictions are automated moderation recommendations and not definitive judgments. "
    "False positives (e.g., artistic or medical imagery) and false negatives can occur. "
    "Always rely on human manual review for sensitive or contentious decisions."
)


@dataclass
class ModerationResult:
    """Standardized result of image content moderation analysis."""

    verdict: str  # Verdict.LIKELY_SAFE, Verdict.NEEDS_REVIEW, Verdict.FLAGGED
    requires_review: bool
    highest_category: str
    highest_score: float
    category_scores: Dict[str, float]
    category_verdicts: Dict[str, str] = field(default_factory=dict)
    model_id: str = ""
    model_name: str = ""
    inference_latency_sec: float = 0.0
    explanation: str = ""
    disclaimer: str = STANDARD_DISCLAIMER
    error_message: Optional[str] = None

    @property
    def highest_category_display(self) -> str:
        """User-friendly name of the highest scoring category."""
        return CATEGORY_DISPLAY_NAMES.get(self.highest_category, self.highest_category.title())

    @property
    def highest_score_pct(self) -> str:
        """Formatted percentage string for top score."""
        return f"{self.highest_score * 100:.1f}%"


class ScoringEngine:
    """Evaluates raw model score distributions against user thresholds."""

    @staticmethod
    def evaluate(
        category_scores: Dict[str, float],
        flag_threshold: float = 0.70,
        review_threshold: float = 0.40,
        enabled_categories: Optional[List[str]] = None,
        model_id: str = "",
        model_name: str = "",
        inference_latency_sec: float = 0.0,
    ) -> ModerationResult:
        """Evaluate raw category scores against configured thresholds.

        Verdict Logic:
        - FLAGGED: Any enabled category score >= flag_threshold.
        - NEEDS_REVIEW: Any enabled category score in [review_threshold, flag_threshold).
        - LIKELY_SAFE: All enabled category scores < review_threshold.
        """
        if review_threshold > flag_threshold:
            # Defensive normalization if user inverted slider bounds
            review_threshold, flag_threshold = flag_threshold, review_threshold

        # Filter scores to only enabled categories if specified
        active_scores: Dict[str, float] = {}
        for cat, score in category_scores.items():
            if enabled_categories is None or cat in enabled_categories:
                active_scores[cat] = float(score)

        # Rule 2: If buttocks_exposed is high without clothing coverage, classify under nudity
        if "buttocks_exposed" in active_scores and active_scores["buttocks_exposed"] >= review_threshold:
            if "nudity" in active_scores:
                active_scores["nudity"] = round(
                    max(active_scores["nudity"], active_scores["buttocks_exposed"]), 4
                )

        category_verdicts: Dict[str, str] = {}
        flagged_categories: List[str] = []
        review_categories: List[str] = []

        for cat, score in active_scores.items():
            if score >= flag_threshold:
                category_verdicts[cat] = Verdict.FLAGGED.value
                flagged_categories.append(cat)
            elif score >= review_threshold:
                category_verdicts[cat] = Verdict.NEEDS_REVIEW.value
                review_categories.append(cat)
            else:
                category_verdicts[cat] = Verdict.LIKELY_SAFE.value

        # Identify highest scoring category
        if active_scores:
            highest_category = max(active_scores, key=active_scores.get)  # type: ignore
            highest_score = active_scores[highest_category]
        else:
            highest_category = "None"
            highest_score = 0.0

        # Determine overall verdict
        if flagged_categories:
            verdict = Verdict.FLAGGED.value
            requires_review = True
            flag_names = [
                CATEGORY_DISPLAY_NAMES.get(c, c)
                + f" ({active_scores[c] * 100:.1f}%)"
                for c in flagged_categories
            ]
            if "buttocks_exposed" in flagged_categories:
                explanation = (
                    f"Flagged for nudity: bare exposed buttocks detected without clothing coverage "
                    f"({active_scores['buttocks_exposed'] * 100:.1f}%). "
                    f"Exceeded flag threshold ({flag_threshold * 100:.0f}%): {', '.join(flag_names)}."
                )
            else:
                explanation = (
                    f"Flagged for potential unsafe content. Exceeded threshold ({flag_threshold * 100:.0f}%): "
                    f"{', '.join(flag_names)}."
                )
        elif review_categories:
            verdict = Verdict.NEEDS_REVIEW.value
            requires_review = True
            rev_names = [
                CATEGORY_DISPLAY_NAMES.get(c, c)
                + f" ({active_scores[c] * 100:.1f}%)"
                for c in review_categories
            ]
            if "partial_nudity" in review_categories:
                explanation = (
                    f"Requires manual review: partially covered buttocks or revealing attire detected "
                    f"({active_scores['partial_nudity'] * 100:.1f}%). Uncertainty range "
                    f"[{review_threshold * 100:.0f}% - {flag_threshold * 100:.0f}%): {', '.join(rev_names)}."
                )
            elif "buttocks_exposed" in review_categories:
                explanation = (
                    f"Requires manual review: borderline gluteal exposure detected "
                    f"({active_scores['buttocks_exposed'] * 100:.1f}%). Falls in uncertainty range "
                    f"[{review_threshold * 100:.0f}% - {flag_threshold * 100:.0f}%)."
                )
            else:
                explanation = (
                    f"Requires manual review. Prediction falls in uncertainty range "
                    f"[{review_threshold * 100:.0f}% - {flag_threshold * 100:.0f}%): {', '.join(rev_names)}."
                )
        else:
            verdict = Verdict.LIKELY_SAFE.value
            requires_review = False
            top_display = (
                f"{CATEGORY_DISPLAY_NAMES.get(highest_category, highest_category)} ({highest_score * 100:.1f}%)"
                if active_scores
                else "None"
            )
            explanation = (
                f"Likely safe. Fully covered clothing / swimwear detected. All active categories remain below "
                f"review threshold ({review_threshold * 100:.0f}%). Top signal: {top_display}."
            )

        return ModerationResult(
            verdict=verdict,
            requires_review=requires_review,
            highest_category=highest_category,
            highest_score=highest_score,
            category_scores=active_scores,
            category_verdicts=category_verdicts,
            model_id=model_id,
            model_name=model_name,
            inference_latency_sec=inference_latency_sec,
            explanation=explanation,
            disclaimer=STANDARD_DISCLAIMER,
        )

    @staticmethod
    def error_result(
        error_message: str,
        model_id: str = "",
        model_name: str = "",
    ) -> ModerationResult:
        """Create a standard error moderation result."""
        return ModerationResult(
            verdict=Verdict.ERROR.value,
            requires_review=True,
            highest_category="Error",
            highest_score=0.0,
            category_scores={},
            category_verdicts={},
            model_id=model_id,
            model_name=model_name,
            inference_latency_sec=0.0,
            explanation=f"Analysis failed: {error_message}",
            disclaimer=STANDARD_DISCLAIMER,
            error_message=error_message,
        )
