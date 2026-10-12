"""Tests for scoring engine, thresholds, verdict determination, and disclaimers."""

import pytest
from moderation.scoring import ScoringEngine, Verdict, STANDARD_DISCLAIMER


def test_verdict_likely_safe() -> None:
    """When all scores are below review threshold, verdict is Likely Safe."""
    scores = {"nudity": 0.12, "violence": 0.05, "gore": 0.08}
    res = ScoringEngine.evaluate(
        scores,
        flag_threshold=0.70,
        review_threshold=0.40,
    )

    assert res.verdict == Verdict.LIKELY_SAFE.value
    assert res.requires_review is False
    assert res.highest_category == "nudity"
    assert res.highest_score == 0.12
    assert res.disclaimer == STANDARD_DISCLAIMER


def test_verdict_needs_review() -> None:
    """When a score falls between review and flag thresholds, verdict is Needs Review."""
    scores = {"nudity": 0.55, "violence": 0.10, "gore": 0.05}
    res = ScoringEngine.evaluate(
        scores,
        flag_threshold=0.70,
        review_threshold=0.40,
    )

    assert res.verdict == Verdict.NEEDS_REVIEW.value
    assert res.requires_review is True
    assert res.highest_category == "nudity"
    assert "uncertainty range" in res.explanation.lower()


def test_verdict_flagged() -> None:
    """When a score exceeds flag threshold, verdict is Flagged."""
    scores = {"nudity": 0.88, "violence": 0.15, "gore": 0.05}
    res = ScoringEngine.evaluate(
        scores,
        flag_threshold=0.70,
        review_threshold=0.40,
    )

    assert res.verdict == Verdict.FLAGGED.value
    assert res.requires_review is True
    assert res.highest_category == "nudity"
    assert res.highest_score == 0.88
    assert "exceeded threshold" in res.explanation.lower()


def test_disabled_category_is_ignored() -> None:
    """Categories omitted from enabled_categories list should not trigger flags."""
    scores = {"nudity": 0.95, "violence": 0.10}
    # User disabled nudity in settings
    res = ScoringEngine.evaluate(
        scores,
        flag_threshold=0.70,
        review_threshold=0.40,
        enabled_categories=["violence"],
    )

    # Since nudity is disabled, violence (0.10) is the only active signal -> Likely Safe!
    assert res.verdict == Verdict.LIKELY_SAFE.value
    assert res.highest_category == "violence"
    assert "nudity" not in res.category_scores


def test_custom_thresholds() -> None:
    """Test custom strict or loose threshold settings."""
    scores = {"violence": 0.50}
    # Strict setting: flag at 0.45
    strict_res = ScoringEngine.evaluate(scores, flag_threshold=0.45, review_threshold=0.30)
    assert strict_res.verdict == Verdict.FLAGGED.value

    # Loose setting: flag at 0.80, review at 0.60
    loose_res = ScoringEngine.evaluate(scores, flag_threshold=0.80, review_threshold=0.60)
    assert loose_res.verdict == Verdict.LIKELY_SAFE.value
