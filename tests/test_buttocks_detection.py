"""Tests for multi-angle buttocks exposure detection, swimwear handling, and false-positive prevention.

Requirements tested:
1. Dedicated category: buttocks_exposed.
2. Detection rules:
   - Fully covered buttocks -> classified as safe clothing (not nudity).
   - Buttocks exposed without clothing coverage -> classified as nudity.
   - Buttocks partially covered -> classified as partial_nudity (needs_review).
   - Swimwear and underwear coverage -> not automatically classified as nudity.
3. Multi-angle support: Front, side, and rear views evaluated without front-facing bias.
4. Confidence handling: Return category and confidence score; return needs_review in uncertainty margin.
5. Avoid false positives: Do not classify ordinary clothing or swimwear as nudity solely due to body shape or angle.
6. Metric reporting: Assert precision, recall, and false-positive rates.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import pytest
from PIL import Image, ImageDraw

from config import AppConfig
from moderation.body_region_detector import BodyRegionDetector, CoverageCategory, ViewAngle
from moderation.classifier import SyntheticDiagnosticClassifier
from moderation.scoring import ScoringEngine, Verdict


def _create_synthetic_test_image(
    view_angle: str = "rear",
    coverage: str = "clothed",
    skin_tone: Tuple[int, int, int] = (210, 155, 120),  # Warm fair/medium skin
    fabric_color: Tuple[int, int, int] = (30, 60, 130),  # Denim blue
) -> Image.Image:
    """Helper to generate standardized synthetic images for multi-angle moderation testing.

    Anatomical layout (128x128):
    - Upper body (y: 0 - 50): torso / back / front
    - Gluteal / pelvic region (y: 50 - 95): evaluated for clothing vs exposed skin
    - Lower legs (y: 95 - 128)
    """
    img = Image.new("RGB", (128, 128), color=(240, 240, 245))  # Neutral background
    draw = ImageDraw.Draw(img)

    if view_angle == "rear":
        # Rear view: bilateral symmetry, back skin or shirt
        # Upper body (torso): skin or shirt
        draw.rectangle([34, 15, 94, 52], fill=skin_tone)

        if coverage == "clothed":
            # Gluteal region covered with denim jeans / dark fabric
            draw.rectangle([30, 52, 98, 98], fill=fabric_color)
            # Legs covered in pants
            draw.rectangle([32, 98, 60, 125], fill=fabric_color)
            draw.rectangle([68, 98, 96, 125], fill=fabric_color)

        elif coverage == "swimwear_covered":
            # Gluteal region covered with standard full-coverage swimwear (e.g. bright cyan/magenta lycra)
            swim_color = (0, 190, 200)  # Bright cyan swimwear
            draw.rectangle([34, 52, 94, 88], fill=swim_color)
            # Legs are bare skin
            draw.rectangle([36, 88, 60, 125], fill=skin_tone)
            draw.rectangle([68, 88, 92, 125], fill=skin_tone)

        elif coverage == "partial_covered":
            # Partial coverage: thong/cheeky bottom - thin central strap with exposed gluteal skin on sides
            # Bare gluteal lobes
            draw.ellipse([32, 52, 68, 94], fill=skin_tone)
            draw.ellipse([60, 52, 96, 94], fill=skin_tone)
            # Thin fabric strap in middle
            draw.rectangle([60, 52, 68, 90], fill=(210, 20, 60))
            # Legs
            draw.rectangle([36, 94, 58, 125], fill=skin_tone)
            draw.rectangle([70, 94, 92, 125], fill=skin_tone)

        elif coverage == "exposed":
            # Fully exposed buttocks: bare gluteal lobes with gluteal cleft shadow
            draw.ellipse([32, 50, 68, 95], fill=skin_tone)
            draw.ellipse([60, 50, 96, 95], fill=skin_tone)
            # Gluteal crease shadow
            draw.line([(64, 55), (64, 90)], fill=(160, 105, 75), width=2)
            # Bare legs
            draw.rectangle([36, 95, 58, 125], fill=skin_tone)
            draw.rectangle([70, 95, 92, 125], fill=skin_tone)

    elif view_angle == "side":
        # Side profile: asymmetric horizontal distribution
        if coverage == "clothed":
            draw.rectangle([45, 15, 80, 52], fill=skin_tone)
            # Clothed lower body
            draw.polygon([(40, 52), (85, 52), (82, 125), (45, 125)], fill=fabric_color)
        elif coverage == "swimwear_covered":
            draw.rectangle([45, 15, 80, 52], fill=skin_tone)
            draw.polygon([(42, 52), (84, 52), (80, 85), (46, 85)], fill=(0, 190, 200))
            draw.rectangle([48, 85, 76, 125], fill=skin_tone)
        elif coverage == "exposed":
            draw.rectangle([45, 15, 80, 52], fill=skin_tone)
            # Lateral gluteal curve with exposed skin
            draw.ellipse([38, 50, 85, 95], fill=skin_tone)
            draw.rectangle([48, 95, 75, 125], fill=skin_tone)

    elif view_angle == "front":
        # Front view: anterior torso and legs
        draw.rectangle([34, 15, 94, 52], fill=skin_tone)
        if coverage == "clothed":
            draw.rectangle([30, 52, 98, 125], fill=fabric_color)
        elif coverage == "swimwear_covered":
            draw.rectangle([34, 52, 94, 88], fill=(0, 190, 200))
            draw.rectangle([36, 88, 60, 125], fill=skin_tone)
            draw.rectangle([68, 88, 92, 125], fill=skin_tone)
        elif coverage == "exposed":
            draw.rectangle([34, 52, 94, 125], fill=skin_tone)

    return img


def test_rear_view_fully_covered_jeans_is_safe() -> None:
    """Fully covered buttocks (rear view in jeans) must classify as Safe (NOT nudity)."""
    img = _create_synthetic_test_image(view_angle="rear", coverage="clothed")
    analysis = BodyRegionDetector.analyze_image(img)

    assert analysis.coverage_category == CoverageCategory.FULLY_COVERED.value
    assert analysis.buttocks_exposed_score < 0.20
    assert analysis.clothed_score > 0.65

    # Moderation scoring evaluation
    classifier = SyntheticDiagnosticClassifier()
    scores = classifier.predict(img)

    result = ScoringEngine.evaluate(scores, flag_threshold=0.70, review_threshold=0.40)
    assert result.verdict == Verdict.LIKELY_SAFE.value
    assert result.requires_review is False
    assert scores["buttocks_exposed"] < 0.20
    assert scores["nudity"] < 0.20


def test_rear_view_swimwear_coverage_is_not_classified_as_nudity() -> None:
    """Swimwear coverage rear view must NOT automatically be classified as nudity."""
    img = _create_synthetic_test_image(view_angle="rear", coverage="swimwear_covered")
    analysis = BodyRegionDetector.analyze_image(img)

    assert analysis.coverage_category == CoverageCategory.SWIMWEAR_COVERED.value
    assert analysis.buttocks_exposed_score < 0.25
    assert analysis.swimwear_score > 0.60

    classifier = SyntheticDiagnosticClassifier()
    scores = classifier.predict(img)

    result = ScoringEngine.evaluate(scores, flag_threshold=0.70, review_threshold=0.40)
    # Must remain safe and below flag threshold
    assert result.verdict != Verdict.FLAGGED.value
    assert scores["buttocks_exposed"] < 0.35
    assert scores["nudity"] < 0.35


def test_rear_view_partially_exposed_thong_triggers_review() -> None:
    """Partially covered buttocks (e.g., thong/cheeky bottom) should trigger partial_nudity or Needs Review."""
    img = _create_synthetic_test_image(view_angle="rear", coverage="partial_covered")
    analysis = BodyRegionDetector.analyze_image(img)

    assert analysis.coverage_category == CoverageCategory.PARTIAL_COVERED.value
    assert analysis.partial_nudity_score > 0.50

    classifier = SyntheticDiagnosticClassifier()
    scores = classifier.predict(img)

    result = ScoringEngine.evaluate(scores, flag_threshold=0.70, review_threshold=0.40)
    # Falls in uncertainty range or flagged as partial exposure
    assert result.requires_review is True
    assert result.verdict in (Verdict.NEEDS_REVIEW.value, Verdict.FLAGGED.value)
    assert scores["partial_nudity"] >= 0.40


def test_rear_view_bare_exposed_buttocks_classified_as_nudity() -> None:
    """Bare exposed buttocks without clothing must classify as buttocks_exposed and Flagged Nudity."""
    img = _create_synthetic_test_image(view_angle="rear", coverage="exposed")
    analysis = BodyRegionDetector.analyze_image(img)

    assert analysis.coverage_category == CoverageCategory.EXPOSED.value
    assert analysis.buttocks_exposed_score >= 0.70

    classifier = SyntheticDiagnosticClassifier()
    scores = classifier.predict(img)

    result = ScoringEngine.evaluate(scores, flag_threshold=0.70, review_threshold=0.40)
    assert result.verdict == Verdict.FLAGGED.value
    assert result.requires_review is True
    assert scores["buttocks_exposed"] >= 0.70
    assert scores["nudity"] >= 0.70
    assert "bare exposed buttocks" in result.explanation.lower() or "nudity" in result.explanation.lower()


def test_side_view_exposed_buttocks_classified_as_nudity() -> None:
    """Side profile view with exposed buttocks must also be flagged as nudity."""
    img = _create_synthetic_test_image(view_angle="side", coverage="exposed")
    analysis = BodyRegionDetector.analyze_image(img)

    assert analysis.view_angle == ViewAngle.SIDE.value
    assert analysis.buttocks_exposed_score >= 0.65

    classifier = SyntheticDiagnosticClassifier()
    scores = classifier.predict(img)
    result = ScoringEngine.evaluate(scores, flag_threshold=0.70, review_threshold=0.40)

    assert result.requires_review is True
    assert scores["buttocks_exposed"] >= 0.60


def test_dark_melanin_complexion_detection() -> None:
    """Detection must accurately handle deep/rich skin tones without false negatives or false positives."""
    dark_skin = (80, 45, 25)

    # Clothed with dark skin: should be safe
    clothed_dark = _create_synthetic_test_image(view_angle="rear", coverage="clothed", skin_tone=dark_skin)
    clothed_analysis = BodyRegionDetector.analyze_image(clothed_dark)
    assert clothed_analysis.coverage_category == CoverageCategory.FULLY_COVERED.value
    assert clothed_analysis.buttocks_exposed_score < 0.20

    # Bare exposed buttocks with dark skin: should be flagged
    exposed_dark = _create_synthetic_test_image(view_angle="rear", coverage="exposed", skin_tone=dark_skin)
    exposed_analysis = BodyRegionDetector.analyze_image(exposed_dark)
    assert exposed_analysis.coverage_category == CoverageCategory.EXPOSED.value
    assert exposed_analysis.buttocks_exposed_score >= 0.65


def test_evaluation_metrics_precision_recall_fpr() -> None:
    """Evaluate full test matrix across angles and clothing and assert Precision, Recall, and FPR."""
    test_cases = [
        # (view_angle, coverage, expected_flagged)
        ("rear", "clothed", False),
        ("rear", "swimwear_covered", False),
        ("side", "clothed", False),
        ("side", "swimwear_covered", False),
        ("front", "clothed", False),
        ("front", "swimwear_covered", False),
        ("rear", "exposed", True),
        ("side", "exposed", True),
        ("rear", "exposed", True),
        ("side", "exposed", True),
    ]

    classifier = SyntheticDiagnosticClassifier()
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for angle, cov, is_nude in test_cases:
        img = _create_synthetic_test_image(view_angle=angle, coverage=cov)
        scores = classifier.predict(img)
        res = ScoringEngine.evaluate(scores, flag_threshold=0.65, review_threshold=0.40)
        predicted_flagged = (res.verdict == Verdict.FLAGGED.value) and (scores["buttocks_exposed"] >= 0.60)

        if is_nude and predicted_flagged:
            tp += 1
        elif is_nude and not predicted_flagged:
            fn += 1
        elif not is_nude and predicted_flagged:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n[Test Metrics] TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"[Test Metrics] Precision={precision:.3f}, Recall={recall:.3f}, FPR={fpr:.3f}")

    # Assert rigorous benchmarks
    assert precision >= 0.90, f"Expected Precision >= 0.90, got {precision}"
    assert recall >= 0.90, f"Expected Recall >= 0.90, got {recall}"
    assert fpr <= 0.10, f"Expected FPR <= 0.10, got {fpr}"
