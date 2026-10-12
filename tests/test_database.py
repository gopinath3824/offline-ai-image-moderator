"""Tests for SQLite database operations and history repository."""

from pathlib import Path
# pyrefly: ignore [missing-import]
import pytest

from storage.history_repository import HistoryRepository, ScanRecord


@pytest.fixture
def test_repo(tmp_path: Path) -> HistoryRepository:
    db_file = tmp_path / "test_history.db"
    return HistoryRepository(db_path=db_file)


def test_save_and_retrieve_record(test_repo: HistoryRepository) -> None:
    """Ensure records are inserted, assigned an ID, and accurately deserialized."""
    record = ScanRecord(
        filename="sample1.png",
        file_path="C:/images/sample1.png",
        file_size_bytes=1048576,
        dimensions="1920 x 1080 px",
        format_name="PNG",
        verdict="Flagged",
        requires_review=True,
        highest_category="Nudity",
        highest_score=0.88,
        category_scores={"nudity": 0.88, "suggestive": 0.45},
        model_id="falconsai_nsfw",
        model_name="Falconsai Baseline",
        inference_latency_sec=0.42,
    )

    rec_id = test_repo.save_record(record)
    assert rec_id > 0

    fetched = test_repo.get_record(rec_id)
    assert fetched is not None
    assert fetched.filename == "sample1.png"
    assert fetched.verdict == "Flagged"
    assert fetched.requires_review is True
    assert fetched.highest_score == 0.88
    assert fetched.category_scores["nudity"] == 0.88


def test_list_records_and_filters(test_repo: HistoryRepository) -> None:
    """Test listing with search query and verdict filters."""
    r1 = ScanRecord(filename="photo_safe.jpg", verdict="Likely Safe", highest_category="None", highest_score=0.05)
    r2 = ScanRecord(filename="photo_flagged.jpg", verdict="Flagged", highest_category="Violence", highest_score=0.91)
    r3 = ScanRecord(filename="photo_review.jpg", verdict="Needs Review", highest_category="Suggestive", highest_score=0.48)

    test_repo.save_record(r1)
    test_repo.save_record(r2)
    test_repo.save_record(r3)

    # Filter by verdict
    flagged_list = test_repo.list_records(verdict_filter="Flagged")
    assert len(flagged_list) == 1
    assert flagged_list[0].filename == "photo_flagged.jpg"

    # Search by keyword
    search_list = test_repo.list_records(search_query="review")
    assert len(search_list) == 1
    assert search_list[0].filename == "photo_review.jpg"


def test_summary_stats(test_repo: HistoryRepository) -> None:
    """Test KPI aggregation summary counts."""
    test_repo.save_record(ScanRecord(filename="s1.png", verdict="Likely Safe"))
    test_repo.save_record(ScanRecord(filename="s2.png", verdict="Likely Safe"))
    test_repo.save_record(ScanRecord(filename="f1.png", verdict="Flagged"))
    test_repo.save_record(ScanRecord(filename="r1.png", verdict="Needs Review"))

    stats = test_repo.get_summary_stats()
    assert stats["total"] == 4
    assert stats["safe"] == 2
    assert stats["flagged"] == 1
    assert stats["review"] == 1


def test_delete_and_clear(test_repo: HistoryRepository) -> None:
    """Test single record deletion and clearing all history."""
    r_id = test_repo.save_record(ScanRecord(filename="delete_me.png", verdict="Likely Safe"))
    assert test_repo.get_record(r_id) is not None

    deleted = test_repo.delete_record(r_id)
    assert deleted is True
    assert test_repo.get_record(r_id) is None

    test_repo.save_record(ScanRecord(filename="temp.png", verdict="Likely Safe"))
    count = test_repo.clear_all()
    assert count >= 1
    assert test_repo.get_summary_stats()["total"] == 0
