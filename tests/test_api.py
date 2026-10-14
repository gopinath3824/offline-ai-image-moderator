"""Tests for REST API endpoints and serialization."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def _make_dummy_image_bytes(color: tuple[int, int, int] = (200, 100, 100)) -> bytes:
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(client: TestClient) -> None:
    """Test health check route."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "active_model" in data


def test_root_endpoint(client: TestClient) -> None:
    """Test API root documentation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "/docs" in data["docs"]


def test_models_catalog(client: TestClient) -> None:
    """Test model listing route."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_model_id" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) >= 1


def test_single_image_moderation(client: TestClient) -> None:
    """Test /api/v1/moderate with synthetic diagnostic model."""
    img_bytes = _make_dummy_image_bytes()
    files = {"file": ("test_pic.png", img_bytes, "image/png")}
    data = {"model_id": "synthetic_diagnostic"}

    response = client.post("/api/v1/moderate", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["filename"] == "test_pic.png"
    assert res["verdict"] in ["Likely Safe", "Needs Review", "Flagged"]
    assert "category_scores" in res
    assert "category_breakdown" in res
    assert res["latency_ms"] >= 0
    assert "explanation" in res


def test_batch_moderation(client: TestClient) -> None:
    """Test /api/v1/moderate/batch endpoint."""
    img1 = _make_dummy_image_bytes((50, 50, 50))
    img2 = _make_dummy_image_bytes((150, 150, 150))

    files = [
        ("files", ("image1.png", img1, "image/png")),
        ("files", ("image2.png", img2, "image/png")),
    ]
    data = {"model_id": "synthetic_diagnostic"}

    response = client.post("/api/v1/moderate/batch", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["total_images"] == 2
    assert len(res["results"]) == 2


def test_history_and_stats(client: TestClient) -> None:
    """Test history querying and summary statistics."""
    stats_resp = client.get("/api/v1/history/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert "total" in stats
    assert "safe" in stats

    hist_resp = client.get("/api/v1/history?limit=10")
    assert hist_resp.status_code == 200
    hist = hist_resp.json()
    assert "records" in hist
