"""Unit tests for the FastAPI API endpoints.

All tests use FastAPI's ``TestClient`` for synchronous request/response
cycles. The YOLO model is mocked throughout so no weights need to be
present on disk.
"""

import pytest
import numpy as np
import cv2
import io
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.detector import DetectionResult, BoundingBox


# ==========================================================================
# Helpers
# ==========================================================================


def _make_test_image_bytes(
    width: int = 640, height: int = 480, channels: int = 3, fmt: str = ".png"
) -> bytes:
    """Generate a synthetic image encoded as bytes.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of colour channels (3 = BGR).
        fmt: OpenCV encode format string (e.g. ``".png"``, ``".jpg"``).

    Returns:
        Raw bytes of the encoded image.
    """
    img = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    success, buffer = cv2.imencode(fmt, img)
    assert success, "Failed to encode test image"
    return buffer.tobytes()


def _make_mock_detection_result() -> DetectionResult:
    """Create a ``DetectionResult`` with a single person detection."""
    bbox = BoundingBox(
        x1=100.0,
        y1=100.0,
        x2=200.0,
        y2=200.0,
        confidence=0.92,
        class_id=0,
        class_name="person",
    )
    return DetectionResult(
        boxes=[bbox],
        annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8),
        inference_time=15.0,
        frame_shape=(480, 640, 3),
        total_objects=1,
        class_counts={"person": 1},
        avg_confidence=0.92,
    )


# ==========================================================================
# Fixtures
# ==========================================================================


@pytest.fixture
def mock_detector():
    """Patch the ``ObjectDetector`` used by the API module.

    The mock detector returns a canned ``DetectionResult`` on every
    ``detect`` call and exposes minimal helper methods.
    """
    mock_instance = MagicMock()

    # detect() → canned result
    mock_instance.detect.return_value = _make_mock_detection_result()
    mock_instance.detect_image.return_value = _make_mock_detection_result()

    # get_class_names() → small COCO subset
    mock_instance.get_class_names.return_value = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "dog",
        4: "cat",
    }

    # model_name and device attributes
    mock_instance.model_name = "yolo11n"
    mock_instance.device = "cpu"
    mock_instance.confidence_threshold = 0.5
    mock_instance.nms_threshold = 0.45

    # get_model_info()
    mock_instance.get_model_info.return_value = {
        "model_size": "nano",
        "confidence": 0.5,
        "iou_threshold": 0.45,
        "num_classes": 80,
    }

    # Statistics tracking
    mock_instance.get_statistics.return_value = {
        "total_detections": 42,
        "total_frames": 10,
        "avg_inference_time": 15.0,
    }

    import app.api
    app.api.detector = mock_instance
    with patch("app.api.ObjectDetector", return_value=mock_instance):
        yield mock_instance


@pytest.fixture
def client(mock_detector) -> TestClient:
    """Return a ``TestClient`` wired to the mocked API application."""
    # Import inside fixture so the patched detector is used at module load
    from app.api import app

    return TestClient(app)


# ==========================================================================
# Health & Info Endpoints
# ==========================================================================


class TestHealthEndpoint:
    """Tests for ``GET /health``."""

    def test_health_endpoint(self, client: TestClient):
        """The health endpoint should return 200 with a status field."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestClassesEndpoint:
    """Tests for ``GET /classes``."""

    def test_classes_endpoint(self, client: TestClient):
        """Should return the class-name dictionary from the detector."""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        # The response should contain class information
        assert isinstance(data, (dict, list))


class TestStatisticsEndpoint:
    """Tests for ``GET /statistics``."""

    def test_statistics_endpoint(self, client: TestClient):
        """Should return detection statistics."""
        response = client.get("/statistics")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


# ==========================================================================
# Detection Endpoints
# ==========================================================================


class TestDetectImageEndpoint:
    """Tests for ``POST /detect/image``."""

    def test_detect_image_endpoint(self, client: TestClient):
        """Uploading a valid image should return detection results."""
        image_bytes = _make_test_image_bytes()
        files = {"file": ("test.png", io.BytesIO(image_bytes), "image/png")}

        response = client.post("/detect/image", files=files)
        assert response.status_code == 200
        data = response.json()
        # Should contain detection results structure
        assert isinstance(data, dict)

    def test_detect_image_invalid_file(self, client: TestClient, mock_detector):
        """Uploading an invalid (non-image) file should return an error."""
        # Provide random non-image bytes
        mock_detector.detect.side_effect = Exception("Invalid image")
        bad_bytes = b"this is not an image at all"
        files = {"file": ("bad.txt", io.BytesIO(bad_bytes), "text/plain")}

        response = client.post("/detect/image", files=files)
        # Expect a 4xx or 5xx error
        assert response.status_code >= 400

    def test_detect_annotated_image(self, client: TestClient):
        """``POST /detect/image/annotated`` should return image bytes."""
        image_bytes = _make_test_image_bytes()
        files = {"file": ("test.png", io.BytesIO(image_bytes), "image/png")}

        response = client.post("/detect/image/annotated", files=files)
        # Should succeed and return an image content type
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "image" in content_type or len(response.content) > 0


# ==========================================================================
# Settings Endpoint
# ==========================================================================


class TestSettingsEndpoint:
    """Tests for ``PUT /settings``."""

    def test_settings_update(self, client: TestClient):
        """Updating settings with valid values should return 200."""
        payload = {
            "confidence": 0.6,
            "nms_threshold": 0.4,
        }
        response = client.put("/settings", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
