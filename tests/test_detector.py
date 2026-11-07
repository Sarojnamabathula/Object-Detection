"""Unit tests for the object detection module.

Tests cover detector initialization, inference, result structures,
model switching, threshold updates, and error handling. All YOLO model
calls are mocked so no weights need to be downloaded.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup — allow imports from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure ultralytics mock is available in sys.modules to avoid permission errors
import unittest.mock
if "ultralytics" not in sys.modules:
    mock_ultralytics = unittest.mock.MagicMock()
    sys.modules["ultralytics"] = mock_ultralytics

from app.detector import ObjectDetector, DetectionResult, BoundingBox
from app.config import ModelSize


# ==========================================================================
# Fixtures
# ==========================================================================


def _make_mock_yolo(class_names: dict | None = None) -> MagicMock:
    """Create a MagicMock that behaves like an Ultralytics YOLO model.

    Args:
        class_names: Optional mapping of class-id → class-name. Defaults to
            a small COCO-like subset.

    Returns:
        A configured MagicMock YOLO instance.
    """
    if class_names is None:
        class_names = {0: "person", 1: "bicycle", 2: "car", 3: "dog", 4: "cat"}

    mock_model = MagicMock()
    mock_model.names = class_names
    return mock_model


def _make_mock_results(
    boxes_xyxy: np.ndarray | None = None,
    confidences: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    speed: dict | None = None,
) -> list[MagicMock]:
    """Build a list containing a single mock Results object.

    When called with no arguments the result set represents zero detections.
    """
    if boxes_xyxy is None:
        boxes_xyxy = np.empty((0, 4), dtype=np.float32)
    if confidences is None:
        confidences = np.empty((0,), dtype=np.float32)
    if class_ids is None:
        class_ids = np.empty((0,), dtype=np.int32)
    if speed is None:
        speed = {"preprocess": 1.0, "inference": 10.0, "postprocess": 2.0}

    mock_boxes = MagicMock()
    mock_boxes.xyxy = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = boxes_xyxy
    mock_boxes.conf = MagicMock()
    mock_boxes.conf.cpu.return_value.numpy.return_value = confidences
    mock_boxes.cls = MagicMock()
    mock_boxes.cls.cpu.return_value.numpy.return_value = class_ids
    # Support len() on boxes to determine detection count
    mock_boxes.__len__ = lambda self: len(boxes_xyxy)

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_result.speed = speed
    # plot() returns an annotated BGR image
    mock_result.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    return [mock_result]


@pytest.fixture
def mock_yolo_class():
    """Patch ``ultralytics.YOLO`` so no weights are loaded.

    Yields the mock class; each call to ``YOLO(...)`` returns a fresh
    mock model configured with COCO-like class names.
    """
    with patch("app.detector.YOLO") as MockYOLO:
        MockYOLO.return_value = _make_mock_yolo()
        yield MockYOLO


@pytest.fixture
def detector(mock_yolo_class) -> ObjectDetector:
    """Return an ``ObjectDetector`` backed by a mocked YOLO model."""
    det = ObjectDetector(model_size=ModelSize.NANO, confidence=0.5, iou_threshold=0.45)
    _ = det.model  # Trigger lazy loading so YOLO constructor is called
    return det


# ==========================================================================
# Initialization
# ==========================================================================


class TestDetectorInitialization:
    """Tests for ``ObjectDetector.__init__``."""

    def test_detector_initialization(self, detector: ObjectDetector, mock_yolo_class):
        """Detector should initialise with the requested parameters."""
        assert detector is not None
        assert detector.confidence == 0.5
        assert detector.iou_threshold == 0.45
        # YOLO constructor should have been called at least once
        mock_yolo_class.assert_called()

    def test_detector_has_class_names(self, detector: ObjectDetector):
        """Detector exposes class names from the underlying model."""
        names = detector.get_class_names()
        assert isinstance(names, dict)
        assert 0 in names
        assert names[0] == "person"


# ==========================================================================
# Inference
# ==========================================================================


class TestDetectorInference:
    """Tests for ``ObjectDetector.detect``."""

    def test_detect_image_returns_result(self, detector: ObjectDetector):
        """``detect`` should return a ``DetectionResult`` with correct fields."""
        # Prepare a fake image and mock model output with one detection
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_results = _make_mock_results(
            boxes_xyxy=np.array([[100.0, 100.0, 200.0, 200.0]], dtype=np.float32),
            confidences=np.array([0.92], dtype=np.float32),
            class_ids=np.array([0], dtype=np.int32),
        )
        detector.model.return_value = mock_results

        result = detector.detect(image)

        assert isinstance(result, DetectionResult)
        assert len(result.boxes) == 1
        assert result.boxes[0].class_name == "person"
        assert result.boxes[0].confidence == pytest.approx(0.92, abs=1e-4)

    def test_detect_empty_image(self, detector: ObjectDetector):
        """A blank image with no objects should yield zero detections."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_results = _make_mock_results()  # no detections
        detector.model.return_value = mock_results

        result = detector.detect(image)

        assert isinstance(result, DetectionResult)
        assert len(result.boxes) == 0


# ==========================================================================
# Data structures
# ==========================================================================


class TestDataStructures:
    """Tests for ``DetectionResult`` and ``BoundingBox`` dataclasses."""

    def test_bounding_box_dataclass(self):
        """``BoundingBox`` should store all fields and serialise via ``to_dict``."""
        bbox = BoundingBox(
            x1=10.0,
            y1=20.0,
            x2=110.0,
            y2=120.0,
            confidence=0.85,
            class_id=2,
            class_name="car",
        )

        assert bbox.x1 == 10.0
        assert bbox.y1 == 20.0
        assert bbox.x2 == 110.0
        assert bbox.y2 == 120.0
        assert bbox.confidence == 0.85
        assert bbox.class_id == 2
        assert bbox.class_name == "car"

        as_dict = bbox.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["class_name"] == "car"
        assert "confidence" in as_dict

    def test_detection_result_dataclass(self):
        """``DetectionResult`` should hold boxes, timing, and annotated frame."""
        bbox = BoundingBox(
            x1=0, y1=0, x2=50, y2=50,
            confidence=0.9, class_id=0, class_name="person",
        )
        annotated = np.zeros((480, 640, 3), dtype=np.uint8)
        result = DetectionResult(
            boxes=[bbox],
            annotated_frame=annotated,
            inference_time=12.5,
            frame_shape=(480, 640, 3),
        )

        assert len(result.boxes) == 1
        assert result.inference_time == 12.5
        assert result.frame_shape == (480, 640, 3)
        assert result.annotated_frame is annotated


# ==========================================================================
# Model management
# ==========================================================================


class TestModelManagement:
    """Tests for model switching, threshold updates, and info retrieval."""

    def test_change_model(self, detector: ObjectDetector, mock_yolo_class):
        """Switching model size should reload the YOLO model."""
        initial_call_count = mock_yolo_class.call_count
        detector.change_model(ModelSize.MEDIUM)
        assert mock_yolo_class.call_count > initial_call_count

    def test_update_thresholds(self, detector: ObjectDetector):
        """Threshold setters should update internal state."""
        detector.update_thresholds(confidence=0.7, iou_threshold=0.3)
        assert detector.confidence == pytest.approx(0.7)
        assert detector.iou_threshold == pytest.approx(0.3)

    def test_get_model_info(self, detector: ObjectDetector):
        """``get_model_info`` should return a dict with expected keys."""
        info = detector.get_model_info()
        assert isinstance(info, dict)
        assert "model_size" in info
        assert "confidence" in info
        assert "iou_threshold" in info
        assert "num_classes" in info


# ==========================================================================
# Error handling
# ==========================================================================


class TestErrorHandling:
    """Tests for invalid inputs and graceful error behaviour."""

    def test_invalid_model_size(self, mock_yolo_class):
        """Passing a completely invalid model size should raise an error."""
        with pytest.raises((ValueError, KeyError, TypeError, Exception)):
            ObjectDetector(model_size="invalid_size", confidence=0.5, iou_threshold=0.45)
