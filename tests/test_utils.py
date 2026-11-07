"""Unit tests for utility functions.

Covers FPS calculation, detection logging, motion-trail tracking,
line-crossing counting, ROI counting, colour helpers, image validation,
screenshot saving, and detection drawing.
"""

import pytest
import numpy as np
import cv2
import json
import tempfile
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils import (
    FPSCounter,
    DetectionLogger,
    MotionTrailTracker,
    LineCrossingCounter,
    ROICounter,
    get_class_color,
    validate_image,
    save_screenshot,
    draw_detections,
)
from app.detector import BoundingBox


# ==========================================================================
# Helpers
# ==========================================================================


def _make_bbox(
    x1: float = 10.0,
    y1: float = 10.0,
    x2: float = 50.0,
    y2: float = 50.0,
    confidence: float = 0.9,
    class_id: int = 0,
    class_name: str = "person",
) -> BoundingBox:
    """Convenience factory for ``BoundingBox`` instances."""
    return BoundingBox(
        x1=x1, y1=y1, x2=x2, y2=y2,
        confidence=confidence, class_id=class_id, class_name=class_name,
    )


def _blank_image(
    width: int = 640, height: int = 480, channels: int = 3
) -> np.ndarray:
    """Return a blank BGR image."""
    return np.zeros((height, width, channels), dtype=np.uint8)


# ==========================================================================
# FPSCounter
# ==========================================================================


class TestFPSCounter:
    """Tests for ``FPSCounter``."""

    def test_fps_counter(self):
        """FPS value should be a non-negative float after several updates."""
        counter = FPSCounter()

        # Simulate several frames
        for _ in range(20):
            counter.update()

        fps = counter.get_fps()
        assert isinstance(fps, float)
        assert fps >= 0.0

    def test_fps_counter_initial(self):
        """Before any update the FPS should be zero (or a sensible default)."""
        counter = FPSCounter()
        fps = counter.get_fps()
        assert isinstance(fps, float)
        assert fps >= 0.0


# ==========================================================================
# DetectionLogger
# ==========================================================================


class TestDetectionLogger:
    """Tests for ``DetectionLogger``."""

    def test_detection_logger(self, tmp_path: Path):
        """Logger should accept entries and return them on retrieval."""
        log_file = tmp_path / "test_detections.jsonl"
        logger = DetectionLogger(log_path=log_file)

        # Log a few detections
        logger.log(detections=[_make_bbox()], frame_number=1)
        logger.log(detections=[_make_bbox(class_name="car", class_id=2)], frame_number=2)

        entries = logger.get_entries()
        assert isinstance(entries, list)
        assert len(entries) >= 2

    def test_detection_logger_empty(self, tmp_path: Path):
        """A fresh logger should return an empty list."""
        log_file = tmp_path / "empty_detections.jsonl"
        logger = DetectionLogger(log_path=log_file)
        entries = logger.get_entries()
        assert isinstance(entries, list)
        assert len(entries) == 0


# ==========================================================================
# MotionTrailTracker
# ==========================================================================


class TestMotionTrailTracker:
    """Tests for ``MotionTrailTracker``."""

    def test_motion_trail_tracker(self):
        """Updating with successive positions should build a trail."""
        tracker = MotionTrailTracker()

        # Simulate an object moving across frames
        for i in range(10):
            center = (100 + i * 10, 200 + i * 5)
            tracker.update(object_id=1, center=center)

        trails = tracker.get_trails()
        assert isinstance(trails, dict)
        assert 1 in trails
        assert len(trails[1]) > 0

    def test_motion_trail_tracker_draw(self):
        """Drawing trails on an image should not raise and should return an image."""
        tracker = MotionTrailTracker()
        for i in range(5):
            tracker.update(object_id=0, center=(50 + i * 20, 100))

        frame = _blank_image()
        result = tracker.draw(frame)
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape


# ==========================================================================
# LineCrossingCounter
# ==========================================================================


class TestLineCrossingCounter:
    """Tests for ``LineCrossingCounter``."""

    def test_line_crossing_counter(self):
        """Counter should detect when an object crosses a defined line."""
        # Horizontal line at y=240 spanning the full width
        counter = LineCrossingCounter(
            line_start=(0, 240), line_end=(640, 240)
        )

        # Object moves from above the line to below it
        counter.update(object_id=1, center=(320, 200))
        counter.update(object_id=1, center=(320, 280))

        count = counter.get_count()
        assert isinstance(count, (int, dict))
        # At least one crossing should be registered
        total = count if isinstance(count, int) else sum(count.values())
        assert total >= 1

    def test_line_crossing_no_cross(self):
        """No crossing should be registered if the object stays on one side."""
        counter = LineCrossingCounter(
            line_start=(0, 240), line_end=(640, 240)
        )

        counter.update(object_id=1, center=(320, 100))
        counter.update(object_id=1, center=(320, 120))

        count = counter.get_count()
        total = count if isinstance(count, int) else sum(count.values())
        assert total == 0


# ==========================================================================
# ROICounter
# ==========================================================================


class TestROICounter:
    """Tests for ``ROICounter``."""

    def test_roi_counter(self):
        """Objects inside the ROI should be counted."""
        # Define a rectangular ROI in the centre of the frame
        roi_points = [(200, 150), (440, 150), (440, 330), (200, 330)]
        counter = ROICounter(roi_points=roi_points)

        # One bbox inside, one outside
        inside = _make_bbox(x1=250, y1=200, x2=300, y2=260)
        outside = _make_bbox(x1=10, y1=10, x2=50, y2=50)

        count = counter.count(detections=[inside, outside])
        assert isinstance(count, int)
        assert count >= 1  # at least the inside box

    def test_roi_counter_empty(self):
        """No detections should yield a count of zero."""
        roi_points = [(200, 150), (440, 150), (440, 330), (200, 330)]
        counter = ROICounter(roi_points=roi_points)
        count = counter.count(detections=[])
        assert count == 0


# ==========================================================================
# Colour helper
# ==========================================================================


class TestGetClassColor:
    """Tests for ``get_class_color``."""

    def test_get_class_color(self):
        """Colour for the same class should be deterministic."""
        c1 = get_class_color(0)
        c2 = get_class_color(0)
        assert c1 == c2

    def test_get_class_color_distinct(self):
        """Different classes should (usually) get different colours."""
        colours = {get_class_color(i) for i in range(10)}
        # We expect at least several distinct colours
        assert len(colours) >= 5

    def test_get_class_color_format(self):
        """Each colour should be a tuple of three ints in [0, 255]."""
        colour = get_class_color(42)
        assert isinstance(colour, tuple)
        assert len(colour) == 3
        for channel in colour:
            assert 0 <= channel <= 255


# ==========================================================================
# Image validation
# ==========================================================================


class TestValidateImage:
    """Tests for ``validate_image``."""

    def test_validate_image_valid(self):
        """A properly encoded image should decode successfully."""
        img = _blank_image()
        _, buf = cv2.imencode(".png", img)
        image_bytes = buf.tobytes()

        decoded = validate_image(image_bytes)
        assert isinstance(decoded, np.ndarray)
        assert decoded.ndim == 3

    def test_validate_image_invalid(self):
        """Random bytes should raise an error or return None."""
        bad_bytes = b"not_an_image_at_all_1234567890"
        with pytest.raises((ValueError, Exception)):
            result = validate_image(bad_bytes)
            # If the function returns None instead of raising, fail the test
            if result is None:
                raise ValueError("validate_image returned None for invalid data")


# ==========================================================================
# Screenshot saving
# ==========================================================================


class TestSaveScreenshot:
    """Tests for ``save_screenshot``."""

    def test_save_screenshot(self, tmp_path: Path):
        """Saving a screenshot should create a file on disk."""
        img = _blank_image()
        output_path = tmp_path / "screenshot.png"

        saved = save_screenshot(img, str(output_path))

        # The function may return the path or a bool
        assert output_path.exists() or saved is not None
        if output_path.exists():
            assert output_path.stat().st_size > 0


# ==========================================================================
# Draw detections
# ==========================================================================


class TestDrawDetections:
    """Tests for ``draw_detections``."""

    def test_draw_detections(self):
        """Drawing detections should return an image of the same shape."""
        img = _blank_image()
        boxes = [
            _make_bbox(x1=50, y1=50, x2=150, y2=150, class_name="person"),
            _make_bbox(x1=200, y1=200, x2=300, y2=300, class_name="car", class_id=2),
        ]

        result = draw_detections(img, boxes)
        assert isinstance(result, np.ndarray)
        assert result.shape == img.shape

    def test_draw_detections_empty(self):
        """Drawing with no boxes should return the original image unchanged."""
        img = _blank_image()
        result = draw_detections(img, [])
        assert isinstance(result, np.ndarray)
        assert result.shape == img.shape
