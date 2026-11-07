"""Utility functions and helper classes for the Object Detection System.

Provides drawing utilities, FPS tracking, detection logging, motion-trail
tracking, line-crossing counting, and ROI counting.

All heavy NumPy / OpenCV operations are kept in this module so that the rest
of the application can stay framework-agnostic.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Colour palette
# ===========================================================================

# 20 visually-distinct BGR colours; indices wrap around via modulo.
_PALETTE: list[tuple[int, int, int]] = [
    (230, 25, 75),    # red
    (60, 180, 75),    # green
    (255, 225, 25),   # yellow
    (0, 130, 200),    # blue
    (245, 130, 48),   # orange
    (145, 30, 180),   # purple
    (70, 240, 240),   # cyan
    (240, 50, 230),   # magenta
    (210, 245, 60),   # lime
    (250, 190, 212),  # pink
    (0, 128, 128),    # teal
    (220, 190, 255),  # lavender
    (170, 110, 40),   # brown
    (255, 250, 200),  # beige
    (128, 0, 0),      # maroon
    (170, 255, 195),  # mint
    (128, 128, 0),    # olive
    (255, 215, 180),  # coral
    (0, 0, 128),      # navy
    (128, 128, 128),  # grey
]


def get_class_color(class_id: int) -> tuple[int, int, int]:
    """Return a consistent, visually distinct BGR colour for *class_id*.

    The colour is deterministic: the same ``class_id`` always produces the
    same colour.

    Args:
        class_id: Integer class index (may be negative or very large).

    Returns:
        ``(B, G, R)`` tuple with values in ``[0, 255]``.
    """
    return _PALETTE[class_id % len(_PALETTE)]


# ===========================================================================
# Drawing helpers
# ===========================================================================


def _draw_rounded_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = -1,
    radius: int = 4,
) -> None:
    """Draw a rectangle with rounded corners on *img* (in-place).

    Args:
        img: Target image (BGR).
        pt1: Top-left corner ``(x, y)``.
        pt2: Bottom-right corner ``(x, y)``.
        color: BGR fill / outline colour.
        thickness: ``-1`` for filled, else outline thickness.
        radius: Corner radius in pixels.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2, 10)
    if r < 1:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return

    # Four arcs + rectangles that together form a rounded rect
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)


def draw_detections(
    frame: np.ndarray,
    detections: list,
    show_labels: bool = True,
    show_confidence: bool = True,
    show_boxes: bool = True,
    show_trails: bool = False,
    trails: Optional["MotionTrailTracker"] = None,
) -> np.ndarray:
    """Draw bounding boxes, labels, confidence scores, and optional motion
    trails on *frame*.

    The frame is **modified in-place** and also returned for convenience.

    Args:
        frame: BGR image (``np.ndarray``) to annotate.
        detections: List of :class:`~app.detector.BoundingBox` objects.
        show_labels: Overlay class-name labels.
        show_confidence: Overlay confidence percentage.
        show_boxes: Draw the bounding rectangle.
        show_trails: Render motion trails (requires *trails*).
        trails: :class:`MotionTrailTracker` instance holding trajectory data.

    Returns:
        The annotated frame (same reference as input).
    """
    if frame is None or len(frame.shape) < 2:
        logger.warning("draw_detections received invalid frame; returning as-is.")
        return frame

    if hasattr(detections, "boxes"):
        detections = detections.boxes
    elif hasattr(detections, "detections"):
        detections = detections.detections

    overlay = frame.copy()

    for det in detections:
        color = get_class_color(det.class_id)
        x1, y1 = det.x, det.y
        x2, y2 = det.x + det.width, det.y + det.height

        # --- Bounding box ---
        if show_boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # --- Label background + text ---
        if show_labels or show_confidence:
            parts: list[str] = []
            if show_labels:
                label = det.label
                if det.track_id is not None:
                    label = f"#{det.track_id} {label}"
                parts.append(label)
            if show_confidence:
                parts.append(f"{det.confidence:.0%}")
            text = " ".join(parts)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Label background rectangle
            pad = 4
            lbl_y1 = max(y1 - th - 2 * pad, 0)
            lbl_y2 = y1
            lbl_x1 = x1
            lbl_x2 = x1 + tw + 2 * pad

            _draw_rounded_rect(overlay, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
            cv2.putText(
                overlay,
                text,
                (lbl_x1 + pad, lbl_y2 - pad),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # --- Motion trail ---
        if show_trails and trails is not None and det.track_id is not None:
            trail_pts = trails.get_trail(det.track_id)
            if trail_pts and len(trail_pts) > 1:
                pts_arr = np.array(trail_pts, dtype=np.int32)
                cv2.polylines(overlay, [pts_arr], False, color, 2, cv2.LINE_AA)

    # Alpha-blend for a subtle overlay effect
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    return frame


# ===========================================================================
# FPS Counter
# ===========================================================================


class FPSCounter:
    """Tracks instantaneous frames-per-second using a sliding window.

    Args:
        window_size: Number of timestamps to keep for the rolling average.
    """

    def __init__(self, window_size: int = 60) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def update(self) -> None:
        """Record the current timestamp."""
        self._timestamps.append(time.perf_counter())

    def tick(self) -> None:
        """Alias for update()."""
        self.update()

    def get_fps(self) -> float:
        """Compute the rolling FPS.

        Returns:
            Frames per second (``0.0`` if fewer than 2 samples).
        """
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()


# ===========================================================================
# Detection Logger
# ===========================================================================


class DetectionLogger:
    """Append detection records to a JSONL (newline-delimited JSON) file.

    Each line is a self-contained JSON object with a UTC timestamp, frame
    identifier, and the list of detections.

    Args:
        log_path: Destination ``.jsonl`` file. Created on first write. Defaults to settings.detection_log_file.
        enabled: If ``False``, all write operations are silently skipped.
    """

    def __init__(
        self,
        log_path: Optional[Path | str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        from app.config import settings
        self._log_path = Path(log_path) if log_path is not None else settings.detection_log_file
        self._enabled = enabled if enabled is not None else settings.enable_detection_logging
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        detections: list,
        frame_id: Optional[int] = None,
        frame_number: Optional[int] = None,
    ) -> None:
        """Write a detection record to the log.

        Args:
            detections: List of :class:`~app.detector.BoundingBox` objects.
            frame_id: Optional sequential frame counter.
            frame_number: Alias for frame_id.
        """
        if not self._enabled or not detections:
            return

        fid = frame_id if frame_id is not None else frame_number
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_id": fid,
            "detections": [],
        }

        for det in detections:
            record["detections"].append(
                {
                    "label": det.label,
                    "confidence": round(det.confidence, 4),
                    "x": det.x,
                    "y": det.y,
                    "width": det.width,
                    "height": det.height,
                    "track_id": det.track_id,
                    "class_id": det.class_id,
                }
            )

        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            logger.exception("Failed to write detection log to %s", self._log_path)

    def get_entries(self, n: int = 50) -> list[dict]:
        """Alias for get_recent."""
        return self.get_recent(n)

    def get_recent(self, n: int = 50) -> list[dict]:
        """Read the most recent *n* records from the log.

        Args:
            n: Number of records to return (from the tail of the file).

        Returns:
            List of parsed JSON records (newest last).
        """
        if not self._log_path.exists():
            return []

        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            tail = lines[-n:] if len(lines) > n else lines
            return [json.loads(line) for line in tail if line.strip()]
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to read detection log.")
            return []

    def clear(self) -> None:
        """Truncate the log file."""
        try:
            if self._log_path.exists():
                self._log_path.write_text("", encoding="utf-8")
                logger.info("Detection log cleared: %s", self._log_path)
        except OSError:
            logger.exception("Failed to clear detection log.")


# ===========================================================================
# Motion Trail Tracker
# ===========================================================================


class MotionTrailTracker:
    """Maintain per-object motion trajectories for visualisation.

    Each tracked object stores up to *max_length* recent centre-point
    positions in a ``deque``.

    Args:
        max_length: Maximum number of points per trail.
    """

    def __init__(self, max_length: int = 30) -> None:
        self._trails: dict[int, deque[tuple[int, int]]] = {}
        self._max_length = max_length

    def update(
        self,
        track_id: Optional[int] = None,
        center_point: Optional[tuple[int, int]] = None,
        object_id: Optional[int] = None,
        center: Optional[tuple[int, int]] = None,
    ) -> None:
        """Append a new point to the trail of *track_id* (or *object_id*).

        Args:
            track_id: Unique tracker-assigned ID.
            center_point: ``(cx, cy)`` position.
            object_id: Alias for track_id.
            center: Alias for center_point.
        """
        tid = track_id if track_id is not None else object_id
        pt = center_point if center_point is not None else center
        if tid is None or pt is None:
            return
        if tid not in self._trails:
            self._trails[tid] = deque(maxlen=self._max_length)
        self._trails[tid].append(pt)

    def get_trail(self, track_id: int) -> list[tuple[int, int]]:
        """Return the trail for a single object.

        Args:
            track_id: Tracker ID to look up.

        Returns:
            Ordered list of ``(cx, cy)`` points (oldest first), or ``[]``.
        """
        if track_id in self._trails:
            return list(self._trails[track_id])
        return []

    def get_trails(self) -> dict[int, list[tuple[int, int]]]:
        """Alias for get_all_trails."""
        return self.get_all_trails()

    def get_all_trails(self) -> dict[int, list[tuple[int, int]]]:
        """Return all current trails.

        Returns:
            Mapping of track_id → list of ``(cx, cy)`` points.
        """
        return {tid: list(pts) for tid, pts in self._trails.items()}

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw all motion trails onto the frame.

        Args:
            frame: BGR image.

        Returns:
            Annotated frame.
        """
        out = frame.copy()
        for tid, points in self._trails.items():
            if len(points) < 2:
                continue
            color = get_class_color(tid)
            for i in range(1, len(points)):
                cv2.line(out, points[i - 1], points[i], color, 2, cv2.LINE_AA)
        return out

    def clear(self) -> None:
        """Remove all stored trails."""
        self._trails.clear()


# ===========================================================================
# Line Crossing Counter
# ===========================================================================


def _cross_product_sign(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
) -> float:
    """Return the sign of the cross product ``(b - a) × (c - a)``.

    Positive → *c* is to the left of ``a→b``; negative → right; zero → collinear.
    """
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


class LineCrossingCounter:
    """Count objects crossing a virtual line defined by two points.

    A crossing is detected when a tracked object's centre moves from one
    side of the line to the other between successive frames.  The direction
    is labelled *entered* or *exited* depending on the sign change.

    Args:
        pt1: First endpoint ``(x, y)`` of the counting line (optional at init).
        pt2: Second endpoint ``(x, y)`` of the counting line (optional at init).
    """

    def __init__(
        self,
        pt1: Optional[tuple[int, int]] = None,
        pt2: Optional[tuple[int, int]] = None,
        line_start: Optional[tuple[int, int]] = None,
        line_end: Optional[tuple[int, int]] = None,
    ) -> None:
        self._pt1: Optional[tuple[int, int]] = pt1 or line_start
        self._pt2: Optional[tuple[int, int]] = pt2 or line_end
        self._prev_positions: dict[int, tuple[int, int]] = {}
        self._entered: int = 0
        self._exited: int = 0

    def set_line(self, pt1: tuple[int, int], pt2: tuple[int, int]) -> None:
        """(Re-)define the counting line.

        Resets all internal counters and previous-position state.

        Args:
            pt1: First endpoint ``(x, y)``.
            pt2: Second endpoint ``(x, y)``.
        """
        self._pt1 = pt1
        self._pt2 = pt2
        self._prev_positions.clear()
        self._entered = 0
        self._exited = 0
        logger.info("Line crossing counter set: %s → %s", pt1, pt2)

    def update(
        self,
        track_id: Optional[int] = None,
        center: Optional[tuple[int, int]] = None,
        object_id: Optional[int] = None,
    ) -> str:
        """Update the position of *track_id* and detect a crossing.

        Args:
            track_id: Unique tracker-assigned ID.
            center: Current ``(cx, cy)`` centre of the object.
            object_id: Alias for track_id.

        Returns:
            ``"entered"`` if the object crossed from right→left of the line
            vector, ``"exited"`` for left→right, or ``"none"`` otherwise.
        """
        tid = track_id if track_id is not None else object_id
        if tid is None or center is None:
            return "none"
        if self._pt1 is None or self._pt2 is None:
            return "none"

        result = "none"

        if tid in self._prev_positions:
            prev = self._prev_positions[tid]
            prev_side = _cross_product_sign(self._pt1, self._pt2, prev)
            curr_side = _cross_product_sign(self._pt1, self._pt2, center)

            # Crossing occurs when the sign changes (and neither is zero)
            if prev_side != 0 and curr_side != 0:
                if prev_side > 0 and curr_side < 0:
                    self._entered += 1
                    result = "entered"
                elif prev_side < 0 and curr_side > 0:
                    self._exited += 1
                    result = "exited"

        self._prev_positions[tid] = center
        return result

    def get_count(self) -> int:
        """Return total crossings as an integer."""
        return self._entered + self._exited

    def get_counts(self) -> dict[str, int]:
        """Return the cumulative crossing counts.

        Returns:
            Dictionary with ``entered``, ``exited``, and ``total`` keys.
        """
        return {
            "entered": self._entered,
            "exited": self._exited,
            "total": self._entered + self._exited,
        }

    def reset(self) -> None:
        """Reset counters and position history (keeps the line definition)."""
        self._prev_positions.clear()
        self._entered = 0
        self._exited = 0
        logger.info("Line crossing counters reset.")

    def draw_line(self, frame: np.ndarray) -> np.ndarray:
        """Draw the counting line on *frame*.

        Args:
            frame: BGR image.

        Returns:
            The annotated frame.
        """
        if self._pt1 is not None and self._pt2 is not None:
            cv2.line(frame, self._pt1, self._pt2, (0, 0, 255), 2, cv2.LINE_AA)
            counts = self.get_counts()
            label = f"In: {counts['entered']}  Out: {counts['exited']}"
            mid_x = (self._pt1[0] + self._pt2[0]) // 2
            mid_y = (self._pt1[1] + self._pt2[1]) // 2
            cv2.putText(
                frame,
                label,
                (mid_x + 10, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return frame


# ===========================================================================
# ROI Counter
# ===========================================================================


class ROICounter:
    """Count objects inside a rectangular Region of Interest.

    Args:
        x: Left edge of the ROI.
        y: Top edge of the ROI.
        w: Width of the ROI.
        h: Height of the ROI.
        roi_points: Optional polygon or corner points [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        w: int = 0,
        h: int = 0,
        roi_points: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        if roi_points is not None and len(roi_points) >= 2:
            xs = [p[0] for p in roi_points]
            ys = [p[1] for p in roi_points]
            self._x = min(xs)
            self._y = min(ys)
            self._w = max(xs) - self._x
            self._h = max(ys) - self._y
        else:
            self._x = x
            self._y = y
            self._w = w
            self._h = h

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Define or redefine the ROI rectangle.

        Args:
            x: Left edge.
            y: Top edge.
            w: Width.
            h: Height.
        """
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        logger.info("ROI set to (%d, %d, %d, %d)", x, y, w, h)

    def _is_inside(self, cx: int, cy: int) -> bool:
        """Check whether a centre point falls within the ROI."""
        return (
            self._x <= cx <= self._x + self._w
            and self._y <= cy <= self._y + self._h
        )

    def count(self, detections: list) -> int:
        """Count total detections falling inside the ROI."""
        res = self.count_in_roi(detections)
        return res["total_in_roi"]

    def count_in_roi(self, detections: list) -> dict:
        """Count how many detections have their centre inside the ROI.

        Args:
            detections: List of :class:`~app.detector.BoundingBox` objects.

        Returns:
            Dictionary with ``total_in_roi`` and per-class ``class_counts``.
        """
        total = 0
        class_counts: dict[str, int] = {}

        for det in detections:
            cx, cy = det.center
            if self._is_inside(cx, cy):
                total += 1
                class_counts[det.label] = class_counts.get(det.label, 0) + 1

        return {
            "total_in_roi": total,
            "class_counts": class_counts,
        }

    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        """Draw the ROI rectangle and label on *frame*.

        Args:
            frame: BGR image.

        Returns:
            The annotated frame.
        """
        if self._w <= 0 or self._h <= 0:
            return frame

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (self._x, self._y),
            (self._x + self._w, self._y + self._h),
            (0, 255, 0),
            -1,
        )
        # Semi-transparent fill
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Solid border
        cv2.rectangle(
            frame,
            (self._x, self._y),
            (self._x + self._w, self._y + self._h),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            "ROI",
            (self._x + 5, self._y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return frame


# ===========================================================================
# Image / video validation
# ===========================================================================


def validate_image(image_or_bytes: bytes | np.ndarray) -> np.ndarray:
    """Decode and validate an image from raw bytes or validate an existing NumPy image array.

    Args:
        image_or_bytes: Raw image file content (JPEG, PNG, etc.) or decoded BGR np.ndarray.

    Returns:
        Decoded BGR ``np.ndarray`` with shape ``(H, W, 3)``.

    Raises:
        ValueError: If the input cannot be decoded or has an invalid shape.
    """
    if isinstance(image_or_bytes, np.ndarray):
        if image_or_bytes.size == 0:
            raise ValueError("Empty image array.")
        if image_or_bytes.ndim != 3 or image_or_bytes.shape[2] != 3:
            raise ValueError(f"Unexpected image shape {image_or_bytes.shape}; expected (H, W, 3).")
        return image_or_bytes

    if not image_or_bytes:
        raise ValueError("Empty file: no bytes received.")

    arr = np.frombuffer(image_or_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            "Could not decode image. Supported formats: JPEG, PNG, BMP, TIFF, WebP."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Unexpected image shape {image.shape}; expected (H, W, 3)."
        )

    logger.debug("Validated image: %dx%d", image.shape[1], image.shape[0])
    return image


def validate_video(file_path: str | Path) -> bool:
    """Check whether a video file can be opened and read.

    Args:
        file_path: Path to the video file.

    Returns:
        ``True`` if at least one frame can be read; ``False`` otherwise.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning("Video file does not exist: %s", file_path)
        return False

    cap = cv2.VideoCapture(str(file_path))
    try:
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", file_path)
            return False

        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("Cannot read first frame: %s", file_path)
            return False

        logger.debug(
            "Valid video: %s  (%dx%d, %.1f FPS, %d frames)",
            file_path.name,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        return True
    finally:
        cap.release()


# ===========================================================================
# Screenshot / file helpers
# ===========================================================================


def save_screenshot(frame: np.ndarray, output_dir: str | Path) -> Path:
    """Save a frame as a timestamped PNG screenshot or directly to destination path if a file is specified.

    Args:
        frame: BGR image to save.
        output_dir: Destination directory or file path.

    Returns:
        Path to the saved PNG file.

    Raises:
        ValueError: If *frame* is ``None`` or empty.
        OSError: If the file cannot be written.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Cannot save empty frame.")

    dest = Path(output_dir)
    if dest.suffix:
        # User passed a full file path
        dest.parent.mkdir(parents=True, exist_ok=True)
        filepath = dest
    else:
        # User passed a directory
        dest.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"screenshot_{timestamp}_{unique_id}.png"
        filepath = dest / filename

    success = cv2.imwrite(str(filepath), frame)
    if not success:
        raise OSError(f"cv2.imwrite failed for {filepath}")

    logger.info("Screenshot saved: %s", filepath)
    return filepath


# ===========================================================================
# Statistics formatting
# ===========================================================================


def format_detection_stats(result) -> dict:
    """Format a :class:`~app.detector.DetectionResult` for display / API response.

    Args:
        result: A ``DetectionResult`` instance.

    Returns:
        Dictionary with human-friendly statistics.
    """
    return {
        "total_objects": result.total_objects,
        "inference_time_ms": round(result.inference_time_ms, 2),
        "fps": round(result.fps, 1),
        "avg_confidence": round(result.avg_confidence * 100, 1),
        "class_counts": result.class_counts,
        "detections": [
            {
                "label": b.label,
                "confidence": round(b.confidence * 100, 1),
                "bbox": {"x": b.x, "y": b.y, "w": b.width, "h": b.height},
                "track_id": b.track_id,
            }
            for b in result.boxes
        ],
    }
