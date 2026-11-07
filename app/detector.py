"""Core object detection engine wrapping Ultralytics YOLO v11.

This module provides a thread-safe, lazily-loaded YOLO inference wrapper that
supports both single-image detection and multi-object tracking via ByteTrack /
BoTSORT.

Typical usage::

    from app.detector import ObjectDetector, DetectionResult

    detector = ObjectDetector()
    result: DetectionResult = detector.detect_image(frame)

    # With tracking (video streams)
    result = detector.detect_and_track(frame)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.config import ModelSize, settings

# Try importing YOLO at module level if available, but fall back gracefully
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

logger = logging.getLogger(__name__)


# ===========================================================================
# Data classes
# ===========================================================================


@dataclass
class BoundingBox:
    """A single detected bounding box with metadata.

    Coordinates use either (x, y, width, height) or (x1, y1, x2, y2).

    Attributes:
        x: Left edge of the box (pixels).
        y: Top edge of the box (pixels).
        width: Box width (pixels).
        height: Box height (pixels).
        label: Human-readable class name.
        confidence: Detection confidence in ``[0, 1]``.
        track_id: Unique tracker-assigned ID (``None`` when tracking is off).
        class_id: Integer COCO class index.
    """

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    label: str = ""
    confidence: float = 0.0
    track_id: Optional[int] = None
    class_id: int = 0

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 0,
        height: int = 0,
        label: str = "",
        confidence: float = 0.0,
        track_id: Optional[int] = None,
        class_id: int = 0,
        x1: Optional[float] = None,
        y1: Optional[float] = None,
        x2: Optional[float] = None,
        y2: Optional[float] = None,
        class_name: Optional[str] = None,
    ) -> None:
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            self.x = int(round(x1))
            self.y = int(round(y1))
            self.width = int(round(x2 - x1))
            self.height = int(round(y2 - y1))
        else:
            self.x = int(round(x))
            self.y = int(round(y))
            self.width = int(round(width))
            self.height = int(round(height))

        self.label = class_name if class_name is not None else label
        self.confidence = float(confidence)
        self.track_id = track_id
        self.class_id = int(class_id)

    # --- Convenience properties -------------------------------------------

    @property
    def x1(self) -> int:
        return self.x

    @property
    def y1(self) -> int:
        return self.y

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def class_name(self) -> str:
        return self.label

    @property
    def center(self) -> tuple[int, int]:
        """Return the ``(cx, cy)`` centre point of the box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Return the area of the box in pixels²."""
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Return the box in ``(x1, y1, x2, y2)`` format."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x, y, x + width, y + height)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def detections(self) -> list[BoundingBox]:
        """Alias for boxes."""
        return self.boxes

    def to_dict(self) -> dict:
        """Serialise the bounding box to a plain dictionary.

        Returns:
            Dictionary with all bounding-box fields.
        """
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "label": self.label,
            "class_name": self.label,
            "confidence": round(self.confidence, 4),
            "track_id": self.track_id,
            "class_id": self.class_id,
        }


@dataclass
class DetectionResult:
    """Aggregated result of a single inference pass.

    Attributes:
        boxes: List of detected bounding boxes.
        frame: Annotated copy of the input frame (BGR, ``np.ndarray``).
        inference_time_ms: Wall-clock inference duration in milliseconds.
        total_objects: Total number of detections.
        class_counts: Mapping of class name → count.
        avg_confidence: Mean confidence across all detections.
        fps: Instantaneous frames-per-second derived from *inference_time_ms*.
    """

    boxes: list[BoundingBox] = field(default_factory=list)
    frame: np.ndarray = field(default_factory=lambda: np.empty(0))
    inference_time_ms: float = 0.0
    total_objects: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    fps: float = 0.0
    annotated_frame: Optional[np.ndarray] = None
    inference_time: Optional[float] = None
    frame_shape: Optional[tuple] = None

    def __post_init__(self):
        if self.annotated_frame is None and len(self.frame) > 0:
            self.annotated_frame = self.frame
        elif self.frame is None or len(self.frame) == 0:
            if self.annotated_frame is not None:
                self.frame = self.annotated_frame
        if self.inference_time is None:
            self.inference_time = self.inference_time_ms
        elif self.inference_time_ms == 0.0 and self.inference_time is not None:
            self.inference_time_ms = self.inference_time

    @property
    def detections(self) -> list[BoundingBox]:
        """Alias for boxes."""
        return self.boxes

    def to_dict(self) -> dict:
        """Serialise the result (excluding the raw frame) for JSON transport.

        Returns:
            Dictionary with detection statistics and box data.
        """
        return {
            "total_objects": self.total_objects,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "fps": round(self.fps, 2),
            "avg_confidence": round(self.avg_confidence, 4),
            "class_counts": self.class_counts,
            "boxes": [b.to_dict() for b in self.boxes],
        }


# ===========================================================================
# Core detector
# ===========================================================================


class ObjectDetector:
    """Thread-safe YOLO v11 object detection and tracking engine.

    The underlying ``ultralytics.YOLO`` model is loaded lazily on first
    inference call and can be swapped at runtime via :meth:`change_model`.

    Args:
        model_size: Initial model variant to load.
        confidence: Minimum detection confidence.
        nms_threshold: IoU threshold for Non-Maximum Suppression.
        device: Inference device (``""`` = auto-detect).
        half: Enable FP16 half-precision inference.
        img_size: Square input image dimension.
        max_det: Maximum detections per frame.
        tracker_type: Tracker backend name (``"bytetrack"`` or ``"botsort"``).

    Example::

        detector = ObjectDetector(
            model_size=ModelSize.SMALL,
            confidence=0.30,
        )
        result = detector.detect_image(cv2.imread("photo.jpg"))
        print(result.class_counts)
    """

    def __init__(
        self,
        model_size: ModelSize = settings.model_size,
        confidence: float = settings.confidence_threshold,
        nms_threshold: float = settings.nms_threshold,
        device: str = settings.device,
        half: bool = settings.half_precision,
        img_size: int = settings.img_size,
        max_det: int = settings.max_det,
        tracker_type: str = settings.tracker_type,
        iou_threshold: Optional[float] = None,
    ) -> None:
        if isinstance(model_size, str):
            model_size = ModelSize(model_size)
        self._model_size: ModelSize = model_size
        self.confidence: float = confidence
        self.nms_threshold: float = iou_threshold if iou_threshold is not None else nms_threshold
        self.device: str = device
        self.half: bool = half
        self.img_size: int = img_size
        self.max_det: int = max_det
        self.tracker_type: str = tracker_type

        # The YOLO model is loaded lazily.
        self._model = None  # type: ignore[assignment]
        self._lock = threading.Lock()
        self._class_names: dict[int, str] = {}

        logger.info(
            "ObjectDetector created – model=%s  conf=%.2f  nms=%.2f  device=%s",
            self._model_size.value,
            self.confidence,
            self.nms_threshold,
            self.device or "auto",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Download (if necessary) and load the YOLO model.

        This method is called automatically on the first inference call
        and whenever the model variant is changed via :meth:`change_model`.

        Raises:
            RuntimeError: If the model fails to load.
        """
        try:
            import sys
            if "app.detector" in sys.modules and hasattr(sys.modules["app.detector"], "YOLO"):
                detector_yolo = getattr(sys.modules["app.detector"], "YOLO")
            else:
                detector_yolo = None

            if detector_yolo is not None and not isinstance(detector_yolo, type(None)):
                yolo_cls = detector_yolo
            else:
                from ultralytics import YOLO as yolo_cls

            model_name = f"{self._model_size.value}.pt"
            weights_file = settings.weights_dir / model_name
            target = str(weights_file) if weights_file.exists() else model_name
            logger.info("Loading YOLO model: %s …", target)

            t0 = time.perf_counter()
            self._model = yolo_cls(target)
            elapsed = (time.perf_counter() - t0) * 1000

            # Cache class-name mapping from the model
            if hasattr(self._model, "names") and self._model.names:
                self._class_names = dict(self._model.names)
            else:
                self._class_names = dict(enumerate(settings.COCO_CLASSES))

            logger.info(
                "Model loaded in %.1f ms – %d classes available.",
                elapsed,
                len(self._class_names),
            )
        except Exception as exc:
            logger.exception("Failed to load YOLO model.")
            raise RuntimeError(f"Model loading failed: {exc}") from exc

    def _ensure_model(self) -> None:
        """Ensure the model is loaded (lazy initialisation)."""
        if self._model is None:
            self._load_model()

    def _get_class_name(self, class_id: int) -> str:
        """Resolve a class ID to a human-readable label.

        Falls back to the settings-level COCO list if the model does not
        expose class names.

        Args:
            class_id: Integer class index.

        Returns:
            Class name string.
        """
        if class_id in self._class_names:
            return self._class_names[class_id]
        return settings.get_class_name(class_id)

    def _parse_results(
        self,
        results,  # ultralytics Results list
        original_frame: np.ndarray,
        elapsed_ms: float,
        with_tracking: bool = False,
    ) -> DetectionResult:
        """Parse raw Ultralytics results into a :class:`DetectionResult`.

        Args:
            results: List of ``ultralytics.engine.results.Results`` objects.
            original_frame: The original input frame (BGR).
            elapsed_ms: Inference wall-clock time in milliseconds.
            with_tracking: If ``True``, attempt to extract tracker IDs.

        Returns:
            Populated :class:`DetectionResult`.
        """
        boxes_list: list[BoundingBox] = []
        result_obj = results[0]

        # Use the plotted / annotated frame provided by Ultralytics
        annotated_frame: np.ndarray = result_obj.plot()

        if result_obj.boxes is not None and len(result_obj.boxes) > 0:
            det_boxes = result_obj.boxes

            # Core tensors → numpy
            xyxy_arr = det_boxes.xyxy.cpu().numpy()  # (N, 4)
            conf_arr = det_boxes.conf.cpu().numpy()  # (N,)
            cls_arr = det_boxes.cls.cpu().numpy().astype(int)  # (N,)

            # Tracker IDs (may be None if tracking was not used or failed)
            track_ids = None
            if with_tracking and det_boxes.id is not None:
                track_ids = det_boxes.id.cpu().numpy().astype(int)

            for idx in range(len(xyxy_arr)):
                x1, y1, x2, y2 = xyxy_arr[idx]
                x = int(round(x1))
                y = int(round(y1))
                w = int(round(x2 - x1))
                h = int(round(y2 - y1))
                class_id = int(cls_arr[idx])
                confidence = float(conf_arr[idx])
                track_id = int(track_ids[idx]) if track_ids is not None else None
                label = self._get_class_name(class_id)

                boxes_list.append(
                    BoundingBox(
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        label=label,
                        confidence=confidence,
                        track_id=track_id,
                        class_id=class_id,
                    )
                )

        # Aggregate statistics
        total = len(boxes_list)
        class_counts: dict[str, int] = dict(Counter(b.label for b in boxes_list))
        avg_conf = (
            float(np.mean([b.confidence for b in boxes_list])) if total > 0 else 0.0
        )
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

        return DetectionResult(
            boxes=boxes_list,
            frame=annotated_frame,
            inference_time_ms=elapsed_ms,
            total_objects=total,
            class_counts=class_counts,
            avg_confidence=avg_conf,
            fps=fps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def iou_threshold(self) -> float:
        """Alias for nms_threshold."""
        return self.nms_threshold

    @iou_threshold.setter
    def iou_threshold(self, val: float) -> None:
        self.nms_threshold = val

    @property
    def model(self):
        """Expose the underlying model (triggering load if needed)."""
        self._ensure_model()
        return self._model

    def get_class_names(self) -> dict[int, str]:
        """Return mapping of class IDs to class names."""
        self._ensure_model()
        return dict(self._class_names)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Alias for detect_image."""
        return self.detect_image(image)

    def detect_image(self, image: np.ndarray) -> DetectionResult:
        """Run single-image object detection (no tracking).

        Args:
            image: BGR input frame as a NumPy array with shape ``(H, W, 3)``.

        Returns:
            :class:`DetectionResult` containing detections and the annotated frame.

        Raises:
            ValueError: If *image* is ``None`` or has an invalid shape.
            RuntimeError: If the model is not loaded and cannot be loaded.
        """
        if image is None or image.ndim < 2:
            raise ValueError("Invalid input image: must be a non-empty NumPy array.")

        with self._lock:
            self._ensure_model()

            t0 = time.perf_counter()
            results = self._model(
                image,
                conf=self.confidence,
                iou=self.nms_threshold,
                device=self.device if self.device else None,
                half=self.half,
                imgsz=self.img_size,
                max_det=self.max_det,
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

        detection_result = self._parse_results(
            results, image, elapsed_ms, with_tracking=False
        )

        logger.debug(
            "detect_image: %d objects in %.1f ms (%.1f FPS)",
            detection_result.total_objects,
            detection_result.inference_time_ms,
            detection_result.fps,
        )
        return detection_result

    def detect_and_track(self, image: np.ndarray) -> DetectionResult:
        """Run object detection **with** multi-object tracking.

        Uses the configured tracker backend (ByteTrack / BoTSORT) and
        ``persist=True`` so that track IDs are maintained across calls.

        Args:
            image: BGR input frame as a NumPy array with shape ``(H, W, 3)``.

        Returns:
            :class:`DetectionResult` with ``track_id`` populated on each box.

        Raises:
            ValueError: If *image* is ``None`` or has an invalid shape.
            RuntimeError: If the model is not loaded and cannot be loaded.
        """
        if image is None or image.ndim < 2:
            raise ValueError("Invalid input image: must be a non-empty NumPy array.")

        with self._lock:
            self._ensure_model()

            t0 = time.perf_counter()
            results = self._model.track(
                image,
                conf=self.confidence,
                iou=self.nms_threshold,
                device=self.device if self.device else None,
                half=self.half,
                imgsz=self.img_size,
                max_det=self.max_det,
                persist=True,
                tracker=self.tracker_type,
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

        detection_result = self._parse_results(
            results, image, elapsed_ms, with_tracking=True
        )

        logger.debug(
            "detect_and_track: %d objects in %.1f ms (%.1f FPS)",
            detection_result.total_objects,
            detection_result.inference_time_ms,
            detection_result.fps,
        )
        return detection_result

    def change_model(self, model_size: ModelSize) -> None:
        """Hot-swap the underlying YOLO model at runtime.

        The new model is loaded immediately.  Any in-progress tracking state
        is implicitly reset because the model instance is replaced.

        Args:
            model_size: Target model variant.
        """
        if model_size == self._model_size and self._model is not None:
            logger.info("Model %s is already loaded – skipping.", model_size.value)
            return

        logger.info(
            "Switching model from %s → %s",
            self._model_size.value,
            model_size.value,
        )
        with self._lock:
            self._model_size = model_size
            self._model = None  # release old model
            self._load_model()

    def update_thresholds(
        self,
        confidence: Optional[float] = None,
        nms: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> None:
        """Update detection thresholds at runtime.

        Args:
            confidence: New confidence threshold (``0.10`` – ``1.00``).
            nms: New NMS IoU threshold (``0.10`` – ``1.00``).
            iou_threshold: Alias for nms.

        Raises:
            ValueError: If a threshold is outside the valid range.
        """
        nms_target = nms if nms is not None else iou_threshold
        if confidence is not None:
            if not 0.10 <= confidence <= 1.00:
                raise ValueError(
                    f"Confidence must be in [0.10, 1.00], got {confidence}"
                )
            self.confidence = confidence
            logger.info("Confidence threshold updated to %.2f", confidence)

        if nms_target is not None:
            if not 0.10 <= nms_target <= 1.00:
                raise ValueError(f"NMS threshold must be in [0.10, 1.00], got {nms_target}")
            self.nms_threshold = nms_target
            logger.info("NMS threshold updated to %.2f", nms_target)

    def get_model_info(self) -> dict:
        """Return metadata about the currently loaded model.

        Returns:
            Dictionary with model name, class count, device info, and
            current threshold values.
        """
        info: dict = {
            "model_size": self._model_size.value,
            "model_loaded": self._model is not None,
            "confidence_threshold": self.confidence,
            "confidence": self.confidence,
            "nms_threshold": self.nms_threshold,
            "iou_threshold": self.nms_threshold,
            "device": self.device or "auto",
            "half_precision": self.half,
            "img_size": self.img_size,
            "max_det": self.max_det,
            "tracker_type": self.tracker_type,
            "num_classes": len(self._class_names),
        }

        # Add device details when the model is loaded
        if self._model is not None:
            try:
                info["model_task"] = getattr(self._model, "task", "detect")
                # Attempt to report actual device from the model parameters
                if hasattr(self._model, "device"):
                    info["active_device"] = str(self._model.device)
            except Exception:
                pass  # non-critical metadata

        return info

    def reset_tracker(self) -> None:
        """Reset the multi-object tracker state.

        Call this when switching to a new video source to avoid carrying over
        stale track IDs.
        """
        with self._lock:
            if self._model is not None:
                # Reloading the model is the most reliable way to reset the
                # internal tracker state in Ultralytics.
                self._load_model()
                logger.info("Tracker state reset (model reloaded).")

    @property
    def model_size(self) -> ModelSize:
        """Return the currently configured model size."""
        return self._model_size

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if the YOLO model is currently loaded in memory."""
        return self._model is not None
