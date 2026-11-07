"""Configuration management for the Object Detection System.

Provides centralized, validated configuration using Pydantic Settings.
All settings can be overridden via environment variables prefixed with ``ODS_``.

Example:
    Override the model size from the shell::

        set ODS_MODEL_SIZE=yolo11s
        python -m app.main
"""

import logging
from pathlib import Path
from enum import Enum
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings


logger = logging.getLogger(__name__)


class ModelSize(str, Enum):
    """Supported YOLO v11 model sizes.

    Each variant trades accuracy for speed:
        - NANO:   fastest, lowest accuracy  (~2.6 M params)
        - SMALL:  balanced                  (~9.4 M params)
        - MEDIUM: slowest, highest accuracy (~20.1 M params)
    """

    NANO = "yolo11n"
    SMALL = "yolo11s"
    MEDIUM = "yolo11m"


# ---------------------------------------------------------------------------
# Pre-computed base directory (module-level so it is resolved once)
# ---------------------------------------------------------------------------
_BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings with validation and env-var support.

    Attributes:
        model_size: YOLO model variant to load.
        confidence_threshold: Minimum detection confidence score.
        nms_threshold: IoU threshold for Non-Maximum Suppression.
        base_dir: Project root directory.
        weights_dir: Directory for model weight files.
        uploads_dir: Directory for uploaded images / videos.
        outputs_dir: Directory for output artefacts.
        device: Inference device (``""`` = auto-detect, ``"cpu"``, ``"0"``).
        half_precision: Use FP16 inference when supported.
        img_size: Input image size for inference (pixels).
        max_det: Maximum detections per image.
        api_host: FastAPI server bind address.
        api_port: FastAPI server bind port.
        log_level: Python logging level name.
        tracker_type: Multi-object tracker backend (``bytetrack`` or ``botsort``).
        enable_detection_logging: Write per-frame detection records to JSONL.
        detection_log_file: Path to the JSONL detection log.
    """

    # --- Model settings ---------------------------------------------------
    model_size: ModelSize = ModelSize.NANO
    confidence_threshold: float = Field(
        default=0.25,
        ge=0.10,
        le=1.00,
        description="Minimum confidence score to keep a detection.",
    )
    nms_threshold: float = Field(
        default=0.45,
        ge=0.10,
        le=1.00,
        description="IoU threshold for Non-Maximum Suppression.",
    )

    # --- Paths ------------------------------------------------------------
    base_dir: Path = _BASE_DIR
    weights_dir: Path = _BASE_DIR / "weights"
    uploads_dir: Path = _BASE_DIR / "uploads"
    outputs_dir: Path = _BASE_DIR / "outputs"

    # --- Performance ------------------------------------------------------
    device: str = Field(
        default="",
        description='Inference device. "" = auto-detect, "cpu", "0" for GPU 0.',
    )
    half_precision: bool = Field(
        default=True,
        description="Enable FP16 half-precision inference (GPU only).",
    )
    img_size: int = Field(
        default=640,
        ge=320,
        le=1280,
        description="Input image size in pixels (square).",
    )
    max_det: int = Field(
        default=300,
        ge=1,
        le=1000,
        description="Maximum number of detections per frame.",
    )

    # --- API --------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1024, le=65535)

    # --- Logging ----------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # --- Tracking ---------------------------------------------------------
    tracker_type: str = Field(
        default="bytetrack",
        description="Multi-object tracker backend: 'bytetrack' or 'botsort'.",
    )

    # --- Detection logging ------------------------------------------------
    enable_detection_logging: bool = True
    detection_log_file: Path = _BASE_DIR / "outputs" / "detections.jsonl"

    # --- COCO class names (80 classes) – used for label look-up -----------
    COCO_CLASSES: ClassVar[list[str]] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    model_config = {
        "env_prefix": "ODS_",
        "case_sensitive": False,
        "protected_namespaces": ("settings_",),
    }

    # --- Derived properties -----------------------------------------------

    @property
    def model_path(self) -> str:
        """Return the weight file name for the configured model size.

        Returns:
            Filename string, e.g. ``"yolo11n.pt"``.
        """
        return f"{self.model_size.value}.pt"

    @property
    def model_weights_path(self) -> Path:
        """Return the full path to the model weight file.

        Returns:
            Absolute ``Path`` to the ``.pt`` file inside *weights_dir*.
        """
        return self.weights_dir / self.model_path

    # --- Helper methods ---------------------------------------------------

    def setup_directories(self) -> None:
        """Create all required directories if they do not exist."""
        for dir_path in [self.weights_dir, self.uploads_dir, self.outputs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", dir_path)

    def setup_logging(self) -> None:
        """Configure the root logger with the application format."""
        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger.info(
            "Logging configured at %s level.", self.log_level.upper()
        )

    def get_class_name(self, class_id: int) -> str:
        """Look up the human-readable class name for a COCO class ID.

        Args:
            class_id: Integer class index (0-79 for COCO).

        Returns:
            Class name string, or ``"unknown"`` if the ID is out of range.
        """
        if 0 <= class_id < len(self.COCO_CLASSES):
            return self.COCO_CLASSES[class_id]
        return "unknown"

    def summary(self) -> dict:
        """Return a compact dict summarising the current configuration.

        Returns:
            Dictionary with key configuration values.
        """
        return {
            "model_size": self.model_size.value,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "device": self.device or "auto",
            "half_precision": self.half_precision,
            "img_size": self.img_size,
            "max_det": self.max_det,
            "tracker_type": self.tracker_type,
            "log_level": self.log_level,
        }


# ---------------------------------------------------------------------------
# Module-level singleton – import ``settings`` anywhere in the project.
# ---------------------------------------------------------------------------
settings = Settings()
settings.setup_directories()
settings.setup_logging()

logger.info("Settings initialised: %s", settings.summary())
