"""
FastAPI REST API for the Object Detection System.

Provides endpoints for image and video object detection,
model management, health checks, and session statistics.
Leverages YOLO-based detection via the ObjectDetector class
and utility functions for annotation, validation, and logging.

Endpoints:
    POST /detect/image          — Detect objects in an uploaded image (JSON response).
    POST /detect/image/annotated — Detect objects and return the annotated image.
    POST /detect/video          — Detect objects frame-by-frame in an uploaded video.
    GET  /health                — Health / readiness check.
    GET  /classes               — List all supported COCO class names.
    GET  /statistics            — Session-level detection statistics.
    PUT  /settings              — Update detector configuration at runtime.
    GET  /logs/recent           — Retrieve recent detection log entries.
"""

from __future__ import annotations

import io
import logging
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.config import ModelSize, settings
from app.detector import BoundingBox, DetectionResult, ObjectDetector
from app.utils import (
    DetectionLogger,
    FPSCounter,
    draw_detections,
    format_detection_stats,
    save_screenshot,
    validate_image,
    validate_video,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state – populated during the application lifespan
# ---------------------------------------------------------------------------
detector: Optional[ObjectDetector] = None
detection_logger: DetectionLogger = DetectionLogger()
fps_counter: FPSCounter = FPSCounter()

detection_stats: Dict[str, Any] = {
    "total_detections": 0,
    "total_images_processed": 0,
    "total_video_frames_processed": 0,
    "total_videos_processed": 0,
    "cumulative_inference_time_ms": 0.0,
    "session_start": None,
}


# ---------------------------------------------------------------------------
# Lifespan – initialise / tear-down the detector
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage the startup and shutdown lifecycle of the application.

    Initialises the ``ObjectDetector`` on startup and releases GPU /
    system resources on shutdown.

    Args:
        application: The FastAPI application instance.

    Yields:
        None – control is handed to the running application.
    """
    global detector

    logger.info("Initialising Object Detection System …")
    logger.info(
        "Model: %s | Device: %s | Confidence: %.2f | NMS: %.2f",
        settings.model_size.value,
        settings.device or "auto",
        settings.confidence_threshold,
        settings.nms_threshold,
    )

    try:
        detector = ObjectDetector(
            model_size=settings.model_size,
            device=settings.device,
            confidence=settings.confidence_threshold,
            nms_threshold=settings.nms_threshold,
        )
        detection_stats["session_start"] = datetime.now().isoformat()
        logger.info("Object Detector initialised successfully.")
    except Exception as exc:
        logger.error("Failed to initialise ObjectDetector: %s", exc, exc_info=True)
        raise RuntimeError(f"Detector initialisation failed: {exc}") from exc

    yield  # ── application is running ──

    # Shutdown
    logger.info("Shutting down Object Detection System …")
    detector = None
    logger.info("Cleanup complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Object Detection System",
    description="AI-Powered Object Detection REST API using YOLO",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (permissive for development) ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware – request timing
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add an ``X-Process-Time`` header indicating total request duration."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1_000
    response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
    return response


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError) -> JSONResponse:
    """Return a 422 JSON response for ``ValueError`` exceptions."""
    logger.warning("ValueError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": str(exc), "type": "validation_error"},
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc: FileNotFoundError) -> JSONResponse:
    """Return a 404 JSON response when a file is not found."""
    logger.warning("FileNotFoundError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": str(exc), "type": "file_not_found"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception) -> JSONResponse:
    """Catch-all handler returning a clean 500 JSON error."""
    logger.error(
        "Unhandled exception on %s: %s", request.url.path, exc, exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
            "type": "internal_error",
        },
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _ensure_detector() -> ObjectDetector:
    """Return the global detector or raise an HTTP 503 if uninitialised."""
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Object detector is not initialised. Please try again later.",
        )
    return detector


def _build_detection_response(
    detection_result_or_boxes: Any,
    inference_time_ms: float,
) -> Dict[str, Any]:
    """Construct the standardised JSON detection response payload.

    Args:
        detection_result_or_boxes: DetectionResult object or list of BoundingBox objects.
        inference_time_ms: Time taken for inference in milliseconds.

    Returns:
        Dictionary ready to be serialised as the response body.
    """
    if isinstance(detection_result_or_boxes, DetectionResult):
        boxes = detection_result_or_boxes.boxes
    elif isinstance(detection_result_or_boxes, list):
        boxes = detection_result_or_boxes
    else:
        boxes = []

    # Aggregate per-class counts
    class_counts: Dict[str, int] = {}
    confidence_sum = 0.0
    serialised: List[Dict[str, Any]] = []

    for det in boxes:
        label = det.label if hasattr(det, "label") else getattr(det, "class_name", "unknown")
        conf = float(det.confidence)
        class_counts[label] = class_counts.get(label, 0) + 1
        confidence_sum += conf
        
        x = getattr(det, "x", getattr(det, "x1", 0))
        y = getattr(det, "y", getattr(det, "y1", 0))
        w = getattr(det, "width", getattr(det, "x2", 0) - x)
        h = getattr(det, "height", getattr(det, "y2", 0) - y)

        serialised.append(
            {
                "label": label,
                "confidence": round(conf, 4),
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                },
                "class_id": getattr(det, "class_id", 0),
                "track_id": getattr(det, "track_id", None),
            }
        )

    total_objects = len(boxes)
    avg_confidence = round(confidence_sum / total_objects, 4) if total_objects else 0.0

    # Update the FPS counter and get the current rate
    fps_counter.update()
    current_fps = fps_counter.get_fps()

    return {
        "success": True,
        "detections": serialised,
        "statistics": {
            "total_objects": total_objects,
            "class_counts": class_counts,
            "avg_confidence": avg_confidence,
            "inference_time_ms": round(inference_time_ms, 2),
            "fps": round(current_fps, 1),
        },
        "timestamp": datetime.now().isoformat(),
    }


async def _read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Read and decode an uploaded image file into a NumPy BGR array.

    Args:
        file: The uploaded file from the client.

    Returns:
        The decoded image as a NumPy array (BGR colour space).

    Raises:
        HTTPException: If the image cannot be read or decoded.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Validate MIME / magic bytes
    validate_image(contents)

    np_arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode image. Ensure the file is a valid image.",
        )
    return image


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# ── Image detection (JSON) ────────────────────────────────────────────────
@app.post("/detect/image", summary="Detect objects in an image")
async def detect_image(
    file: UploadFile = File(..., description="Image file to analyse"),
    confidence: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Confidence threshold override"
    ),
    nms_threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="NMS threshold override"
    ),
    model: Optional[str] = Query(None, description="Model size override (e.g. yolo11s)"),
) -> JSONResponse:
    """Detect objects in an uploaded image and return results as JSON.

    Args:
        file: Uploaded image file (JPEG, PNG, BMP, WEBP).
        confidence: Optional confidence threshold override.
        nms_threshold: Optional NMS IoU threshold override.
        model: Optional model size string override.

    Returns:
        JSONResponse with detection results, statistics, and timestamp.
    """
    det = _ensure_detector()
    image = await _read_image_from_upload(file)

    # Apply optional overrides
    kwargs: Dict[str, Any] = {}
    if confidence is not None:
        kwargs["confidence_threshold"] = confidence
    if nms_threshold is not None:
        kwargs["nms_threshold"] = nms_threshold

    # Run inference
    start_time = time.perf_counter()
    detection_res = det.detect(image, **kwargs)
    inference_ms = (time.perf_counter() - start_time) * 1_000

    boxes = detection_res.boxes if isinstance(detection_res, DetectionResult) else detection_res

    # Update session stats
    detection_stats["total_detections"] += len(boxes)
    detection_stats["total_images_processed"] += 1
    detection_stats["cumulative_inference_time_ms"] += inference_ms

    # Log detections
    detection_logger.log(
        detections=boxes,
        frame_id=None,
    )

    payload = _build_detection_response(detection_res, inference_ms)
    return JSONResponse(content=payload)


# ── Image detection (annotated PNG) ───────────────────────────────────────
@app.post("/detect/image/annotated", summary="Detect objects and return annotated image")
async def detect_image_annotated(
    file: UploadFile = File(..., description="Image file to analyse"),
    confidence: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Confidence threshold override"
    ),
    nms_threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="NMS threshold override"
    ),
    model: Optional[str] = Query(None, description="Model size override"),
) -> StreamingResponse:
    """Detect objects in an uploaded image and return the annotated image as PNG.

    Args:
        file: Uploaded image file.
        confidence: Optional confidence threshold override.
        nms_threshold: Optional NMS IoU threshold override.
        model: Optional model size string override.

    Returns:
        StreamingResponse containing the annotated PNG image.
    """
    det = _ensure_detector()
    image = await _read_image_from_upload(file)

    kwargs: Dict[str, Any] = {}
    if confidence is not None:
        kwargs["confidence_threshold"] = confidence
    if nms_threshold is not None:
        kwargs["nms_threshold"] = nms_threshold

    start_time = time.perf_counter()
    detection_res = det.detect(image, **kwargs)
    inference_ms = (time.perf_counter() - start_time) * 1_000

    boxes = detection_res.boxes if isinstance(detection_res, DetectionResult) else detection_res

    # Update session stats
    detection_stats["total_detections"] += len(boxes)
    detection_stats["total_images_processed"] += 1
    detection_stats["cumulative_inference_time_ms"] += inference_ms

    detection_logger.log(
        detections=boxes,
        frame_id=None,
    )

    # Draw bounding boxes on the image
    if isinstance(detection_res, DetectionResult) and detection_res.annotated_frame is not None and len(detection_res.annotated_frame) > 0:
        annotated = detection_res.annotated_frame
    else:
        annotated = draw_detections(image, boxes)

    # Encode to PNG
    success, buffer = cv2.imencode(".png", annotated)
    if not success:
        raise HTTPException(
            status_code=500, detail="Failed to encode annotated image."
        )

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=annotated.png"},
    )


# ── Video detection ───────────────────────────────────────────────────────
@app.post("/detect/video", summary="Detect objects in a video")
async def detect_video(
    file: UploadFile = File(..., description="Video file to process"),
    confidence: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Confidence threshold override"
    ),
    nms_threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="NMS threshold override"
    ),
) -> FileResponse:
    """Process a video frame-by-frame, annotate detections, and return the result.

    The uploaded video is saved to a temporary file, processed with
    ``cv2.VideoCapture`` / ``cv2.VideoWriter``, and the annotated output
    video is streamed back as an MP4.

    Args:
        file: Uploaded video file (MP4, AVI, MOV, MKV).
        confidence: Optional confidence threshold override.
        nms_threshold: Optional NMS IoU threshold override.

    Returns:
        FileResponse containing the annotated output video.
    """
    det = _ensure_detector()

    # Read the uploaded video bytes and validate
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")
    validate_video(contents)

    # Persist the upload to a temp file so OpenCV can seek over it
    input_tmp = tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False, dir=str(settings.upload_dir)
    )
    output_path = Path(tempfile.mktemp(suffix=".mp4", dir=str(settings.output_dir)))

    try:
        input_tmp.write(contents)
        input_tmp.flush()
        input_tmp.close()

        cap = cv2.VideoCapture(input_tmp.name)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Failed to open video. Ensure the file is a valid video.",
            )

        # Extract video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_value = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(str(output_path), fourcc, fps_value, (width, height))
        if not writer.isOpened():
            cap.release()
            raise HTTPException(
                status_code=500, detail="Failed to initialise video writer."
            )

        logger.info(
            "Processing video: %s — %d frames @ %.1f FPS (%dx%d)",
            file.filename,
            total_frames,
            fps_value,
            width,
            height,
        )

        kwargs: Dict[str, Any] = {}
        if confidence is not None:
            kwargs["confidence_threshold"] = confidence
        if nms_threshold is not None:
            kwargs["nms_threshold"] = nms_threshold

        frame_count = 0
        total_inference_ms = 0.0
        total_detections_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.perf_counter()
            detections = det.detect(frame, **kwargs)
            inference_ms = (time.perf_counter() - start_time) * 1_000

            total_inference_ms += inference_ms
            total_detections_count += len(detections)
            frame_count += 1

            # Annotate and write
            annotated = draw_detections(frame, detections)
            writer.write(annotated)

            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logger.info(
                    "Video progress: %d/%d frames processed", frame_count, total_frames
                )

        cap.release()
        writer.release()

        # Update session stats
        detection_stats["total_detections"] += total_detections_count
        detection_stats["total_video_frames_processed"] += frame_count
        detection_stats["total_videos_processed"] += 1
        detection_stats["cumulative_inference_time_ms"] += total_inference_ms

        logger.info(
            "Video processing complete: %d frames, %d detections, avg %.1f ms/frame",
            frame_count,
            total_detections_count,
            total_inference_ms / max(frame_count, 1),
        )

        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=f"detected_{file.filename or 'output.mp4'}",
            background=None,  # Let FastAPI handle cleanup
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as exc:
        logger.error("Video processing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Video processing failed: {exc}"
        ) from exc
    finally:
        # Clean up the input temp file; output is cleaned by FileResponse
        input_path = Path(input_tmp.name)
        if input_path.exists():
            try:
                input_path.unlink()
            except OSError:
                logger.warning("Could not remove temp input file: %s", input_path)


# ── Health check ──────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
async def health_check() -> JSONResponse:
    """Return the health status of the service including model and device info.

    Returns:
        JSONResponse with status, model name, device, and uptime.
    """
    det = _ensure_detector()

    uptime_seconds = 0.0
    if detection_stats["session_start"]:
        session_start = datetime.fromisoformat(detection_stats["session_start"])
        uptime_seconds = (datetime.now() - session_start).total_seconds()

    return JSONResponse(
        content={
            "status": "healthy",
            "model": det.model_name,
            "device": str(det.device),
            "uptime_seconds": round(uptime_seconds, 1),
        }
    )


# ── Class list ────────────────────────────────────────────────────────────
@app.get("/classes", summary="List all COCO class names")
async def get_classes() -> JSONResponse:
    """Return the full mapping of class IDs to human-readable labels.

    Returns:
        JSONResponse containing ``{classes: {0: "person", 1: "bicycle", …}}``.
    """
    det = _ensure_detector()
    class_names = det.get_class_names()
    return JSONResponse(content={"classes": class_names})


# ── Session statistics ────────────────────────────────────────────────────
@app.get("/statistics", summary="Session detection statistics")
async def get_statistics() -> JSONResponse:
    """Return aggregate statistics for the current session.

    Returns:
        JSONResponse with total detections, average inference time,
        model info, and session start time.
    """
    det = _ensure_detector()

    total_processed = (
        detection_stats["total_images_processed"]
        + detection_stats["total_video_frames_processed"]
    )
    avg_inference = (
        round(detection_stats["cumulative_inference_time_ms"] / total_processed, 2)
        if total_processed > 0
        else 0.0
    )

    return JSONResponse(
        content={
            "total_detections": detection_stats["total_detections"],
            "total_images_processed": detection_stats["total_images_processed"],
            "total_videos_processed": detection_stats["total_videos_processed"],
            "total_video_frames_processed": detection_stats[
                "total_video_frames_processed"
            ],
            "average_inference_time_ms": avg_inference,
            "model": det.model_name,
            "device": str(det.device),
            "session_start": detection_stats["session_start"],
        }
    )


# ── Runtime settings update ───────────────────────────────────────────────
@app.put("/settings", summary="Update detector settings")
async def update_settings(body: Dict[str, Any]) -> JSONResponse:
    """Update detector configuration at runtime.

    Accepted keys in the JSON body:
        - ``confidence`` (float): New confidence threshold.
        - ``nms_threshold`` (float): New NMS IoU threshold.
        - ``model`` (str): New model size (e.g. ``"yolo11s"``).

    Args:
        body: Dictionary of settings to update.

    Returns:
        JSONResponse confirming the updated settings.
    """
    det = _ensure_detector()

    updated: Dict[str, Any] = {}

    if "confidence" in body:
        conf = float(body["confidence"])
        if not 0.0 <= conf <= 1.0:
            raise HTTPException(
                status_code=422,
                detail="confidence must be between 0.0 and 1.0",
            )
        det.confidence_threshold = conf
        updated["confidence"] = conf
        logger.info("Confidence threshold updated to %.2f", conf)

    if "nms_threshold" in body:
        nms = float(body["nms_threshold"])
        if not 0.0 <= nms <= 1.0:
            raise HTTPException(
                status_code=422,
                detail="nms_threshold must be between 0.0 and 1.0",
            )
        det.nms_threshold = nms
        updated["nms_threshold"] = nms
        logger.info("NMS threshold updated to %.2f", nms)

    if "model" in body:
        model_str = str(body["model"])
        try:
            new_size = ModelSize(model_str)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid model size '{model_str}'. "
                f"Valid options: {[m.value for m in ModelSize]}",
            ) from exc
        det.load_model(new_size)
        updated["model"] = new_size.value
        logger.info("Model switched to %s", new_size.value)

    if not updated:
        raise HTTPException(
            status_code=422,
            detail="No valid settings provided. "
            "Accepted keys: confidence, nms_threshold, model.",
        )

    return JSONResponse(
        content={
            "success": True,
            "updated_settings": updated,
            "current_settings": {
                "confidence": det.confidence_threshold,
                "nms_threshold": det.nms_threshold,
                "model": det.model_name,
                "device": str(det.device),
            },
        }
    )


# ── Recent logs ───────────────────────────────────────────────────────────
@app.get("/logs/recent", summary="Recent detection logs")
async def get_recent_logs(
    n: int = Query(50, ge=1, le=1000, description="Number of recent log entries"),
) -> JSONResponse:
    """Return the *n* most recent detection log entries.

    Args:
        n: Number of entries to retrieve (default 50, max 1000).

    Returns:
        JSONResponse with a list of recent log records.
    """
    logs = detection_logger.get_recent(n)
    return JSONResponse(content={"count": len(logs), "logs": logs})
