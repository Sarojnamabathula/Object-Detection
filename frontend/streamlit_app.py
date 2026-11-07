"""Streamlit frontend for the AI Object Detection System.

A modern, dark-themed dashboard providing real-time object detection
capabilities through image upload, video processing, and webcam capture.
Communicates directly with the core backend modules (no API layer).

Usage:
    streamlit run frontend/streamlit_app.py

Author: ODS Team
"""

import logging
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup – ensure the project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import ModelSize, settings  # noqa: E402
from app.detector import DetectionResult, ObjectDetector  # noqa: E402
from app.utils import (  # noqa: E402
    DetectionLogger,
    FPSCounter,
    LineCrossingCounter,
    MotionTrailTracker,
    ROICounter,
    draw_detections,
    format_detection_stats,
    save_screenshot,
    validate_image,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Object Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS – Modern Dark Theme
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
    /* ── Global overrides ────────────────────────────────────────────── */
    .stApp {
        background-color: #0e1117;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #16213e;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00d4ff;
    }

    /* ── Metric cards ────────────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background-color: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0, 212, 255, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #00d4ff !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
        font-weight: 700;
    }

    /* ── Buttons ──────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: #00d4ff;
        border: 1px solid #00d4ff;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00d4ff 0%, #0f3460 100%);
        color: #0e1117;
        border-color: #00d4ff;
    }

    /* ── Tabs ─────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #16213e;
        border-radius: 8px 8px 0 0;
        color: #a0a0a0;
        padding: 10px 20px;
        border: 1px solid #0f3460;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f3460;
        color: #00d4ff !important;
    }

    /* ── Expanders ────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background-color: #16213e;
        border-radius: 8px;
        color: #00d4ff;
    }

    /* ── File uploader ───────────────────────────────────────────────── */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #0f3460;
        border-radius: 10px;
        padding: 10px;
    }

    /* ── Progress bar accent ─────────────────────────────────────────── */
    .stProgress > div > div > div > div {
        background-color: #00d4ff;
    }

    /* ── Typography helpers ───────────────────────────────────────────── */
    .accent  { color: #00d4ff; }
    .dimmed  { color: #6c757d; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner="Loading object detection model …")
def get_detector() -> ObjectDetector:
    """Return a cached :class:`ObjectDetector` singleton.

    The detector is instantiated once per Streamlit server lifetime and
    shared across all sessions.

    Returns:
        A ready-to-use ``ObjectDetector`` instance.
    """
    logger.info("Initialising ObjectDetector (cached resource).")
    return ObjectDetector()


def _init_session_state() -> None:
    """Ensure all required keys exist in ``st.session_state``."""
    defaults: dict = {
        "detector": get_detector(),
        "detection_history": [],
        "fps_counter": FPSCounter(),
        "detection_logger": DetectionLogger(log_path=settings.detection_log_file),
        "trail_tracker": MotionTrailTracker(),
        "line_counter": LineCrossingCounter(),
        "roi_counter": ROICounter(),
        "webcam_running": False,
        "recording": False,
        "session_start": datetime.now(),
        "total_frames_processed": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════


def _model_display_name(size: ModelSize) -> str:
    """Return a human-friendly label for a model variant.

    Args:
        size: The ``ModelSize`` enum member.

    Returns:
        A descriptive string such as ``"YOLOv11n  ·  Nano – Fastest"``.
    """
    mapping = {
        ModelSize.NANO: "YOLOv11n  ·  Nano – Fastest",
        ModelSize.SMALL: "YOLOv11s  ·  Small – Balanced",
        ModelSize.MEDIUM: "YOLOv11m  ·  Medium – Accurate",
    }
    return mapping.get(size, size.value)


def process_detection(
    image: np.ndarray,
    use_tracking: bool = False,
) -> DetectionResult:
    """Run detection (and optional tracking) on a single frame.

    This is the shared pipeline used by every input mode (image, video,
    webcam).  It also updates the session-level detection history and
    frame counter.

    Args:
        image: BGR ``numpy`` array from OpenCV.
        use_tracking: If ``True``, call the tracker-enabled path.

    Returns:
        A ``DetectionResult`` with boxes, labels, and timing info.
    """
    detector: ObjectDetector = st.session_state["detector"]

    if use_tracking:
        result = detector.detect_and_track(image)
    else:
        result = detector.detect(image)

    # Append to history (keep latest 500 entries to avoid memory bloat)
    st.session_state["detection_history"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "num_objects": len(result.detections),
            "classes": [d.label for d in result.detections],
            "avg_confidence": (
                float(np.mean([d.confidence for d in result.detections]))
                if result.detections
                else 0.0
            ),
            "inference_ms": result.inference_time_ms,
        }
    )
    if len(st.session_state["detection_history"]) > 500:
        st.session_state["detection_history"] = st.session_state[
            "detection_history"
        ][-500:]

    st.session_state["total_frames_processed"] += 1

    # Optional detection logging
    if settings.enable_detection_logging:
        try:
            st.session_state["detection_logger"].log(result)
        except Exception:
            logger.debug("Detection logging failed – continuing silently.")

    return result


def display_detection_metrics(result: DetectionResult) -> None:
    """Render metric cards for a detection result.

    Displays four key performance indicators in a single row of
    ``st.metric`` widgets.

    Args:
        result: The ``DetectionResult`` to summarise.
    """
    detections = result.detections
    unique_classes = set(d.label for d in detections)
    avg_conf = (
        float(np.mean([d.confidence for d in detections]))
        if detections
        else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Objects", len(detections))
    with c2:
        st.metric("Unique Classes", len(unique_classes))
    with c3:
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    with c4:
        st.metric("Inference Time", f"{result.inference_time_ms:.1f} ms")


def display_detection_details(result: DetectionResult) -> None:
    """Render per-object details inside an expander.

    Each detection gets its own mini-section with a confidence progress
    bar and bounding-box coordinate display.

    Args:
        result: The ``DetectionResult`` containing individual detections.
    """
    if not result.detections:
        st.info("No objects detected.")
        return

    with st.expander(
        f"🔍 Detection Details — {len(result.detections)} object(s)", expanded=False
    ):
        for idx, det in enumerate(result.detections, 1):
            col_label, col_conf = st.columns([1, 2])
            with col_label:
                st.markdown(f"**#{idx}  {det.label}**")
                # Show bounding box as small text
                bbox = det.bbox
                st.caption(
                    f"Box: ({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})"
                )
            with col_conf:
                st.progress(
                    det.confidence,
                    text=f"Confidence: {det.confidence:.1%}",
                )
            st.divider()


def _class_distribution_chart(result: DetectionResult) -> None:
    """Display a bar chart of detected class counts.

    Args:
        result: The ``DetectionResult`` to visualise.
    """
    if not result.detections:
        return
    counts = Counter(d.label for d in result.detections)
    import pandas as pd

    df = pd.DataFrame(
        list(counts.items()), columns=["Class", "Count"]
    ).set_index("Class")
    st.bar_chart(df)


def _annotate_frame(
    frame: np.ndarray,
    result: DetectionResult,
    *,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_confidence: bool = True,
    show_trails: bool = False,
) -> np.ndarray:
    """Draw detections and optional motion trails on a frame.

    Args:
        frame: The BGR image to annotate (will be copied).
        result: Detection results to overlay.
        show_boxes: Whether to draw bounding boxes.
        show_labels: Whether to draw class labels.
        show_confidence: Whether to show confidence scores.
        show_trails: Whether to overlay motion trails.

    Returns:
        A new annotated BGR image.
    """
    annotated = draw_detections(
        frame,
        result,
        show_boxes=show_boxes,
        show_labels=show_labels,
        show_confidence=show_confidence,
    )
    if show_trails:
        try:
            annotated = st.session_state["trail_tracker"].draw(
                annotated, result
            )
        except Exception:
            logger.debug("Motion trail drawing failed – skipping.")
    return annotated


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🎯 AI Object Detection")
    st.caption("Powered by YOLOv11 & Streamlit")
    st.divider()

    # ── Model Settings ───────────────────────────────────────────────
    st.markdown("### 🎯 Model Settings")
    model_options = list(ModelSize)
    model_labels = [_model_display_name(m) for m in model_options]
    current_idx = model_options.index(settings.model_size)
    selected_label = st.selectbox(
        "Model",
        model_labels,
        index=current_idx,
        help="Larger models are more accurate but slower.",
    )
    selected_model = model_options[model_labels.index(selected_label)]

    if selected_model != settings.model_size:
        with st.spinner("Switching model …"):
            try:
                st.session_state["detector"].change_model(selected_model)
                settings.model_size = selected_model
                st.toast(f"Model switched to **{selected_model.value}**", icon="✅")
            except Exception as exc:
                st.error(f"Failed to change model: {exc}")

    st.divider()

    # ── Detection Settings ───────────────────────────────────────────
    st.markdown("### ⚙️ Detection Settings")
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=settings.confidence_threshold,
        step=0.05,
        help="Minimum confidence score to keep a detection.",
    )
    nms_thresh = st.slider(
        "NMS Threshold",
        min_value=0.10,
        max_value=1.00,
        value=settings.nms_threshold,
        step=0.05,
        help="IoU threshold for Non-Maximum Suppression.",
    )

    # Apply threshold changes
    if (
        conf_thresh != settings.confidence_threshold
        or nms_thresh != settings.nms_threshold
    ):
        try:
            st.session_state["detector"].update_thresholds(
                confidence=conf_thresh, nms=nms_thresh
            )
            settings.confidence_threshold = conf_thresh
            settings.nms_threshold = nms_thresh
        except Exception as exc:
            st.error(f"Threshold update failed: {exc}")

    st.divider()

    # ── Display Options ──────────────────────────────────────────────
    st.markdown("### 🎨 Display Options")
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    show_labels = st.checkbox("Show Labels", value=True)
    show_confidence = st.checkbox("Show Confidence", value=True)
    show_trails = st.checkbox("Show Motion Trails", value=False)

    st.divider()

    # ── Advanced Features ────────────────────────────────────────────
    st.markdown("### 📊 Advanced Features")
    enable_tracking = st.checkbox(
        "Enable Object Tracking",
        value=False,
        help="Use ByteTrack / BotSORT multi-object tracking.",
    )
    enable_line_counter = st.checkbox(
        "Enable Line Crossing Counter",
        value=False,
        help="Count objects crossing a virtual line.",
    )
    enable_roi_counter = st.checkbox(
        "Enable ROI Counting",
        value=False,
        help="Count objects inside a region of interest.",
    )

    st.divider()

    # ── Output Options ───────────────────────────────────────────────
    st.markdown("### 💾 Output Options")
    enable_logging = st.checkbox("Enable Detection Logging", value=True)
    save_outputs = st.checkbox("Save Outputs", value=True)
    settings.enable_detection_logging = enable_logging


# ═══════════════════════════════════════════════════════════════════════════
# Main Content – Tabs
# ═══════════════════════════════════════════════════════════════════════════

tab_image, tab_video, tab_webcam, tab_stats = st.tabs(
    ["📷 Image Detection", "🎬 Video Detection", "📹 Webcam Detection", "📊 Statistics"]
)

# ─── Tab 1: Image Detection ─────────────────────────────────────────────
with tab_image:
    st.markdown("## 📷 Image Detection")
    st.caption("Upload an image to detect objects using YOLOv11.")

    uploaded_img = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_uploader",
        help="Supported formats: JPG, PNG, BMP, WebP",
    )

    if uploaded_img is not None:
        # Decode the uploaded file into an OpenCV image
        file_bytes = np.frombuffer(uploaded_img.read(), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if original_img is None:
            st.error(
                "❌ Could not decode the uploaded file. "
                "Please upload a valid image."
            )
        else:
            # Validate the image
            try:
                validate_image(original_img)
            except ValueError as ve:
                st.warning(f"⚠️ Image validation: {ve}")

            # Run detection
            with st.status("Running detection …", expanded=True) as status:
                st.write("Pre-processing image …")
                result = process_detection(
                    original_img, use_tracking=enable_tracking
                )
                st.write(
                    f"Detected **{len(result.detections)}** object(s) "
                    f"in **{result.inference_time_ms:.1f} ms**."
                )
                status.update(label="Detection complete ✅", state="complete")

            # Annotated frame
            annotated_img = _annotate_frame(
                original_img,
                result,
                show_boxes=show_boxes,
                show_labels=show_labels,
                show_confidence=show_confidence,
                show_trails=show_trails,
            )

            # Side-by-side display
            col_orig, col_det = st.columns(2)
            with col_orig:
                st.markdown("#### Original")
                st.image(
                    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
            with col_det:
                st.markdown("#### Detected")
                st.image(
                    cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

            # Metrics row
            st.markdown("---")
            display_detection_metrics(result)

            # Per-object details
            display_detection_details(result)

            # Class distribution chart
            if result.detections:
                st.markdown("#### Class Distribution")
                _class_distribution_chart(result)

            # Line-crossing & ROI overlays (if enabled)
            if enable_line_counter and result.detections:
                try:
                    count = st.session_state["line_counter"].update(result)
                    st.info(f"🚶 Line crossing count: **{count}**")
                except Exception:
                    pass

            if enable_roi_counter and result.detections:
                try:
                    count = st.session_state["roi_counter"].update(result)
                    st.info(f"📍 Objects in ROI: **{count}**")
                except Exception:
                    pass

            # Download button for annotated image
            st.markdown("---")
            _, png_buffer = cv2.imencode(".png", annotated_img)
            st.download_button(
                label="⬇️  Download Annotated Image",
                data=png_buffer.tobytes(),
                file_name=f"detected_{datetime.now():%Y%m%d_%H%M%S}.png",
                mime="image/png",
            )

# ─── Tab 2: Video Detection ─────────────────────────────────────────────
with tab_video:
    st.markdown("## 🎬 Video Detection")
    st.caption("Upload a video for frame-by-frame object detection.")

    uploaded_vid = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_uploader",
        help="Supported formats: MP4, AVI, MOV, MKV",
    )

    if uploaded_vid is not None:
        # Save uploaded video to a temporary file for OpenCV
        tmp_input = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        )
        tmp_input.write(uploaded_vid.read())
        tmp_input.flush()
        tmp_input_path = tmp_input.name
        tmp_input.close()

        cap = cv2.VideoCapture(tmp_input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.info(
            f"📹 **{uploaded_vid.name}** — "
            f"{total_frames} frames, {fps:.1f} FPS, "
            f"{width}×{height} px"
        )

        if st.button("▶️  Process Video", key="process_video_btn"):
            # Prepare output writer
            tmp_output = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            )
            tmp_output_path = tmp_output.name
            tmp_output.close()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                tmp_output_path, fourcc, fps, (width, height)
            )

            cap = cv2.VideoCapture(tmp_input_path)
            progress_bar = st.progress(0, text="Processing video …")
            frame_display = st.empty()
            col_fps, col_objects = st.columns(2)
            metric_fps = col_fps.empty()
            metric_objects = col_objects.empty()

            vid_fps_counter = FPSCounter()
            all_detections: list[int] = []
            total_inference_ms: float = 0.0
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                vid_fps_counter.tick()
                result = process_detection(frame, use_tracking=enable_tracking)

                annotated = _annotate_frame(
                    frame,
                    result,
                    show_boxes=show_boxes,
                    show_labels=show_labels,
                    show_confidence=show_confidence,
                    show_trails=show_trails,
                )

                writer.write(annotated)
                all_detections.append(len(result.detections))
                total_inference_ms += result.inference_time_ms
                frame_idx += 1

                # Update UI every 3 frames (for performance)
                if frame_idx % 3 == 0 or frame_idx == total_frames:
                    pct = min(frame_idx / max(total_frames, 1), 1.0)
                    progress_bar.progress(
                        pct,
                        text=f"Frame {frame_idx}/{total_frames}",
                    )
                    frame_display.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_idx}",
                        use_container_width=True,
                    )
                    metric_fps.metric(
                        "Processing FPS",
                        f"{vid_fps_counter.get_fps():.1f}",
                    )
                    metric_objects.metric(
                        "Objects (current)",
                        len(result.detections),
                    )

            cap.release()
            writer.release()

            progress_bar.progress(1.0, text="✅ Processing complete!")
            st.toast("Video processing complete!", icon="🎬")

            # Summary statistics
            st.markdown("### 📊 Processing Summary")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Total Frames", frame_idx)
            with s2:
                st.metric(
                    "Avg Objects/Frame",
                    f"{np.mean(all_detections):.1f}" if all_detections else "0",
                )
            with s3:
                st.metric(
                    "Avg Inference",
                    f"{total_inference_ms / max(frame_idx, 1):.1f} ms",
                )
            with s4:
                st.metric(
                    "Avg FPS",
                    f"{vid_fps_counter.get_fps():.1f}",
                )

            # Download button for processed video
            output_bytes = Path(tmp_output_path).read_bytes()
            st.download_button(
                label="⬇️  Download Processed Video",
                data=output_bytes,
                file_name=f"detected_{datetime.now():%Y%m%d_%H%M%S}.mp4",
                mime="video/mp4",
            )

# ─── Tab 3: Webcam Detection ────────────────────────────────────────────
with tab_webcam:
    st.markdown("## 📹 Webcam Detection")
    st.caption("Real-time object detection using your camera feed.")

    st.warning(
        "⚠️ Webcam access requires a local Streamlit deployment. "
        "It may not work in cloud/remote environments."
    )

    col_start, col_screenshot, col_record = st.columns(3)

    with col_start:
        if not st.session_state["webcam_running"]:
            start_btn = st.button("▶️  Start Webcam", key="start_webcam")
        else:
            start_btn = False
            stop_btn = st.button("⏹️  Stop Webcam", key="stop_webcam")
            if stop_btn:
                st.session_state["webcam_running"] = False
                st.rerun()

    with col_screenshot:
        screenshot_btn = st.button(
            "📸 Screenshot",
            key="screenshot_btn",
            disabled=not st.session_state["webcam_running"],
        )

    with col_record:
        if st.session_state["recording"]:
            rec_label = "⏹️  Stop Recording"
        else:
            rec_label = "🔴  Start Recording"
        record_btn = st.button(
            rec_label,
            key="record_btn",
            disabled=not st.session_state["webcam_running"],
        )
        if record_btn:
            st.session_state["recording"] = not st.session_state["recording"]

    # Centered layout for webcam stream with comfortable standard size
    cam_col_left, cam_col_center, cam_col_right = st.columns([1, 4, 1])
    with cam_col_center:
        webcam_frame_placeholder = st.empty()
        webcam_metrics_placeholder = st.empty()

    if start_btn:
        st.session_state["webcam_running"] = True

    if st.session_state["webcam_running"]:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error(
                "❌ Could not access webcam. Make sure:\n"
                "- Your camera is connected and not in use.\n"
                "- You are running Streamlit locally.\n"
                "- Camera permissions are granted."
            )
            st.session_state["webcam_running"] = False
        else:
            webcam_fps = FPSCounter()
            video_writer: Optional[cv2.VideoWriter] = None
            tmp_rec_path: Optional[str] = None

            # Set up recording writer if recording was toggled on
            if st.session_state["recording"]:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                tmp_rec = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                )
                tmp_rec_path = tmp_rec.name
                tmp_rec.close()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    tmp_rec_path, fourcc, 20.0, (w, h)
                )

            last_frame: Optional[np.ndarray] = None

            while st.session_state["webcam_running"]:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Lost webcam feed.")
                    break

                webcam_fps.update()
                result = process_detection(frame, use_tracking=enable_tracking)
                annotated = _annotate_frame(
                    frame,
                    result,
                    show_boxes=show_boxes,
                    show_labels=show_labels,
                    show_confidence=show_confidence,
                    show_trails=show_trails,
                )

                last_frame = annotated.copy()

                # Write to recording if active
                if video_writer is not None:
                    video_writer.write(annotated)

                # Handle screenshot request
                if screenshot_btn and last_frame is not None:
                    save_path = save_screenshot(
                        last_frame, output_dir=settings.outputs_dir
                    )
                    st.toast(f"Screenshot saved: {save_path}", icon="📸")
                    screenshot_btn = False  # Only fire once

                # Update display
                webcam_frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Live Feed",
                    use_container_width=True,
                )

                # Metrics
                class_counts = Counter(d.label for d in result.detections)
                with webcam_metrics_placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("FPS", f"{webcam_fps.get_fps():.1f}")
                    with m2:
                        st.metric("Objects", len(result.detections))
                    with m3:
                        top_classes = ", ".join(
                            f"{cls} ({cnt})"
                            for cls, cnt in class_counts.most_common(3)
                        )
                        st.metric("Top Classes", top_classes or "—")

            # Cleanup
            cap.release()
            if video_writer is not None:
                video_writer.release()
                if tmp_rec_path:
                    rec_bytes = Path(tmp_rec_path).read_bytes()
                    st.download_button(
                        label="⬇️  Download Recording",
                        data=rec_bytes,
                        file_name=f"recording_{datetime.now():%Y%m%d_%H%M%S}.mp4",
                        mime="video/mp4",
                    )

# ─── Tab 4: Statistics Dashboard ────────────────────────────────────────
with tab_stats:
    st.markdown("## 📊 Statistics Dashboard")
    st.caption("Session analytics and detection history.")

    history = st.session_state["detection_history"]

    # ── Session overview metrics ─────────────────────────────────────
    session_duration = datetime.now() - st.session_state["session_start"]
    duration_str = str(session_duration).split(".")[0]  # drop microseconds

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Detections", len(history))
    with s2:
        st.metric("Session Duration", duration_str)
    with s3:
        avg_fps = (
            float(np.mean([h["inference_ms"] for h in history]))
            if history
            else 0.0
        )
        st.metric(
            "Avg Inference",
            f"{avg_fps:.1f} ms" if history else "—",
        )
    with s4:
        st.metric(
            "Frames Processed",
            st.session_state["total_frames_processed"],
        )

    st.divider()

    if history:
        import pandas as pd

        df = pd.DataFrame(history)

        # ── Detection history table ──────────────────────────────────
        st.markdown("### 📋 Detection History")
        display_df = df[
            ["timestamp", "num_objects", "avg_confidence", "inference_ms"]
        ].copy()
        display_df.columns = [
            "Timestamp",
            "Objects",
            "Avg Confidence",
            "Inference (ms)",
        ]
        st.dataframe(
            display_df.tail(50).sort_index(ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        # ── Class distribution ───────────────────────────────────────
        st.markdown("### 🏷️ Class Distribution (All Detections)")
        all_classes: list[str] = []
        for row in history:
            all_classes.extend(row["classes"])

        if all_classes:
            class_counter = Counter(all_classes)
            class_df = pd.DataFrame(
                class_counter.most_common(20),
                columns=["Class", "Count"],
            ).set_index("Class")
            st.bar_chart(class_df)

        # ── Inference time chart ─────────────────────────────────────
        st.markdown("### ⏱️ Inference Time Trend")
        inf_df = df[["inference_ms"]].copy()
        inf_df.columns = ["Inference (ms)"]
        st.line_chart(inf_df.tail(100))

        st.divider()

        # ── Recent detection log entries ─────────────────────────────
        st.markdown("### 📝 Recent Detection Logs")
        try:
            stats_text = format_detection_stats(history[-10:])
            st.code(stats_text, language="text")
        except Exception:
            st.info("No formatted logs available.")

    else:
        st.info(
            "📭 No detections yet. Use the Image, Video, or Webcam tabs "
            "to start detecting objects."
        )

    # ── Action buttons ───────────────────────────────────────────────
    st.divider()
    col_clear, col_export = st.columns(2)

    with col_clear:
        if st.button("🗑️  Clear History", key="clear_history"):
            st.session_state["detection_history"] = []
            st.session_state["total_frames_processed"] = 0
            st.session_state["session_start"] = datetime.now()
            st.toast("History cleared!", icon="🗑️")
            st.rerun()

    with col_export:
        if history:
            import json as _json

            log_json = _json.dumps(history, indent=2, default=str)
            st.download_button(
                label="📥  Export Logs (JSON)",
                data=log_json,
                file_name=f"detection_logs_{datetime.now():%Y%m%d_%H%M%S}.json",
                mime="application/json",
            )
        else:
            st.button("📥  Export Logs (JSON)", disabled=True, key="export_disabled")

# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.divider()
st.sidebar.caption(
    f"**Model:** {settings.model_size.value}  \n"
    f"**Confidence:** {settings.confidence_threshold:.0%}  \n"
    f"**NMS:** {settings.nms_threshold:.0%}  \n"
    f"**Device:** {settings.device or 'auto'}"
)
st.sidebar.caption(
    f"Session started: {st.session_state['session_start']:%H:%M:%S}"
)
