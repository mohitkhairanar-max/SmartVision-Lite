"""Streamlit interface for SmartVision Lite.

The module deliberately keeps all user interface work inside main(). Importing
it is therefore safe in tests and never opens a camera, downloads a model, or
starts media processing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import inspect
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

from detection import DetectionResult, ImageProcessor, ObjectDetector, VideoProcessor


APP_TITLE = "SmartVision Lite"
APP_SUBTITLE = "Real-Time Object Detection"
PRETRAINED_MODELS = {
    "YOLO11 Nano - recommended": "yolo11n.pt",
    "YOLOv8 Nano - compatibility option": "yolov8n.pt",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
DETECTION_COLUMNS = [
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]


@dataclass(frozen=True)
class DetectorSettings:
    """User-selectable inference settings."""

    model_source: str
    model_path: str
    confidence: float
    iou: float
    image_size: int
    device: str

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return sha256(payload).hexdigest()


def _inject_styles() -> None:
    """Apply a small, self-contained visual theme."""

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .sv-hero {
            padding: 2.1rem 2.25rem;
            border: 1px solid rgba(124, 58, 237, .24);
            border-radius: 22px;
            background:
                radial-gradient(circle at 92% 8%, rgba(14,165,233,.23), transparent 28%),
                linear-gradient(135deg, rgba(124,58,237,.16), rgba(15,23,42,.04));
            margin-bottom: 1.5rem;
        }
        .sv-kicker {
            display: inline-block;
            padding: .3rem .7rem;
            border-radius: 999px;
            background: rgba(124,58,237,.13);
            color: #8b5cf6;
            font-size: .77rem;
            font-weight: 750;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .sv-hero h1 {
            font-size: clamp(2rem, 5vw, 3.65rem);
            line-height: 1.03;
            letter-spacing: -.045em;
            margin: .8rem 0 .65rem;
        }
        .sv-hero p {
            max-width: 760px;
            font-size: 1.08rem;
            opacity: .82;
            margin: 0;
        }
        .sv-card {
            min-height: 150px;
            padding: 1.1rem 1.15rem;
            border: 1px solid rgba(148,163,184,.24);
            border-radius: 16px;
            background: rgba(148,163,184,.055);
        }
        .sv-card h3 { margin: .2rem 0 .45rem; font-size: 1.04rem; }
        .sv-card p { margin: 0; opacity: .78; font-size: .92rem; }
        .sv-section {
            margin-top: 1.35rem;
            margin-bottom: .35rem;
        }
        .sv-step {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.7rem;
            height: 1.7rem;
            margin-right: .45rem;
            border-radius: 50%;
            color: white;
            background: #7c3aed;
            font-weight: 750;
            font-size: .8rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,.22);
            border-radius: 14px;
            padding: .8rem 1rem;
            background: rgba(148,163,184,.045);
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(148,163,184,.16);
        }
        .sv-footer {
            margin-top: 2.6rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(148,163,184,.18);
            opacity: .62;
            font-size: .82rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _stretch(component: Any) -> dict[str, Any]:
    """Use the non-deprecated full-width option across Streamlit versions."""

    try:
        parameters = inspect.signature(component).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "width" in parameters:
        return {"width": "stretch"}
    if "use_container_width" in parameters:
        return {"use_container_width": True}
    if "use_column_width" in parameters:
        return {"use_column_width": True}
    return {}


def _render_hero(title: str, description: str, kicker: str) -> None:
    st.markdown(
        f"""
        <section class="sv-hero">
            <span class="sv-kicker">{kicker}</span>
            <h1>{title}</h1>
            <p>{description}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _cuda_status() -> tuple[bool, str]:
    """Return CUDA availability and a human-readable device label."""

    try:
        import torch

        available = bool(torch.cuda.is_available())
        if available:
            return True, str(torch.cuda.get_device_name(0))
        return False, "No CUDA-capable GPU detected"
    except Exception:
        return False, "PyTorch could not query CUDA"


def _sidebar() -> tuple[str, DetectorSettings, str | None]:
    """Render navigation/settings and return a validation message if needed."""

    with st.sidebar:
        st.markdown("## SmartVision Lite")
        st.caption("A small, practical YOLO workspace")
        page = st.radio(
            "Navigate",
            ("Home", "Image Detection", "Video Detection", "Live Webcam"),
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### Detection controls")
        model_source = st.selectbox("Model source", ("Pretrained", "Custom best.pt"))

        if model_source == "Pretrained":
            model_label = st.selectbox("Pretrained model", tuple(PRETRAINED_MODELS))
            model_path = PRETRAINED_MODELS[model_label]
            st.caption("The weights download automatically on first use.")
        else:
            custom_path = st.text_input(
                "Custom model path",
                value="models/best.pt",
                help="Enter a local path to a trusted Ultralytics .pt model.",
            )
            model_path = custom_path.strip().strip('"')

        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Higher values keep fewer, more confident detections.",
        )
        iou = st.slider(
            "IoU threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Controls overlap suppression between competing boxes.",
        )
        image_size = st.select_slider(
            "Inference resolution",
            options=(320, 480, 640, 960, 1280),
            value=640,
            help="Larger images may improve small-object detection but take longer.",
        )
        device_label = st.selectbox("Device", ("Auto", "CPU", "CUDA"))
        device = {"Auto": "auto", "CPU": "cpu", "CUDA": "cuda"}[device_label]

        cuda_available, cuda_label = _cuda_status()
        if device == "cuda":
            if cuda_available:
                st.success(f"CUDA ready: {cuda_label}")
            else:
                st.error(cuda_label)
        elif device == "auto":
            selected = cuda_label if cuda_available else "CPU"
            st.caption(f"Auto will use: {selected}")

        st.divider()
        st.caption("Processing stays on this machine. Uploaded media is not committed.")

    settings = DetectorSettings(
        model_source=model_source,
        model_path=model_path,
        confidence=float(confidence),
        iou=float(iou),
        image_size=int(image_size),
        device=device,
    )
    return page, settings, _settings_error(settings, cuda_available)


def _settings_error(
    settings: DetectorSettings, cuda_available: bool | None = None
) -> str | None:
    """Validate model and device choices without loading weights."""

    if settings.device == "cuda":
        if cuda_available is None:
            cuda_available, _ = _cuda_status()
        if not cuda_available:
            return "CUDA was selected, but PyTorch cannot access a CUDA GPU. Choose Auto or CPU."

    if settings.model_source == "Custom best.pt":
        if not settings.model_path:
            return "Enter the path to a custom best.pt model."
        candidate = Path(settings.model_path).expanduser()
        if candidate.suffix.lower() != ".pt":
            return "Custom models must be Ultralytics .pt weight files."
        if not candidate.is_file():
            return f"Custom model not found: {candidate}"
    return None


@st.cache_resource(show_spinner=False)
def _get_detector(
    model_path: str,
    confidence: float,
    iou: float,
    image_size: int,
    device: str,
) -> ObjectDetector:
    """Create one lazy detector per unique settings combination."""

    return ObjectDetector(
        model_path=model_path,
        confidence=confidence,
        iou=iou,
        image_size=image_size,
        device=device,
    )


def _detector(settings: DetectorSettings) -> ObjectDetector:
    return _get_detector(
        settings.model_path,
        settings.confidence,
        settings.iou,
        settings.image_size,
        settings.device,
    )


def _friendly_error(error: Exception, context: str) -> str:
    """Translate common model/media errors into useful next steps."""

    detail = str(error).strip() or error.__class__.__name__
    lowered = detail.lower()

    if "cuda" in lowered or "cudnn" in lowered:
        guidance = "CUDA could not be used. Choose Auto or CPU, then try again."
    elif any(token in lowered for token in ("download", "connection", "network", "url")):
        guidance = (
            "The pretrained model could not be downloaded. Check the internet "
            "connection, or select a local custom model."
        )
    elif any(token in lowered for token in ("model", "weight", ".pt")):
        guidance = (
            "The model could not be loaded. Check that the weight file exists and "
            "was created by a compatible Ultralytics version."
        )
    elif any(
        token in lowered
        for token in ("codec", "video", "frame", "corrupt", "decode", "capture")
    ):
        guidance = (
            "The media could not be decoded. Try a valid, non-corrupted file; MP4 "
            "with H.264 or MPEG-4 encoding is usually the most portable."
        )
    else:
        guidance = f"{context} failed. Review the selected file and settings, then retry."
    return f"{guidance}\n\nTechnical detail: {detail}"


def _media_key(data: bytes, settings: DetectorSettings) -> str:
    digest = sha256()
    digest.update(data)
    digest.update(settings.fingerprint.encode("ascii"))
    return digest.hexdigest()


def _safe_stem(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_")
    return stem[:80] or "detection"


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    )
    if not ok:
        raise RuntimeError("OpenCV could not encode the annotated image.")
    return encoded.tobytes()


def _rows(result: DetectionResult) -> list[dict[str, Any]]:
    return [dict(row) for row in result.rows]


def _detections_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=DETECTION_COLUMNS)
    frame = pd.DataFrame(rows)
    preferred = [column for column in DETECTION_COLUMNS if column in frame.columns]
    remaining = [column for column in frame.columns if column not in preferred]
    return frame[preferred + remaining]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _result_json(result: DetectionResult, filename: str) -> str:
    payload = {
        "source": filename,
        "summary": {
            "total_detections": result.total_detections,
            "class_counts": dict(result.class_counts),
            "inference_time_ms": result.inference_time_ms,
            "source_shape": result.source_shape,
        },
        "detections": _rows(result),
    }
    return json.dumps(payload, indent=2, default=_json_default)


def _render_detection_metrics(result: DetectionResult) -> None:
    inference_ms = float(result.inference_time_ms)
    inference_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    columns = st.columns(4)
    columns[0].metric("Detections", f"{result.total_detections:,}")
    columns[1].metric("Classes", f"{len(result.class_counts):,}")
    columns[2].metric("Inference time", f"{inference_ms:.1f} ms")
    columns[3].metric(
        "Inference FPS",
        f"{inference_fps:.1f}" if inference_fps > 0 else "-",
        help="Model inference only; display and file handling are excluded.",
    )


def _render_class_counts(class_counts: Mapping[str, int]) -> None:
    st.markdown("#### Class-wise count")
    if not class_counts:
        st.info("No objects met the selected confidence threshold.")
        return
    count_frame = pd.DataFrame(
        [
            {"Class": class_name, "Count": int(count)}
            for class_name, count in sorted(
                class_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]
    )
    st.dataframe(count_frame, hide_index=True, **_stretch(st.dataframe))


def _render_home() -> None:
    _render_hero(
        "See what your camera sees.",
        "Run lightweight YOLO object detection on images, recorded video, or a "
        "live browser camera - through one beginner-friendly interface.",
        "SmartVision Lite - Real-time object detection",
    )

    columns = st.columns(3)
    cards = (
        (
            "IMAGE",
            "Image detection",
            "Inspect one JPG or PNG, review every confidence score, and export the result.",
        ),
        (
            "VIDEO",
            "Video detection",
            "Process clips frame by frame with progress, aggregate counts, and measured FPS.",
        ),
        (
            "LIVE",
            "Live webcam",
            "Use the browser camera for responsive detection without opening a desktop window.",
        ),
    )
    for column, (icon, title, body) in zip(columns, cards):
        column.markdown(
            f'<div class="sv-card"><div>{icon}</div><h3>{title}</h3><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<h2 class="sv-section">Start in three steps</h2>', unsafe_allow_html=True)
    steps = st.columns(3)
    steps[0].markdown('<span class="sv-step">1</span>Choose a mode', unsafe_allow_html=True)
    steps[0].caption("Use the navigation in the left sidebar.")
    steps[1].markdown('<span class="sv-step">2</span>Tune the model', unsafe_allow_html=True)
    steps[1].caption("The defaults are a good starting point for most media.")
    steps[2].markdown('<span class="sv-step">3</span>Run detection', unsafe_allow_html=True)
    steps[2].caption("Review measured results and download useful outputs.")

    with st.expander("What do the metrics mean?"):
        st.markdown(
            """
            - **Confidence** estimates how strongly the model supports a detection.
            - **IoU** measures box overlap and helps suppress duplicate detections.
            - **Inference time** is the model's processing time for one image or frame.
            - **FPS** is measured throughput. It depends on hardware, media size, and settings.
            - **Precision** asks how many predicted objects were correct; **recall** asks how
              many real objects were found.
            - **mAP** summarizes precision/recall across classes and overlap thresholds.
            """
        )

    st.info(
        "First run note: a pretrained model may need to download its weights. "
        "No fixed accuracy or FPS is promised - both depend on the model, data, and hardware.",
    )


def _validate_image(data: bytes, filename: str) -> Image.Image:
    extension = Path(filename).suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image type. Upload JPG, JPEG, or PNG.")
    try:
        with Image.open(BytesIO(data)) as probe:
            if probe.width * probe.height > 50_000_000:
                raise ValueError("Image dimensions are too large for safe processing.")
            probe.verify()
        with Image.open(BytesIO(data)) as opened:
            image = opened.convert("RGB")
            image.load()
            return image
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise ValueError(
            "The uploaded image is invalid, corrupted, or too large."
        ) from error


def _render_image(settings: DetectorSettings, settings_error: str | None) -> None:
    _render_hero(
        "Image Detection",
        "Upload a photo, run YOLO once, then inspect and export every real detection.",
        "JPG - JPEG - PNG",
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=("jpg", "jpeg", "png"),
        accept_multiple_files=False,
        key="image_upload",
    )
    if uploaded is None:
        st.markdown(
            '<div class="sv-card"><h3>Ready when you are</h3>'
            '<p>Upload a supported image to reveal the preview and detection controls.</p></div>',
            unsafe_allow_html=True,
        )
        return

    data = uploaded.getvalue()
    try:
        image = _validate_image(data, uploaded.name)
    except ValueError as error:
        st.error(str(error))
        return

    preview, action = st.columns((1.55, 1), gap="large")
    with preview:
        st.image(
            image,
            caption=f"Original - {uploaded.name}",
            **_stretch(st.image),
        )
    with action:
        st.markdown("#### Ready to detect")
        st.write(f"**Image size:** {image.width} x {image.height} pixels")
        st.write(f"**Model:** {settings.model_path}")
        st.write(f"**Threshold:** {settings.confidence:.0%} confidence")
        if settings_error:
            st.error(settings_error)
        run = st.button(
            "Run image detection",
            type="primary",
            disabled=bool(settings_error),
            **_stretch(st.button),
        )

    key = _media_key(data, settings)
    if run:
        try:
            with st.spinner("Loading the model and detecting objects..."):
                result = ImageProcessor(_detector(settings)).process(data)
            st.session_state["image_detection_run"] = {
                "key": key,
                "filename": uploaded.name,
                "result": result,
            }
        except Exception as error:
            st.session_state.pop("image_detection_run", None)
            st.error(_friendly_error(error, "Image detection"))
            return

    stored = st.session_state.get("image_detection_run")
    if not stored or stored.get("key") != key:
        return

    result: DetectionResult = stored["result"]
    st.markdown("### Detection result")
    _render_detection_metrics(result)

    output, details = st.columns((1.55, 1), gap="large")
    with output:
        st.image(
            _bgr_to_rgb(result.annotated_image),
            caption="Annotated result",
            **_stretch(st.image),
        )
    with details:
        _render_class_counts(result.class_counts)

    rows = _rows(result)
    frame = _detections_frame(rows)
    st.markdown("#### Detection details")
    if frame.empty:
        st.info(
            "No objects were detected. Try lowering the confidence threshold or "
            "using a clearer image.",
        )
    else:
        display_frame = frame.copy()
        if "confidence" in display_frame:
            display_frame["confidence"] = display_frame["confidence"].map(
                lambda value: f"{float(value):.3f}"
            )
        st.dataframe(
            display_frame,
            hide_index=True,
            **_stretch(st.dataframe),
        )

    try:
        annotated_bytes = _encode_jpeg(result.annotated_image)
    except RuntimeError as error:
        st.warning(str(error))
        annotated_bytes = b""

    stem = _safe_stem(uploaded.name)
    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download annotated image",
        data=annotated_bytes,
        file_name=f"{stem}_detected.jpg",
        mime="image/jpeg",
        disabled=not annotated_bytes,
        **_stretch(st.download_button),
    )
    download_columns[1].download_button(
        "Export CSV",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_detections.csv",
        mime="text/csv",
        **_stretch(st.download_button),
    )
    download_columns[2].download_button(
        "Export JSON",
        data=_result_json(result, uploaded.name).encode("utf-8"),
        file_name=f"{stem}_detections.json",
        mime="application/json",
        **_stretch(st.download_button),
    )


def _validate_video(data: bytes, filename: str) -> None:
    extension = Path(filename).suffix.lower()
    if extension not in VIDEO_EXTENSIONS:
        raise ValueError("Unsupported video type. Upload MP4, AVI, or MOV.")
    if not data:
        raise ValueError("The uploaded video is empty.")


def _video_summary_json(result: Any, filename: str) -> bytes:
    payload = {
        "source": filename,
        "frame_count": result.frame_count,
        "total_detection_events": result.total_detections,
        "class_counts": dict(result.class_counts),
        "elapsed_seconds": result.elapsed_seconds,
        "average_fps": result.average_fps,
        "average_inference_time_ms": result.average_inference_time_ms,
        "stopped_early": result.stopped_early,
        "note": (
            "Counts are per-frame detection events, not unique objects tracked "
            "across the video."
        ),
    }
    return json.dumps(payload, indent=2, default=_json_default).encode("utf-8")


def _render_video(settings: DetectorSettings, settings_error: str | None) -> None:
    _render_hero(
        "Video Detection",
        "Process a recorded clip frame by frame, follow progress, and download an "
        "annotated MP4 with measured performance.",
        "MP4 - AVI - MOV",
    )

    uploaded = st.file_uploader(
        "Choose a video",
        type=("mp4", "avi", "mov"),
        accept_multiple_files=False,
        key="video_upload",
    )
    if uploaded is None:
        st.markdown(
            '<div class="sv-card"><h3>Choose a short clip</h3>'
            '<p>Start with a small video while learning how resolution and device '
            'selection affect processing time.</p></div>',
            unsafe_allow_html=True,
        )
        return

    data = uploaded.getvalue()
    try:
        _validate_video(data, uploaded.name)
    except ValueError as error:
        st.error(str(error))
        return

    extension = Path(uploaded.name).suffix.lower()
    if extension == ".mp4":
        st.video(data, format="video/mp4")
    else:
        st.caption(
            f"{extension.upper()[1:]} preview depends on browser codec support. "
            "The file can still be processed by OpenCV."
        )

    st.caption(
        "Video processing is local and sequential. Longer clips and larger inference "
        "resolutions take more time."
    )
    if settings_error:
        st.error(settings_error)
    run = st.button(
        "Process video",
        type="primary",
        disabled=bool(settings_error),
        **_stretch(st.button),
    )

    key = _media_key(data, settings)
    if run:
        progress = st.progress(0.0, text="Preparing video...")
        status = st.empty()

        def on_progress(fraction: float) -> None:
            value = max(0.0, min(1.0, float(fraction)))
            progress.progress(value, text=f"Processing frames... {value:.0%}")

        try:
            with tempfile.TemporaryDirectory(prefix="smartvision_lite_") as directory:
                temporary = Path(directory)
                input_path = temporary / f"input{extension}"
                output_path = temporary / "annotated.mp4"
                input_path.write_bytes(data)

                processor = VideoProcessor(_detector(settings))
                started = time.perf_counter()
                result = processor.process_video(
                    input_path,
                    output_path=output_path,
                    progress_callback=on_progress,
                )
                wall_time = time.perf_counter() - started
                processed_path = (
                    Path(result.output_path)
                    if result.output_path is not None
                    else output_path
                )
                if not processed_path.is_file() or processed_path.stat().st_size == 0:
                    raise RuntimeError("The processed video output was not created.")
                output_bytes = processed_path.read_bytes()

            progress.progress(1.0, text="Video processing complete")
            status.success(
                f"Processed {result.frame_count:,} frames in {wall_time:.1f} seconds.",
            )
            st.session_state["video_detection_run"] = {
                "key": key,
                "filename": uploaded.name,
                "result": result,
                "output_bytes": output_bytes,
            }
        except Exception as error:
            progress.empty()
            status.empty()
            st.session_state.pop("video_detection_run", None)
            st.error(_friendly_error(error, "Video processing"))
            return

    stored = st.session_state.get("video_detection_run")
    if not stored or stored.get("key") != key:
        return

    result = stored["result"]
    output_bytes: bytes = stored["output_bytes"]
    st.markdown("### Video result")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Frames", f"{result.frame_count:,}")
    metric_columns[1].metric("Detection events", f"{result.total_detections:,}")
    metric_columns[2].metric("Average FPS", f"{result.average_fps:.1f}")
    metric_columns[3].metric(
        "Avg. inference", f"{result.average_inference_time_ms:.1f} ms"
    )
    metric_columns[4].metric("Elapsed", f"{result.elapsed_seconds:.1f} s")
    st.caption(
        "Detection events are counted per frame; the same physical object may be "
        "counted in many frames. Object tracking is not enabled."
    )

    video_column, counts_column = st.columns((1.55, 1), gap="large")
    with video_column:
        st.video(output_bytes, format="video/mp4")
    with counts_column:
        _render_class_counts(result.class_counts)

    stem = _safe_stem(uploaded.name)
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download processed video",
        data=output_bytes,
        file_name=f"{stem}_detected.mp4",
        mime="video/mp4",
        **_stretch(st.download_button),
    )
    downloads[1].download_button(
        "Export summary JSON",
        data=_video_summary_json(result, uploaded.name),
        file_name=f"{stem}_video_summary.json",
        mime="application/json",
        **_stretch(st.download_button),
    )


class _LiveVideoProcessor:
    """streamlit-webrtc video callback with thread-safe observable metrics."""

    def __init__(self, detector: ObjectDetector, av_module: Any) -> None:
        self.detector = detector
        self.av = av_module
        self.lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.last_frame_at: float | None = None
        self.started_at = time.perf_counter()
        self.frame_count = 0
        self.current_fps = 0.0
        self.average_fps = 0.0
        self.inference_time_ms = 0.0
        self.total_detections = 0
        self.class_counts: dict[str, int] = {}
        self.error: str | None = None

    def recv(self, frame: Any) -> Any:
        source = frame.to_ndarray(format="bgr24")
        now = time.perf_counter()
        try:
            # Async WebRTC delivery can overlap callbacks. Ultralytics model
            # instances are not guaranteed to be re-entrant, so inference is
            # serialized while metric reads remain independently responsive.
            with self.inference_lock:
                result = self.detector.detect(source)
            annotated = result.annotated_image.copy()
            elapsed = max(now - self.started_at, 1e-9)
            current_fps = (
                1.0 / max(now - self.last_frame_at, 1e-9)
                if self.last_frame_at is not None
                else 0.0
            )
            frame_count = self.frame_count + 1
            average_fps = frame_count / elapsed
            cv2.putText(
                annotated,
                f"FPS {current_fps:.1f} | Objects {result.total_detections}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 240, 120),
                2,
                cv2.LINE_AA,
            )
            with self.lock:
                self.last_frame_at = now
                self.frame_count = frame_count
                self.current_fps = current_fps
                self.average_fps = average_fps
                self.inference_time_ms = float(result.inference_time_ms)
                self.total_detections = int(result.total_detections)
                self.class_counts = dict(result.class_counts)
                self.error = None
            return self.av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as error:
            message = _friendly_error(error, "Live detection")
            compact = message.splitlines()[0][:95]
            cv2.putText(
                source,
                compact,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (30, 30, 240),
                2,
                cv2.LINE_AA,
            )
            with self.lock:
                self.error = message
            return self.av.VideoFrame.from_ndarray(source, format="bgr24")

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "frame_count": self.frame_count,
                "current_fps": self.current_fps,
                "average_fps": self.average_fps,
                "inference_time_ms": self.inference_time_ms,
                "total_detections": self.total_detections,
                "class_counts": dict(self.class_counts),
                "error": self.error,
            }


def _render_live(settings: DetectorSettings, settings_error: str | None) -> None:
    _render_hero(
        "Live Webcam Detection",
        "Grant browser camera access, press Start, and see boxes, labels, confidence, "
        "FPS, and counts update on live video.",
        "Browser camera - Start and stop anytime",
    )

    instructions, privacy = st.columns(2)
    with instructions:
        st.markdown("#### How to start")
        st.markdown(
            "1. Confirm the model controls in the sidebar.\n"
            "2. Press **Start** below and allow camera permission.\n"
            "3. Press **Stop** before changing models or leaving this page."
        )
    with privacy:
        st.markdown("#### Camera troubleshooting")
        st.markdown(
            "- Close apps already using the camera.\n"
            "- Allow camera access for this browser.\n"
            "- Use localhost or HTTPS when prompted.\n"
            "- For a desktop window, run **python webcam_demo.py**."
        )

    if settings_error:
        st.error(settings_error)
        return

    try:
        import av
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.warning(
            "Live browser mode needs the optional streamlit-webrtc and av packages. "
            "Install the project requirements, restart Streamlit, and try again.",
        )
        st.code("python -m pip install streamlit-webrtc av", language="powershell")
        return

    try:
        detector = _detector(settings)
    except Exception as error:
        st.error(_friendly_error(error, "Model setup"))
        return

    try:
        context = webrtc_streamer(
            key=f"smartvision-live-{settings.fingerprint[:12]}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: _LiveVideoProcessor(detector, av),
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            async_processing=True,
        )
    except Exception as error:
        st.warning(
            "The browser camera component could not start. Refresh the page, "
            "check camera permission, and confirm that the app is running on "
            f"localhost or HTTPS.\n\nTechnical detail: {error}"
        )
        return

    metrics_placeholder = st.empty()
    counts_placeholder = st.empty()
    error_placeholder = st.empty()

    if not context.state.playing:
        st.info(
            "If Start does not open a camera prompt, check browser permission and "
            "make sure another program is not using the camera.",
        )
        return

    while context.state.playing:
        processor = context.video_processor
        if processor is None:
            time.sleep(0.1)
            continue
        snapshot = processor.snapshot()
        with metrics_placeholder.container():
            columns = st.columns(4)
            columns[0].metric("Objects now", snapshot["total_detections"])
            columns[1].metric("Current FPS", f'{snapshot["current_fps"]:.1f}')
            columns[2].metric("Average FPS", f'{snapshot["average_fps"]:.1f}')
            columns[3].metric(
                "Inference", f'{snapshot["inference_time_ms"]:.1f} ms'
            )
        with counts_placeholder.container():
            _render_class_counts(snapshot["class_counts"])
        if snapshot["error"]:
            error_placeholder.error(snapshot["error"])
        else:
            error_placeholder.empty()
        time.sleep(0.25)


def _footer() -> None:
    st.markdown(
        '<div class="sv-footer">SmartVision Lite - Built for transparent, '
        'measured computer-vision experiments</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(
        page_title=f"{APP_TITLE} - {APP_SUBTITLE}",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    page, settings, settings_error = _sidebar()

    if page == "Home":
        _render_home()
    elif page == "Image Detection":
        _render_image(settings, settings_error)
    elif page == "Video Detection":
        _render_video(settings, settings_error)
    else:
        _render_live(settings, settings_error)
    _footer()


if __name__ == "__main__":
    main()
