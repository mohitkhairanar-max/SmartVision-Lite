"""Frame-wise video processing with safe resource cleanup."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from config.settings import DEFAULT_VIDEO_FPS

from .detector import DetectionResult, ObjectDetector
from .metrics import DetectionMetrics, calculate_fps
from .utilities import validate_video_filename

ProgressCallback = Callable[[float], None]
StopCallback = Callable[[], bool]


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be opened, decoded, processed, or written."""


class CorruptVideoError(VideoProcessingError):
    """Raised when OpenCV opens a file but cannot decode any video frame."""


@dataclass(frozen=True, slots=True)
class VideoProcessingResult:
    """Summary of a completed or user-stopped video-processing run."""

    input_path: Path
    output_path: Path | None
    frame_count: int
    total_detections: int
    class_counts: dict[str, int]
    elapsed_seconds: float
    average_fps: float
    average_inference_time_ms: float
    stopped_early: bool = False
    source_fps: float = 0.0

    def __post_init__(self) -> None:
        if self.frame_count < 0 or self.total_detections < 0:
            raise ValueError("Frame and detection counts cannot be negative.")
        numeric_values = (
            self.elapsed_seconds,
            self.average_fps,
            self.average_inference_time_ms,
            self.source_fps,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in numeric_values):
            raise ValueError("Video timing metrics must be finite and non-negative.")
        if any(int(count) < 0 for count in self.class_counts.values()):
            raise ValueError("Per-class detection counts cannot be negative.")

        object.__setattr__(self, "input_path", Path(self.input_path))
        if self.output_path is not None:
            object.__setattr__(self, "output_path", Path(self.output_path))
        object.__setattr__(
            self,
            "class_counts",
            dict(sorted((str(name), int(count)) for name, count in self.class_counts.items())),
        )

    @property
    def output_format(self) -> str | None:
        """Lowercase output extension without the leading dot."""

        return self.output_path.suffix.lower().lstrip(".") if self.output_path else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path) if self.output_path else None,
            "output_format": self.output_format,
            "frame_count": self.frame_count,
            "total_detections": self.total_detections,
            "class_counts": dict(self.class_counts),
            "elapsed_seconds": self.elapsed_seconds,
            "average_fps": self.average_fps,
            "average_inference_time_ms": self.average_inference_time_ms,
            "source_fps": self.source_fps,
            "stopped_early": self.stopped_early,
        }


def _load_cv2() -> Any:
    try:
        import cv2
    except (ImportError, OSError) as exc:
        raise VideoProcessingError(
            "OpenCV could not be imported. Install the project dependencies in the "
            "active virtual environment."
        ) from exc
    return cv2


def _capture_number(capture: Any, property_id: int) -> float:
    try:
        value = float(capture.get(property_id))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def _codec_for(extension: str) -> str:
    return "XVID" if extension == ".avi" else "mp4v"


class VideoProcessor:
    """Apply a detector to every decodable frame in a local video."""

    def __init__(self, detector: ObjectDetector) -> None:
        if detector is None or not callable(getattr(detector, "detect", None)):
            raise TypeError("detector must provide a detect(source) method.")
        self.detector = detector

    def process_frame(self, frame: Any) -> DetectionResult:
        """Process a single BGR frame, as used by video and webcam modes."""

        if frame is None:
            raise VideoProcessingError("The camera or video returned an empty frame.")
        size = getattr(frame, "size", None)
        if size is not None and int(size) == 0:
            raise VideoProcessingError("The camera or video returned an empty frame.")
        return self.detector.detect(frame)

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
        stop_requested: StopCallback | None = None,
    ) -> VideoProcessingResult:
        """Process a video and optionally write its annotated frames.

        ``progress_callback`` receives a fraction from ``0.0`` to ``1.0``. A
        final ``1.0`` is emitted only after normal completion. Counts are
        frame-wise detection events, not unique tracked objects.
        """

        source = Path(input_path).expanduser()
        validate_video_filename(source.name)
        if not source.is_file():
            raise VideoProcessingError(f"Video file was not found: {source}")

        destination: Path | None = None
        output_extension = ""
        if output_path is not None:
            destination = Path(output_path).expanduser()
            output_extension = validate_video_filename(destination.name)
            if source.resolve() == destination.resolve():
                raise VideoProcessingError("Input and output video paths must be different.")
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise VideoProcessingError(
                    f"Could not create output directory '{destination.parent}': {exc}"
                ) from exc

        cv2 = _load_cv2()
        capture = cv2.VideoCapture(str(source))
        writer: Any | None = None
        try:
            if not capture or not capture.isOpened():
                raise VideoProcessingError(
                    "OpenCV could not open the video. It may be corrupted or use an "
                    "unsupported codec."
                )

            total_frames = int(
                _capture_number(capture, getattr(cv2, "CAP_PROP_FRAME_COUNT", 7))
            )
            source_fps = _capture_number(capture, getattr(cv2, "CAP_PROP_FPS", 5))
            writer_fps = source_fps or DEFAULT_VIDEO_FPS
            metrics = DetectionMetrics()
            stopped_early = False

            if progress_callback is not None:
                progress_callback(0.0)
            started = perf_counter()

            while True:
                if stop_requested is not None and bool(stop_requested()):
                    stopped_early = True
                    break

                ok, frame = capture.read()
                if not ok:
                    break
                result = self.process_frame(frame)
                annotated = result.annotated_image
                if annotated is None:
                    copy = getattr(frame, "copy", None)
                    annotated = copy() if callable(copy) else frame

                if destination is not None and writer is None:
                    shape = getattr(annotated, "shape", None)
                    if shape is None or len(shape) < 2:
                        raise VideoProcessingError(
                            "Annotated frames do not expose a valid image shape."
                        )
                    height, width = int(shape[0]), int(shape[1])
                    if height <= 0 or width <= 0:
                        raise VideoProcessingError("Annotated frame dimensions are invalid.")
                    fourcc = cv2.VideoWriter_fourcc(*_codec_for(output_extension))
                    writer = cv2.VideoWriter(
                        str(destination), fourcc, writer_fps, (width, height)
                    )
                    if not writer or not writer.isOpened():
                        raise VideoProcessingError(
                            "OpenCV could not create the processed video. Try an MP4 output "
                            "path or install a compatible video codec."
                        )

                if writer is not None:
                    writer.write(annotated)
                metrics.update(result)

                if progress_callback is not None and total_frames > 0:
                    progress = min(1.0, metrics.frame_count / total_frames)
                    if progress < 1.0:
                        progress_callback(progress)

            elapsed = max(0.0, perf_counter() - started)
            if metrics.frame_count == 0 and not stopped_early:
                raise CorruptVideoError(
                    "No frames could be decoded. The video is empty, corrupted, or uses "
                    "an unsupported codec."
                )
            if progress_callback is not None and not stopped_early:
                progress_callback(1.0)

            return VideoProcessingResult(
                input_path=source,
                output_path=destination,
                frame_count=metrics.frame_count,
                total_detections=metrics.total_detections,
                class_counts=dict(metrics.class_counts),
                elapsed_seconds=elapsed,
                average_fps=calculate_fps(metrics.frame_count, elapsed),
                average_inference_time_ms=metrics.average_inference_time_ms,
                stopped_early=stopped_early,
                source_fps=source_fps,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except VideoProcessingError:
            raise
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            raise VideoProcessingError(f"Video processing failed: {message}") from exc
        finally:
            try:
                capture.release()
            except (AttributeError, RuntimeError):
                pass
            if writer is not None:
                try:
                    writer.release()
                except (AttributeError, RuntimeError):
                    pass

    process = process_video


def process_video(
    detector: ObjectDetector,
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
    stop_requested: StopCallback | None = None,
) -> VideoProcessingResult:
    """Functional convenience wrapper around :class:`VideoProcessor`."""

    return VideoProcessor(detector).process_video(
        input_path,
        output_path,
        progress_callback=progress_callback,
        stop_requested=stop_requested,
    )


__all__ = [
    "CorruptVideoError",
    "ProgressCallback",
    "StopCallback",
    "VideoProcessingError",
    "VideoProcessingResult",
    "VideoProcessor",
    "process_video",
]
