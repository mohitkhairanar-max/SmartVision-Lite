"""Run SmartVision Lite object detection with a desktop webcam.

This module is intentionally safe to import: OpenCV, the webcam, and the YOLO
model are only initialized after :func:`main` is called.  Run it from the
project root and press ``q`` in the video window to stop::

    python webcam_demo.py
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from typing import Any

from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    DEFAULT_MODEL,
)


WINDOW_TITLE = "SmartVision Lite - press q to quit"


class WebcamDemoError(RuntimeError):
    """An expected, user-facing webcam demo error."""


def _unit_interval(value: str) -> float:
    """Parse an argparse float constrained to the inclusive range 0..1."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive_integer(value: str) -> int:
    """Parse a positive integer command-line value."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _camera_index(value: str) -> int:
    """Parse a non-negative OpenCV camera index."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface without initializing any hardware."""
    parser = argparse.ArgumentParser(
        description=(
            "Run real-time SmartVision Lite object detection on an OpenCV "
            "webcam. Press q in the video window to stop."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Ultralytics model name or path to custom weights "
            f"(default: {DEFAULT_MODEL})"
        ),
    )
    parser.add_argument(
        "--camera",
        type=_camera_index,
        default=0,
        help="OpenCV camera index (default: 0)",
    )
    parser.add_argument(
        "--confidence",
        type=_unit_interval,
        default=DEFAULT_CONFIDENCE,
        help=(
            "minimum confidence threshold from 0 to 1 "
            f"(default: {DEFAULT_CONFIDENCE})"
        ),
    )
    parser.add_argument(
        "--iou",
        type=_unit_interval,
        default=DEFAULT_IOU,
        help=f"IoU threshold from 0 to 1 (default: {DEFAULT_IOU})",
    )
    parser.add_argument(
        "--image-size",
        type=_positive_integer,
        default=DEFAULT_IMAGE_SIZE,
        help=(
            "YOLO inference image size in pixels "
            f"(default: {DEFAULT_IMAGE_SIZE})"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=DEFAULT_DEVICE,
        help=f"inference device (default: {DEFAULT_DEVICE})",
    )
    return parser


def _load_cv2() -> Any:
    """Import OpenCV lazily and provide an actionable dependency error."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise WebcamDemoError(
            "OpenCV is not installed. Install the project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc
    return cv2


def _load_detector_class() -> type[Any]:
    """Import the project detector lazily so importing this module stays light."""
    try:
        from detection import ObjectDetector
    except (ImportError, ModuleNotFoundError) as exc:
        raise WebcamDemoError(
            "The object-detection dependencies could not be imported. Install "
            "the project dependencies with "
            "'python -m pip install -r requirements.txt'. "
            f"Details: {exc}"
        ) from exc
    return ObjectDetector


def _detection_error(exc: Exception, model_path: str, device: str) -> WebcamDemoError:
    """Translate common model and CUDA failures into concise CLI guidance."""
    details = str(exc).strip() or exc.__class__.__name__
    lowered = details.lower()

    if "cuda" in lowered or "cudnn" in lowered:
        return WebcamDemoError(
            "CUDA could not be used. Confirm that CUDA-enabled PyTorch and a "
            "compatible NVIDIA driver are installed, or retry with "
            f"'--device cpu'. Details: {details}"
        )
    if isinstance(exc, FileNotFoundError) or any(
        phrase in lowered
        for phrase in ("no such file", "not found", "does not exist", "invalid model")
    ):
        return WebcamDemoError(
            f"The model '{model_path}' could not be found or opened. Check the "
            f"model name/path and try again. Details: {details}"
        )
    if any(phrase in lowered for phrase in ("download", "network", "connection")):
        return WebcamDemoError(
            f"The model '{model_path}' could not be downloaded. Check the "
            f"internet connection or provide a local .pt file. Details: {details}"
        )
    return WebcamDemoError(
        f"Object detection failed while loading or running '{model_path}' on "
        f"'{device}'. Details: {details}"
    )


def _draw_status(
    cv2: Any,
    image: Any,
    *,
    fps: float,
    object_count: int,
    inference_time_ms: float,
) -> None:
    """Draw readable live performance information over an annotated frame."""
    lines = (
        f"FPS: {fps:.1f}",
        f"Objects: {object_count}",
        f"Inference: {inference_time_ms:.1f} ms",
        "Press q to quit",
    )
    for index, line in enumerate(lines):
        origin = (12, 30 + index * 27)
        # The dark outline keeps the text legible on both light and dark frames.
        cv2.putText(
            image,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def run_webcam(
    *,
    model_path: str = DEFAULT_MODEL,
    camera_index: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: str = DEFAULT_DEVICE,
) -> None:
    """Open a webcam and display real-time annotated object detections.

    The project detector supplies the bounding boxes, class labels, confidence
    values, and inference time.  This function adds a smoothed processing FPS
    display and guarantees that camera/window resources are cleaned up.
    """
    cv2 = _load_cv2()
    capture: Any | None = None

    try:
        try:
            capture = cv2.VideoCapture(camera_index)
            camera_opened = capture is not None and bool(capture.isOpened())
        except Exception as exc:
            raise WebcamDemoError(
                f"OpenCV could not access camera index {camera_index}. "
                f"Details: {exc}"
            ) from exc

        if not camera_opened:
            raise WebcamDemoError(
                f"Could not open camera index {camera_index}. Check that a "
                "webcam is connected, camera permission is enabled, and no "
                "other application is using it."
            )

        ObjectDetector = _load_detector_class()
        try:
            detector = ObjectDetector(
                model_path=model_path,
                confidence=confidence,
                iou=iou,
                image_size=image_size,
                device=device,
            )
        except Exception as exc:
            raise _detection_error(exc, model_path, device) from exc

        print(
            f"Camera {camera_index} opened. Running '{model_path}' on "
            f"'{device}'. Press q in the video window to stop."
        )

        smoothed_fps = 0.0
        failed_reads = 0
        while True:
            frame_started = time.perf_counter()
            try:
                ok, frame = capture.read()
            except Exception as exc:
                raise WebcamDemoError(
                    f"OpenCV lost access to camera index {camera_index}. "
                    f"Details: {exc}"
                ) from exc
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= 3:
                    raise WebcamDemoError(
                        f"Camera {camera_index} opened but did not return valid "
                        "frames. Reconnect it or try another --camera index."
                    )
                continue
            failed_reads = 0

            try:
                result = detector.detect(frame)
            except Exception as exc:
                raise _detection_error(exc, model_path, device) from exc

            elapsed = max(time.perf_counter() - frame_started, 1e-9)
            current_fps = 1.0 / elapsed
            smoothed_fps = (
                current_fps
                if smoothed_fps == 0.0
                else 0.85 * smoothed_fps + 0.15 * current_fps
            )

            try:
                annotated = result.annotated_image.copy()
                _draw_status(
                    cv2,
                    annotated,
                    fps=smoothed_fps,
                    object_count=int(result.total_detections),
                    inference_time_ms=float(result.inference_time_ms),
                )
                cv2.imshow(WINDOW_TITLE, annotated)
                key = cv2.waitKey(1) & 0xFF
            except Exception as exc:
                raise WebcamDemoError(
                    "OpenCV could not display the webcam window. Run this "
                    f"script in a desktop session with GUI support. Details: {exc}"
                ) from exc

            if key in (ord("q"), ord("Q")):
                break
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            # Cleanup should not hide the useful camera/model error above.
            pass


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        run_webcam(
            model_path=args.model,
            camera_index=args.camera,
            confidence=args.confidence,
            iou=args.iou,
            image_size=args.image_size,
            device=args.device,
        )
    except WebcamDemoError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nWebcam demo stopped.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
