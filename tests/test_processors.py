"""Focused tests for image decoding and video resource handling."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from detection.detector import Detection, DetectionResult
from detection.image_processor import ImageProcessingError, ImageProcessor, encode_image, load_image
from detection.video_processor import CorruptVideoError, VideoProcessor

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


class RecordingDetector:
    def __init__(self) -> None:
        self.sources: list[Any] = []

    def detect(self, source: Any) -> DetectionResult:
        self.sources.append(source)
        annotated = source.copy() if callable(getattr(source, "copy", None)) else source
        return DetectionResult(
            (Detection(0, "object", 0.5, (0, 0, 1, 1)),),
            annotated_image=annotated,
            inference_time_ms=5.0,
            source_shape=tuple(source.shape[:2]),
        )


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


def test_image_processor_decodes_bytes_and_preserves_upload_position() -> None:
    upload = BytesIO(png_bytes())
    upload.name = "sample.png"  # type: ignore[attr-defined]
    detector = RecordingDetector()

    result = ImageProcessor(detector).process(upload)

    assert upload.tell() == 0
    assert detector.sources[0].shape == (2, 3, 3)
    assert result.total_detections == 1


def test_invalid_image_bytes_are_rejected() -> None:
    with pytest.raises(ImageProcessingError, match="invalid, corrupted"):
        load_image(b"not an image", filename="bad.png")


def test_annotated_bgr_image_can_be_encoded_for_download() -> None:
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :] = (30, 20, 10)

    encoded = encode_image(bgr)
    decoded = load_image(encoded)

    assert tuple(decoded[0, 0]) == (10, 20, 30)


class FakeCapture:
    def __init__(self, frames: list[Any], fps: float = 25.0) -> None:
        self.frames = list(frames)
        self.original_count = len(frames)
        self.fps = fps
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return float(self.original_count if property_id == 7 else self.fps)

    def read(self) -> tuple[bool, Any | None]:
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self) -> None:
        self.frames: list[Any] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: Any) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


def fake_cv2(capture: FakeCapture, writer: FakeWriter) -> SimpleNamespace:
    return SimpleNamespace(
        CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_FPS=5,
        VideoCapture=lambda _: capture,
        VideoWriter=lambda *_: writer,
        VideoWriter_fourcc=lambda *_: 1234,
    )


def test_video_processor_writes_frames_reports_progress_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(2)]
    capture = FakeCapture(frames)
    writer = FakeWriter()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2(capture, writer))
    source = tmp_path / "input.mp4"
    source.write_bytes(b"placeholder")
    progress: list[float] = []

    result = VideoProcessor(RecordingDetector()).process_video(
        source,
        tmp_path / "output.mp4",
        progress_callback=progress.append,
    )

    assert result.frame_count == 2
    assert result.total_detections == 2
    assert result.class_counts == {"object": 2}
    assert result.average_inference_time_ms == pytest.approx(5.0)
    assert result.output_format == "mp4"
    assert progress == [0.0, 0.5, 1.0]
    assert len(writer.frames) == 2
    assert capture.released and writer.released


def test_video_with_no_decodable_frames_is_rejected_and_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = FakeCapture([])
    writer = FakeWriter()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2(capture, writer))
    source = tmp_path / "empty.mp4"
    source.write_bytes(b"placeholder")

    with pytest.raises(CorruptVideoError, match="No frames"):
        VideoProcessor(RecordingDetector()).process_video(source)

    assert capture.released
