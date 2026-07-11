"""Tests for input validation, safe filenames, devices, and exports."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from detection.detector import Detection
from detection.utilities import (
    DeviceUnavailableError,
    UnsupportedFileError,
    ValidationError,
    build_output_path,
    is_supported_image,
    is_supported_video,
    records_to_csv,
    records_to_json,
    resolve_device,
    safe_filename,
    validate_image_filename,
    validate_image_size,
    validate_threshold,
    validate_video_filename,
)


@pytest.mark.parametrize("filename", ["photo.jpg", "PHOTO.JPEG", "image.PnG"])
def test_supported_image_extensions(filename: str) -> None:
    assert validate_image_filename(filename).startswith(".")
    assert is_supported_image(filename)


@pytest.mark.parametrize("filename", ["clip.mp4", "CLIP.AVI", "movie.Mov"])
def test_supported_video_extensions(filename: str) -> None:
    assert validate_video_filename(filename).startswith(".")
    assert is_supported_video(filename)


def test_unsupported_or_missing_extensions_are_clear() -> None:
    with pytest.raises(UnsupportedFileError, match="Supported formats"):
        validate_image_filename("payload.exe")
    with pytest.raises(UnsupportedFileError, match="no extension"):
        validate_video_filename("video")
    assert not is_supported_image("video.mp4")
    assert not is_supported_video("image.png")


def test_safe_filename_removes_paths_and_windows_reserved_names(tmp_path: Path) -> None:
    assert safe_filename("../../my unsafe photo?.jpg") == "my_unsafe_photo_.jpg"
    assert safe_filename(r"C:\uploads\CON.txt") == "_CON.txt"
    assert build_output_path("cat photo.png", tmp_path) == tmp_path / "detected_cat_photo.png"


def test_threshold_and_image_size_validation() -> None:
    assert validate_threshold("0.25", name="confidence") == pytest.approx(0.25)
    assert validate_image_size(640) == 640
    with pytest.raises(ValidationError, match="Confidence"):
        validate_threshold(float("nan"), name="confidence")
    with pytest.raises(ValidationError, match="one of"):
        validate_image_size(123)
    with pytest.raises(ValidationError):
        validate_image_size(True)


class FakeCuda:
    def __init__(self, available: bool, count: int = 0) -> None:
        self.available = available
        self.count = count

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count


def fake_torch(available: bool, count: int = 0) -> SimpleNamespace:
    return SimpleNamespace(cuda=FakeCuda(available, count))


def test_device_resolution_is_explicit_and_testable() -> None:
    assert resolve_device("cpu", torch_module=None) == "cpu"
    assert resolve_device("auto", torch_module=fake_torch(False)) == "cpu"
    assert resolve_device("auto", torch_module=fake_torch(True, 1)) == "cuda"
    assert resolve_device("cuda:0", torch_module=fake_torch(True, 1)) == "cuda:0"
    with pytest.raises(DeviceUnavailableError, match="not available"):
        resolve_device("cuda", torch_module=fake_torch(False))
    with pytest.raises(DeviceUnavailableError, match="only 1"):
        resolve_device("cuda:2", torch_module=fake_torch(True, 1))


def test_csv_and_json_exports_are_deterministic() -> None:
    detections = [
        Detection(0, "person", 0.95, (1, 2, 3, 4)),
        Detection(2, "car", 0.75, (10, 20, 30, 40)),
    ]

    json_rows = json.loads(records_to_json(detections))
    assert json_rows[0]["class_name"] == "person"
    assert json_rows[1]["confidence"] == pytest.approx(0.75)

    csv_rows = list(csv.DictReader(StringIO(records_to_csv(detections))))
    assert [row["class_name"] for row in csv_rows] == ["person", "car"]
    assert csv_rows[0]["x1"] == "1.0"


def test_empty_exports_still_have_a_csv_header() -> None:
    assert records_to_json([]) == "[]"
    header = records_to_csv([]).splitlines()[0]
    assert header == "class_id,class_name,confidence,x1,y1,x2,y2"
