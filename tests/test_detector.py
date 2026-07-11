"""Offline tests for the lazy YOLO wrapper."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import detection.detector as detector_module
from detection.detector import (
    Detection,
    InferenceError,
    ModelLoadError,
    ObjectDetector,
)


class FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = values

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> Any:
        return self.values


class FakeBoxes:
    def __init__(
        self,
        coordinates: list[list[float]],
        confidences: list[float],
        classes: list[float],
    ) -> None:
        self.xyxy = FakeTensor(coordinates)
        self.conf = FakeTensor(confidences)
        self.cls = FakeTensor(classes)


class FakeResult:
    def __init__(self, boxes: FakeBoxes) -> None:
        self.boxes = boxes
        self.names = {0: "person", 2: "car"}
        self.orig_shape = (480, 640)
        self.speed = {"inference": 12.5}

    def plot(self) -> str:
        return "annotated pixels"


class FakeModel:
    names = {0: "person", 2: "car"}

    def __init__(self, result: FakeResult | None = None) -> None:
        self.result = result or FakeResult(
            FakeBoxes(
                [[10.0, 20.0, 50.0, 80.0], [100.0, 120.0, 180.0, 200.0]],
                [0.9, 0.75],
                [0.0, 2.0],
            )
        )
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[FakeResult]:
        self.calls.append(kwargs)
        return [self.result]


def test_constructor_is_lazy_and_detect_normalizes_result() -> None:
    model = FakeModel()
    load_calls: list[str] = []

    def loader(reference: str) -> FakeModel:
        load_calls.append(reference)
        return model

    detector = ObjectDetector(
        "fake.pt",
        confidence=0.3,
        iou=0.5,
        image_size=480,
        device="cpu",
        model_loader=loader,
    )

    assert not detector.is_loaded
    assert load_calls == []

    source = object()
    result = detector.detect(source)

    assert detector.is_loaded
    assert load_calls == ["fake.pt"]
    assert result.total_detections == 2
    assert result.class_counts == {"car": 1, "person": 1}
    assert result.inference_time_ms == pytest.approx(12.5)
    assert result.source_shape == (480, 640)
    assert result.annotated_image == "annotated pixels"
    assert result.rows[0] == {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "x1": 10.0,
        "y1": 20.0,
        "x2": 50.0,
        "y2": 80.0,
    }
    assert model.calls[0] == {
        "source": source,
        "conf": 0.3,
        "iou": 0.5,
        "imgsz": 480,
        "device": "cpu",
        "verbose": False,
    }

    detector.predict(source)
    assert load_calls == ["fake.pt"], "The loaded model should be cached."


def test_empty_boxes_are_a_valid_no_detection_result() -> None:
    model = FakeModel(FakeResult(FakeBoxes([], [], [])))
    result = ObjectDetector(model=model, device="cpu").detect(object())

    assert result.detections == ()
    assert result.total_detections == 0
    assert result.class_counts == {}
    assert result.to_json() == "[]"
    assert result.to_csv().startswith("class_id,class_name,confidence,x1,y1,x2,y2")


def test_detection_validates_values_and_exports_flat_record() -> None:
    detection = Detection(3, "dog", 0.875, (1, 2, 11, 22))

    assert detection.bbox == (1.0, 2.0, 11.0, 22.0)
    assert detection.to_dict()["class_name"] == "dog"
    with pytest.raises(ValueError, match="confidence"):
        Detection(3, "dog", 1.1, (1, 2, 11, 22))
    with pytest.raises(ValueError, match="x2"):
        Detection(3, "dog", 0.5, (11, 2, 1, 22))


def test_model_loader_failure_has_actionable_context() -> None:
    def failing_loader(_: str) -> Any:
        raise RuntimeError("bad checkpoint")

    detector = ObjectDetector("broken.pt", device="cpu", model_loader=failing_loader)
    with pytest.raises(ModelLoadError, match="broken.pt.*bad checkpoint"):
        detector.load_model()


def test_missing_explicit_custom_path_fails_before_ultralytics_import(
    tmp_path: Path,
) -> None:
    detector = ObjectDetector(tmp_path / "missing.pt", device="cpu")

    with pytest.raises(ModelLoadError, match="not found"):
        detector.load_model()


def test_default_loader_contains_ultralytics_settings_in_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[str] = []

    def fake_yolo(reference: str) -> object:
        loaded.append(reference)
        return object()

    monkeypatch.setattr(detector_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    detector = ObjectDetector("yolo11n.pt", device="cpu")

    assert "YOLO_CONFIG_DIR" not in os.environ
    detector.load_model()

    expected = tmp_path / ".ultralytics"
    assert expected.is_dir()
    assert os.environ["YOLO_CONFIG_DIR"] == str(expected)
    assert loaded == ["yolo11n.pt"]


def test_inference_exception_is_wrapped() -> None:
    class BrokenModel:
        def predict(self, **_: Any) -> Any:
            raise RuntimeError("decoder failed")

    detector = ObjectDetector(model=BrokenModel(), device="cpu")
    with pytest.raises(InferenceError, match="decoder failed"):
        detector.detect(object())


def test_misaligned_box_arrays_are_rejected() -> None:
    result = FakeResult(FakeBoxes([[1, 2, 3, 4]], [], [0]))
    detector = ObjectDetector(model=FakeModel(result), device="cpu")

    with pytest.raises(InferenceError, match="do not align"):
        detector.detect(object())


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("confidence", -0.1, "Confidence"),
        ("iou", 1.1, "IoU"),
        ("image_size", 0, "Image size"),
        ("device", "tpu", "Device"),
    ],
)
def test_invalid_settings_are_rejected(argument: str, value: Any, message: str) -> None:
    kwargs = {argument: value}
    with pytest.raises(ValueError, match=message):
        ObjectDetector(**kwargs)
