"""Lazy, testable wrapper around Ultralytics YOLO inference."""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from config.settings import (
    DEFAULT_CONFIDENCE,
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    PROJECT_ROOT,
    DetectionSettings,
)

from .utilities import (
    DeviceUnavailableError,
    records_to_csv,
    records_to_json,
    resolve_device,
)


class DetectorError(RuntimeError):
    """Base class for model loading and inference failures."""


class ModelLoadError(DetectorError):
    """Raised when a requested YOLO model cannot be loaded."""


class InferenceError(DetectorError):
    """Raised when model inference fails or returns malformed data."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One object detection in pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ValueError("class_id must be a non-negative integer.")
        name = str(self.class_name).strip()
        if not name:
            raise ValueError("class_name cannot be empty.")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain x1, y1, x2, and y2.")
        coordinates = tuple(float(value) for value in self.bbox)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("bbox coordinates must be finite numbers.")
        x1, y1, x2, y2 = coordinates
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox must satisfy x2 >= x1 and y2 >= y1.")

        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(self, "class_name", name)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "bbox", coordinates)

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a flat record suitable for a table, CSV, or JSON export."""

        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Normalized output from one image or video-frame prediction."""

    detections: tuple[Detection, ...] = ()
    annotated_image: Any | None = field(default=None, repr=False, compare=False)
    inference_time_ms: float = 0.0
    source_shape: tuple[int, int] | None = None
    raw_result: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        detections = tuple(self.detections)
        if not all(isinstance(item, Detection) for item in detections):
            raise TypeError("detections must contain Detection instances.")
        inference_time = float(self.inference_time_ms)
        if not math.isfinite(inference_time) or inference_time < 0:
            raise ValueError("inference_time_ms must be a finite non-negative value.")

        shape = self.source_shape
        if shape is not None:
            if len(shape) < 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
                raise ValueError("source_shape must contain positive height and width values.")
            shape = (int(shape[0]), int(shape[1]))

        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "inference_time_ms", inference_time)
        object.__setattr__(self, "source_shape", shape)

    @property
    def total_detections(self) -> int:
        """Number of objects detected in this image or frame."""

        return len(self.detections)

    @property
    def class_counts(self) -> dict[str, int]:
        """Per-class counts sorted by label for deterministic display."""

        counts = Counter(item.class_name for item in self.detections)
        return dict(sorted(counts.items()))

    @property
    def confidence_values(self) -> tuple[float, ...]:
        return tuple(item.confidence for item in self.detections)

    @property
    def rows(self) -> list[dict[str, int | float | str]]:
        """Flat detection rows suitable for ``pandas.DataFrame``."""

        return [item.to_dict() for item in self.detections]

    def to_json(self, *, indent: int = 2) -> str:
        return records_to_json(self.detections, indent=indent)

    def to_csv(self) -> str:
        return records_to_csv(self.detections)


ModelLoader = Callable[[str], Any]


class ObjectDetector:
    """Run YOLO predictions while deferring all heavy work until first use.

    The constructor never imports Ultralytics or PyTorch and never downloads a
    model. A custom ``model_loader`` provides a clean seam for offline tests.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        image_size: int = DEFAULT_IMAGE_SIZE,
        device: str = DEFAULT_DEVICE,
        *,
        model_loader: ModelLoader | None = None,
        model: Any | None = None,
    ) -> None:
        self.settings = DetectionSettings(
            model_path=model_path,
            confidence=confidence,
            iou=iou,
            image_size=image_size,
            device=device,
        )
        self._model_reference_was_path = isinstance(model_path, Path)
        self._model_loader = model_loader
        self._model = model

    @property
    def model_path(self) -> str:
        return str(self.settings.model_path)

    @property
    def confidence(self) -> float:
        return self.settings.confidence

    @property
    def conf(self) -> float:
        """Short alias matching Ultralytics terminology."""

        return self.confidence

    @property
    def iou(self) -> float:
        return self.settings.iou

    @property
    def image_size(self) -> int:
        return self.settings.image_size

    @property
    def device(self) -> str:
        return str(self.settings.device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> Any:
        """Return the model, loading it on demand if necessary."""

        return self.load_model()

    def _is_required_local_path(self) -> bool:
        reference = self.model_path
        return (
            self._model_reference_was_path
            or "/" in reference
            or "\\" in reference
            or reference.startswith(".")
            or Path(reference).name.lower() == "best.pt"
        )

    def load_model(self) -> Any:
        """Load and cache the configured YOLO model."""

        if self._model is not None:
            return self._model

        if self._model_loader is None and self._is_required_local_path():
            path = Path(self.model_path).expanduser()
            if not path.is_file():
                raise ModelLoadError(
                    f"Custom model was not found: {path}. Select an existing .pt file."
                )
            if path.suffix.lower() != ".pt":
                raise ModelLoadError("Custom models must use the .pt file extension.")

        loader = self._model_loader
        if loader is None:
            # Ultralytics otherwise creates its settings file under the user's
            # roaming profile on Windows. Keep that runtime state inside this
            # project's ignored directory, and do so only at explicit load time.
            settings_directory = PROJECT_ROOT / ".ultralytics"
            matplotlib_directory = PROJECT_ROOT / ".matplotlib"
            try:
                settings_directory.mkdir(parents=True, exist_ok=True)
                matplotlib_directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ModelLoadError(
                    "Could not create project-local dependency settings "
                    f"directories: {exc}"
                ) from exc
            os.environ["YOLO_CONFIG_DIR"] = str(settings_directory)
            os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_directory))
            try:
                from ultralytics import YOLO
            except (ImportError, OSError) as exc:
                raise ModelLoadError(
                    "Ultralytics could not be imported. Install the project dependencies "
                    "inside the virtual environment."
                ) from exc
            loader = YOLO

        try:
            model = loader(self.model_path)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load model '{self.model_path}': {exc}"
            ) from exc
        if model is None:
            raise ModelLoadError(f"Model loader returned no model for '{self.model_path}'.")
        self._model = model
        return model

    # Beginner-friendly alias used in examples.
    load = load_model

    def unload(self) -> None:
        """Release this wrapper's model reference."""

        self._model = None

    def detect(self, source: Any, *, annotate: bool = True) -> DetectionResult:
        """Run one prediction and normalize Ultralytics' first result.

        ``source`` may be any single-image source accepted by Ultralytics,
        including a NumPy frame, PIL image, or local path.
        """

        if source is None:
            raise InferenceError("No image or video frame was provided for detection.")

        model = self.load_model()
        try:
            selected_device = resolve_device(self.device)
        except DeviceUnavailableError:
            raise

        arguments: dict[str, Any] = {
            "source": source,
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.image_size,
            "device": selected_device,
            "verbose": False,
        }
        started = perf_counter()
        try:
            predictor = getattr(model, "predict", None)
            raw_results = predictor(**arguments) if callable(predictor) else model(**arguments)
        except (KeyboardInterrupt, SystemExit):
            raise
        except DeviceUnavailableError:
            raise
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            raise InferenceError(f"Object detection failed: {message}") from exc
        elapsed_ms = (perf_counter() - started) * 1000.0

        result = _first_result(raw_results)
        if result is None:
            raise InferenceError("The model returned no prediction result.")

        try:
            detections = _extract_detections(result, model)
            annotated = _annotate(result, source) if annotate else None
            shape = _extract_source_shape(result, source)
            inference_ms = _extract_inference_time(result, elapsed_ms)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"The model returned an invalid prediction: {exc}") from exc

        return DetectionResult(
            detections=detections,
            annotated_image=annotated,
            inference_time_ms=inference_ms,
            source_shape=shape,
            raw_result=result,
        )

    def predict(self, source: Any, *, annotate: bool = True) -> DetectionResult:
        """Alias for :meth:`detect` for users familiar with Ultralytics."""

        return self.detect(source, annotate=annotate)

    __call__ = detect


def _first_result(raw_results: Any) -> Any | None:
    if raw_results is None:
        return None
    if hasattr(raw_results, "boxes"):
        return raw_results
    if isinstance(raw_results, Sequence) and not isinstance(raw_results, (str, bytes)):
        return raw_results[0] if raw_results else None
    try:
        return next(iter(raw_results))
    except (TypeError, StopIteration):
        return None


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        current = tolist()
    if isinstance(current, list):
        return current
    if isinstance(current, tuple):
        return list(current)
    try:
        return list(current)
    except TypeError:
        return [current]


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, Mapping):
        value = names.get(class_id, names.get(str(class_id)))
        if value is not None:
            return str(value)
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return f"class_{class_id}"


def _extract_detections(result: Any, model: Any) -> tuple[Detection, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()

    coordinates = _to_list(getattr(boxes, "xyxy", None))
    confidences = _to_list(getattr(boxes, "conf", None))
    classes = _to_list(getattr(boxes, "cls", None))
    if not coordinates and not confidences and not classes:
        return ()
    if not (len(coordinates) == len(confidences) == len(classes)):
        raise InferenceError("Prediction boxes, confidence scores, and classes do not align.")

    names = getattr(result, "names", None)
    if names is None:
        names = getattr(model, "names", {})

    detections: list[Detection] = []
    for coordinate, confidence, class_value in zip(
        coordinates, confidences, classes, strict=True
    ):
        values = _to_list(coordinate)
        if len(values) < 4:
            raise InferenceError("A prediction bounding box has fewer than four coordinates.")
        class_id = int(class_value)
        detections.append(
            Detection(
                class_id=class_id,
                class_name=_class_name(names, class_id),
                confidence=float(confidence),
                bbox=tuple(float(value) for value in values[:4]),
            )
        )
    return tuple(detections)


def _annotate(result: Any, source: Any) -> Any | None:
    plot = getattr(result, "plot", None)
    if callable(plot):
        try:
            return plot()
        except Exception as exc:
            raise InferenceError(f"Could not draw prediction annotations: {exc}") from exc
    copy = getattr(source, "copy", None)
    return copy() if callable(copy) else None


def _extract_source_shape(result: Any, source: Any) -> tuple[int, int] | None:
    shape = getattr(result, "orig_shape", None)
    if shape is None:
        shape = getattr(source, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    return (height, width) if height > 0 and width > 0 else None


def _extract_inference_time(result: Any, measured_ms: float) -> float:
    speed = getattr(result, "speed", None)
    if isinstance(speed, Mapping):
        reported = speed.get("inference")
        try:
            value = float(reported)
        except (TypeError, ValueError):
            value = -1.0
        if math.isfinite(value) and value >= 0:
            return value
    return max(0.0, float(measured_ms))


# Compatibility aliases kept intentionally simple for classroom examples.
Detector = ObjectDetector
DetectionRecord = Detection


__all__ = [
    "Detection",
    "DetectionRecord",
    "DetectionResult",
    "Detector",
    "DetectorError",
    "InferenceError",
    "ModelLoadError",
    "ObjectDetector",
]
