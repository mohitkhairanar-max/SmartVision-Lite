"""Deterministic detection counters and FPS measurements."""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable


def _attribute(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        if name not in item:
            raise ValueError(f"Detection record is missing '{name}'.")
        return item[name]
    if not hasattr(item, name):
        raise ValueError(f"Detection object is missing '{name}'.")
    return getattr(item, name)


def count_by_class(detections: Iterable[Any]) -> dict[str, int]:
    """Return alphabetically sorted counts for detection objects or mappings."""

    counts: Counter[str] = Counter()
    for detection in detections:
        class_name = str(_attribute(detection, "class_name")).strip()
        if not class_name:
            raise ValueError("Detection class names cannot be empty.")
        counts[class_name] += 1
    return dict(sorted(counts.items()))


def mean_confidence(detections: Iterable[Any]) -> float:
    """Return the arithmetic mean confidence, or ``0.0`` for no detections."""

    values: list[float] = []
    for detection in detections:
        confidence = float(_attribute(detection, "confidence"))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1.")
        values.append(confidence)
    return sum(values) / len(values) if values else 0.0


def calculate_fps(frame_count: int, elapsed_seconds: float) -> float:
    """Calculate end-to-end FPS, returning zero when no duration is measurable."""

    if isinstance(frame_count, bool) or int(frame_count) < 0:
        raise ValueError("frame_count must be a non-negative integer.")
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_seconds must be finite and non-negative.")
    return int(frame_count) / elapsed if elapsed > 0 else 0.0


def summarize_detections(detections: Iterable[Any]) -> dict[str, Any]:
    """Return total, per-class counts, and mean confidence for one result."""

    items = tuple(detections)
    return {
        "total_detections": len(items),
        "class_counts": count_by_class(items),
        "mean_confidence": mean_confidence(items),
    }


class FPSMeter:
    """Measure current and rolling-average FPS from frame completion times.

    Pass a deterministic ``clock`` in tests. The first tick has no preceding
    interval and therefore reports a current FPS of zero.
    """

    def __init__(
        self,
        window_size: int = 30,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if isinstance(window_size, bool) or int(window_size) < 2:
            raise ValueError("window_size must be at least 2.")
        self.window_size = int(window_size)
        self._clock = clock
        self._timestamps: deque[float] = deque(maxlen=self.window_size + 1)
        self._current_fps = 0.0

    def tick(self, timestamp: float | None = None) -> float:
        """Record a completed frame and return the current instantaneous FPS."""

        now = float(self._clock() if timestamp is None else timestamp)
        if not math.isfinite(now):
            raise ValueError("FPS timestamps must be finite.")
        if self._timestamps and now < self._timestamps[-1]:
            raise ValueError("FPS timestamps cannot move backwards.")
        if self._timestamps:
            interval = now - self._timestamps[-1]
            self._current_fps = 1.0 / interval if interval > 0 else 0.0
        else:
            self._current_fps = 0.0
        self._timestamps.append(now)
        return self._current_fps

    @property
    def current_fps(self) -> float:
        return self._current_fps

    @property
    def average_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def frame_count(self) -> int:
        return len(self._timestamps)

    def reset(self) -> None:
        self._timestamps.clear()
        self._current_fps = 0.0


@dataclass(slots=True)
class DetectionMetrics:
    """Accumulate frame-wise detection events and inference timings.

    Video totals intentionally count every detection in every frame. They are
    not unique-object counts because object tracking is outside this project.
    """

    total_detections: int = 0
    frame_count: int = 0
    total_inference_time_ms: float = 0.0
    confidence_sum: float = 0.0
    class_counts: Counter[str] = field(default_factory=Counter)

    def update(
        self,
        detections: Iterable[Any] | Any,
        *,
        inference_time_ms: float | None = None,
    ) -> None:
        """Add one image/frame result or an iterable of detection records."""

        if hasattr(detections, "detections"):
            result = detections
            items = tuple(result.detections)
            if inference_time_ms is None:
                inference_time_ms = getattr(result, "inference_time_ms", 0.0)
        else:
            items = tuple(detections)

        inference = float(0.0 if inference_time_ms is None else inference_time_ms)
        if not math.isfinite(inference) or inference < 0:
            raise ValueError("inference_time_ms must be finite and non-negative.")

        per_class = count_by_class(items)
        confidence_total = 0.0
        for item in items:
            value = float(_attribute(item, "confidence"))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Detection confidence must be between 0 and 1.")
            confidence_total += value

        self.frame_count += 1
        self.total_detections += len(items)
        self.total_inference_time_ms += inference
        self.confidence_sum += confidence_total
        self.class_counts.update(per_class)

    @property
    def average_inference_time_ms(self) -> float:
        return (
            self.total_inference_time_ms / self.frame_count
            if self.frame_count
            else 0.0
        )

    @property
    def average_confidence(self) -> float:
        return self.confidence_sum / self.total_detections if self.total_detections else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frame_count,
            "total_detections": self.total_detections,
            "class_counts": dict(sorted(self.class_counts.items())),
            "average_confidence": self.average_confidence,
            "average_inference_time_ms": self.average_inference_time_ms,
        }

    def reset(self) -> None:
        self.total_detections = 0
        self.frame_count = 0
        self.total_inference_time_ms = 0.0
        self.confidence_sum = 0.0
        self.class_counts.clear()


# Descriptive aliases for beginner-facing UI code.
average_confidence = mean_confidence
get_class_counts = count_by_class


__all__ = [
    "DetectionMetrics",
    "FPSMeter",
    "average_confidence",
    "calculate_fps",
    "count_by_class",
    "get_class_counts",
    "mean_confidence",
    "summarize_detections",
]
