"""Tests for counters and truthful FPS calculations."""

from __future__ import annotations

import pytest

from detection.detector import Detection, DetectionResult
from detection.metrics import (
    DetectionMetrics,
    FPSMeter,
    calculate_fps,
    count_by_class,
    mean_confidence,
    summarize_detections,
)


def sample_detections() -> tuple[Detection, ...]:
    return (
        Detection(0, "person", 0.8, (0, 0, 10, 20)),
        Detection(0, "person", 0.6, (20, 20, 30, 40)),
        Detection(2, "car", 0.7, (50, 50, 90, 80)),
    )


def test_detection_summary() -> None:
    detections = sample_detections()

    assert count_by_class(detections) == {"car": 1, "person": 2}
    assert mean_confidence(detections) == pytest.approx(0.7)
    assert summarize_detections(detections) == {
        "total_detections": 3,
        "class_counts": {"car": 1, "person": 2},
        "mean_confidence": pytest.approx(0.7),
    }
    assert mean_confidence(()) == 0.0


def test_mapping_records_are_supported() -> None:
    records = [
        {"class_name": "cat", "confidence": 0.5},
        {"class_name": "cat", "confidence": 1.0},
    ]

    assert count_by_class(records) == {"cat": 2}
    assert mean_confidence(records) == pytest.approx(0.75)


def test_calculate_fps_handles_zero_duration_without_inventing_a_rate() -> None:
    assert calculate_fps(120, 4.0) == pytest.approx(30.0)
    assert calculate_fps(0, 4.0) == 0.0
    assert calculate_fps(10, 0.0) == 0.0
    with pytest.raises(ValueError):
        calculate_fps(-1, 1.0)


def test_fps_meter_uses_real_intervals_and_a_rolling_window() -> None:
    meter = FPSMeter(window_size=3)

    assert meter.tick(10.0) == 0.0
    assert meter.tick(10.5) == pytest.approx(2.0)
    assert meter.tick(11.0) == pytest.approx(2.0)
    assert meter.average_fps == pytest.approx(2.0)
    meter.tick(11.25)
    meter.tick(11.5)
    assert meter.frame_count == 4  # window_size intervals plus one timestamp
    assert meter.average_fps == pytest.approx(3.0)

    meter.reset()
    assert meter.frame_count == 0
    assert meter.average_fps == 0.0


def test_fps_meter_rejects_backwards_time() -> None:
    meter = FPSMeter()
    meter.tick(5.0)
    with pytest.raises(ValueError, match="backwards"):
        meter.tick(4.9)


def test_detection_metrics_accumulate_frame_events() -> None:
    accumulator = DetectionMetrics()
    accumulator.update(
        DetectionResult(sample_detections(), inference_time_ms=10.0)
    )
    accumulator.update(DetectionResult((), inference_time_ms=20.0))

    assert accumulator.frame_count == 2
    assert accumulator.total_detections == 3
    assert dict(accumulator.class_counts) == {"person": 2, "car": 1}
    assert accumulator.average_inference_time_ms == pytest.approx(15.0)
    assert accumulator.average_confidence == pytest.approx(0.7)
    assert accumulator.as_dict()["class_counts"] == {"car": 1, "person": 2}

    accumulator.reset()
    assert accumulator.as_dict()["total_detections"] == 0
