"""Run one real, headless SmartVision Lite inference.

The first run may download the configured Ultralytics checkpoint. Model weights
are intentionally ignored by Git. This script does not open a window or camera.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

# Make direct execution from ``scripts/`` resolve the project packages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
YOLO_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
YOLO_CONFIG_DIR.mkdir(exist_ok=True)
MATPLOTLIB_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

from detection import ObjectDetector


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the smoke test."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser


def main() -> int:
    """Load a real model and verify one inference result."""

    args = build_parser().parse_args()
    synthetic_image = np.zeros((320, 320, 3), dtype=np.uint8)
    try:
        detector = ObjectDetector(
            model_path=args.model,
            confidence=0.25,
            iou=0.45,
            image_size=320,
            device=args.device,
        )
        result = detector.detect(synthetic_image)
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1

    if result.annotated_image is None:
        print("Smoke test failed: the detector returned no annotated image.", file=sys.stderr)
        return 1
    if result.source_shape != (320, 320):
        print(
            f"Smoke test failed: unexpected source shape {result.source_shape}.",
            file=sys.stderr,
        )
        return 1

    print(
        "Smoke test passed: "
        f"model={args.model}, device={args.device}, "
        f"detections={result.total_detections}, "
        f"inference_ms={result.inference_time_ms:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
