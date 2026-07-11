"""Application-wide defaults and validated detector settings.

This module deliberately contains no imports from PyTorch, Ultralytics, OpenCV,
or Streamlit.  Importing configuration should always be quick and must never
initialise hardware or download model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"

DEFAULT_MODEL: Final[str] = "yolo11n.pt"
PRETRAINED_MODELS: Final[tuple[str, ...]] = (
    "yolo11n.pt",
    "yolov8n.pt",
)

DEFAULT_CONFIDENCE: Final[float] = 0.25
DEFAULT_IOU: Final[float] = 0.45
DEFAULT_IMAGE_SIZE: Final[int] = 640
DEFAULT_DEVICE: Final[str] = "auto"
DEFAULT_VIDEO_FPS: Final[float] = 30.0

ALLOWED_IMAGE_SIZES: Final[tuple[int, ...]] = (320, 480, 640, 960, 1280)
SUPPORTED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png"}
)
SUPPORTED_VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".mp4", ".avi", ".mov"}
)

DeviceChoice = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    """Validated settings passed to Ultralytics for each prediction.

    Model existence is intentionally checked by :class:`ObjectDetector` at
    load time.  Official model names may need to be downloaded, so validating
    them as ordinary local paths here would reject a supported use case.
    """

    model_path: str | Path = DEFAULT_MODEL
    confidence: float = DEFAULT_CONFIDENCE
    iou: float = DEFAULT_IOU
    image_size: int = DEFAULT_IMAGE_SIZE
    device: DeviceChoice | str = DEFAULT_DEVICE

    def __post_init__(self) -> None:
        model_reference = str(self.model_path).strip()
        if not model_reference:
            raise ValueError("A model name or path is required.")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if not 0.0 <= float(self.iou) <= 1.0:
            raise ValueError("IoU threshold must be between 0 and 1.")
        if isinstance(self.image_size, bool) or int(self.image_size) <= 0:
            raise ValueError("Image size must be a positive integer.")

        device = str(self.device).strip().lower()
        if device not in {"auto", "cpu", "cuda"} and not device.startswith("cuda:"):
            raise ValueError("Device must be Auto, CPU, CUDA, or a CUDA device index.")

        object.__setattr__(self, "model_path", model_reference)
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "iou", float(self.iou))
        object.__setattr__(self, "image_size", int(self.image_size))
        object.__setattr__(self, "device", device)
