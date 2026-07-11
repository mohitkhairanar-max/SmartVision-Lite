"""Public, lightweight API for SmartVision Lite detection features.

Importing this package does not load model weights, probe a webcam, or import
Ultralytics/OpenCV. Heavy dependencies are used only when their corresponding
operations are called.
"""

from .detector import (
    Detection,
    DetectionRecord,
    DetectionResult,
    Detector,
    DetectorError,
    InferenceError,
    ModelLoadError,
    ObjectDetector,
)
from .image_processor import (
    ImageProcessingError,
    ImageProcessor,
    encode_image,
    load_image,
    process_image,
)
from .metrics import (
    DetectionMetrics,
    FPSMeter,
    average_confidence,
    calculate_fps,
    count_by_class,
    get_class_counts,
    mean_confidence,
    summarize_detections,
)
from .utilities import (
    DeviceUnavailableError,
    UnsupportedFileError,
    ValidationError,
    build_output_path,
    export_csv,
    export_json,
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
from .video_processor import (
    CorruptVideoError,
    VideoProcessingError,
    VideoProcessingResult,
    VideoProcessor,
    process_video,
)

__all__ = [
    "CorruptVideoError",
    "Detection",
    "DetectionMetrics",
    "DetectionRecord",
    "DetectionResult",
    "Detector",
    "DetectorError",
    "DeviceUnavailableError",
    "FPSMeter",
    "ImageProcessingError",
    "ImageProcessor",
    "InferenceError",
    "ModelLoadError",
    "ObjectDetector",
    "UnsupportedFileError",
    "ValidationError",
    "VideoProcessingError",
    "VideoProcessingResult",
    "VideoProcessor",
    "average_confidence",
    "build_output_path",
    "calculate_fps",
    "count_by_class",
    "encode_image",
    "export_csv",
    "export_json",
    "get_class_counts",
    "is_supported_image",
    "is_supported_video",
    "load_image",
    "mean_confidence",
    "process_image",
    "process_video",
    "records_to_csv",
    "records_to_json",
    "resolve_device",
    "safe_filename",
    "summarize_detections",
    "validate_image_filename",
    "validate_image_size",
    "validate_threshold",
    "validate_video_filename",
]
