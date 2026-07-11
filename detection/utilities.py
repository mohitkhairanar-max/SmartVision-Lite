"""Small validation, device-selection, and export utilities.

Heavy optional dependencies are imported only inside the functions that need
them.  In particular, importing this module never probes CUDA.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from config.settings import (
    ALLOWED_IMAGE_SIZES,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)

DETECTION_EXPORT_FIELDS = (
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
)


class ValidationError(ValueError):
    """Raised when user-controlled configuration or input is invalid."""


class UnsupportedFileError(ValidationError):
    """Raised when a media filename has an unsupported extension."""


class DeviceUnavailableError(RuntimeError):
    """Raised when CUDA is explicitly requested but cannot be used."""


def validate_file_extension(
    filename: str | Path,
    allowed_extensions: Iterable[str],
    *,
    media_type: str = "file",
) -> str:
    """Return a normalized extension or raise :class:`UnsupportedFileError`.

    The function validates only the declared filename.  Decoders still need to
    validate the actual bytes because an extension does not guarantee that a
    file is well formed.
    """

    name = str(filename).strip()
    if not name:
        raise UnsupportedFileError(f"A {media_type} filename is required.")

    extension = Path(name).suffix.lower()
    allowed = {
        item.lower() if str(item).startswith(".") else f".{str(item).lower()}"
        for item in allowed_extensions
    }
    if extension not in allowed:
        supported = ", ".join(sorted(allowed))
        shown = extension or "no extension"
        raise UnsupportedFileError(
            f"Unsupported {media_type} format ({shown}). Supported formats: {supported}."
        )
    return extension


def validate_image_filename(filename: str | Path) -> str:
    """Validate and return the extension of an uploaded image filename."""

    return validate_file_extension(
        filename, SUPPORTED_IMAGE_EXTENSIONS, media_type="image"
    )


def validate_video_filename(filename: str | Path) -> str:
    """Validate and return the extension of an uploaded video filename."""

    return validate_file_extension(
        filename, SUPPORTED_VIDEO_EXTENSIONS, media_type="video"
    )


def is_supported_image(filename: str | Path) -> bool:
    """Return whether *filename* declares a supported image format."""

    try:
        validate_image_filename(filename)
    except UnsupportedFileError:
        return False
    return True


def is_supported_video(filename: str | Path) -> bool:
    """Return whether *filename* declares a supported video format."""

    try:
        validate_video_filename(filename)
    except UnsupportedFileError:
        return False
    return True


def validate_threshold(value: float, *, name: str = "threshold") -> float:
    """Return *value* as a float after checking the inclusive ``0..1`` range."""

    if isinstance(value, bool):
        raise ValidationError(f"{name.capitalize()} must be a number between 0 and 1.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{name.capitalize()} must be a number between 0 and 1."
        ) from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValidationError(f"{name.capitalize()} must be between 0 and 1.")
    return result


def validate_image_size(
    value: int,
    *,
    allowed_sizes: Iterable[int] | None = ALLOWED_IMAGE_SIZES,
) -> int:
    """Validate a positive image size and optionally restrict it to presets."""

    if isinstance(value, bool):
        raise ValidationError("Image size must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Image size must be a positive integer.") from exc
    if result <= 0:
        raise ValidationError("Image size must be a positive integer.")
    if allowed_sizes is not None:
        choices = tuple(int(size) for size in allowed_sizes)
        if result not in choices:
            joined = ", ".join(str(size) for size in choices)
            raise ValidationError(f"Image size must be one of: {joined}.")
    return result


def safe_filename(filename: str | Path, *, fallback: str = "file", max_length: int = 120) -> str:
    """Return a portable basename suitable for a generated output file."""

    if max_length < 8:
        raise ValueError("max_length must be at least 8 characters.")
    basename = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip(" ._")
    if not basename:
        basename = fallback

    stem = Path(basename).stem or fallback
    suffix = Path(basename).suffix
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    if stem.upper() in reserved:
        stem = f"_{stem}"

    available = max(1, max_length - len(suffix))
    return f"{stem[:available]}{suffix}"[:max_length]


def build_output_path(
    input_name: str | Path,
    output_directory: str | Path,
    *,
    prefix: str = "detected",
    extension: str | None = None,
) -> Path:
    """Build, but do not create, a safe output path for a media file."""

    clean_name = safe_filename(input_name)
    source = Path(clean_name)
    suffix = extension if extension is not None else source.suffix
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    clean_prefix = safe_filename(prefix, fallback="detected").replace(".", "_")
    return Path(output_directory) / f"{clean_prefix}_{source.stem}{suffix}"


def resolve_device(device: str = "auto", *, torch_module: Any | None = None) -> str:
    """Resolve ``auto``/``cpu``/``cuda`` without probing CUDA at import time.

    ``torch_module`` is an injection point used by deterministic unit tests. If
    PyTorch is unavailable, Auto safely resolves to CPU while an explicit CUDA
    request produces an actionable error.
    """

    choice = str(device).strip().lower()
    if choice == "cpu":
        return "cpu"
    if choice not in {"auto", "cuda"} and not choice.startswith("cuda:"):
        raise ValidationError("Device must be Auto, CPU, CUDA, or a CUDA device index.")

    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch
        except (ImportError, OSError):
            imported_torch = None
        torch = imported_torch

    cuda_available = False
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
        except (AttributeError, RuntimeError, OSError):
            cuda_available = False

    if choice == "auto":
        return "cuda" if cuda_available else "cpu"
    if not cuda_available:
        raise DeviceUnavailableError(
            "CUDA was requested but is not available. Select Auto or CPU, or install "
            "a CUDA-compatible PyTorch build and GPU driver."
        )

    if choice.startswith("cuda:"):
        try:
            index = int(choice.split(":", 1)[1])
        except ValueError as exc:
            raise ValidationError("CUDA device index must be an integer, for example cuda:0.") from exc
        if index < 0:
            raise ValidationError("CUDA device index cannot be negative.")
        try:
            count = int(torch.cuda.device_count())
        except (AttributeError, RuntimeError, OSError):
            count = 0
        if index >= count:
            raise DeviceUnavailableError(
                f"CUDA device {index} was requested, but only {count} device(s) are available."
            )
    return choice


def _json_safe(value: Any) -> Any:
    """Convert common scientific-Python values to strict JSON primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=repr)]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)


def _record_to_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        mapping = dict(record)
    else:
        to_dict = getattr(record, "to_dict", None)
        if callable(to_dict):
            mapping = dict(to_dict())
        elif is_dataclass(record) and not isinstance(record, type):
            mapping = asdict(record)
        else:
            raise TypeError("Each export record must be a mapping or provide to_dict().")
    return {str(key): _json_safe(value) for key, value in mapping.items()}


def records_to_json(records: Iterable[Any], *, indent: int = 2) -> str:
    """Serialize detection records as strict, human-readable JSON."""

    normalized = [_record_to_dict(record) for record in records]
    return json.dumps(normalized, ensure_ascii=False, indent=indent, allow_nan=False)


def records_to_csv(
    records: Iterable[Any],
    *,
    fieldnames: Iterable[str] | None = None,
) -> str:
    """Serialize detection records as CSV with deterministic column ordering."""

    normalized = [_record_to_dict(record) for record in records]
    if fieldnames is None:
        fields = list(DETECTION_EXPORT_FIELDS)
        for record in normalized:
            for key in record:
                if key not in fields:
                    fields.append(key)
    else:
        fields = [str(field) for field in fieldnames]
    if not fields:
        raise ValueError("At least one CSV field is required.")

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for record in normalized:
        writer.writerow(record)
    return stream.getvalue()


# Friendly aliases used by the UI and beginner-facing examples.
export_json = records_to_json
export_csv = records_to_csv


__all__ = [
    "DETECTION_EXPORT_FIELDS",
    "DeviceUnavailableError",
    "UnsupportedFileError",
    "ValidationError",
    "build_output_path",
    "export_csv",
    "export_json",
    "is_supported_image",
    "is_supported_video",
    "records_to_csv",
    "records_to_json",
    "resolve_device",
    "safe_filename",
    "validate_file_extension",
    "validate_image_filename",
    "validate_image_size",
    "validate_threshold",
    "validate_video_filename",
]
