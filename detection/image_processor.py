"""Safe image decoding and detector integration."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

from .detector import DetectionResult, ObjectDetector
from .utilities import validate_image_filename


class ImageProcessingError(ValueError):
    """Raised when an uploaded image cannot be decoded safely."""


def _read_file_like(source: BinaryIO) -> bytes:
    position: int | None = None
    try:
        position = source.tell()
    except (AttributeError, OSError):
        pass
    try:
        data = source.read()
    except (AttributeError, OSError, ValueError) as exc:
        raise ImageProcessingError(f"Could not read the uploaded image: {exc}") from exc
    finally:
        if position is not None:
            try:
                source.seek(position)
            except (AttributeError, OSError, ValueError):
                pass
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ImageProcessingError("The uploaded image did not provide binary data.")
    return bytes(data)


def _decode_bytes(data: bytes) -> Any:
    if not data:
        raise ImageProcessingError("The uploaded image is empty.")
    try:
        import numpy as np
        from PIL import Image, UnidentifiedImageError
    except (ImportError, OSError) as exc:
        raise ImageProcessingError(
            "NumPy and Pillow are required to decode images. Install project dependencies."
        ) from exc

    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            return np.asarray(image.convert("RGB")).copy()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageProcessingError(
            "The image is invalid, corrupted, or uses an unsupported encoding."
        ) from exc


def _normalize_array(source: Any) -> Any:
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise ImageProcessingError("NumPy is required to process images.") from exc

    if not isinstance(source, np.ndarray):
        raise ImageProcessingError("Expected a NumPy image array.")
    if source.size == 0:
        raise ImageProcessingError("The image array is empty.")
    if source.ndim == 2:
        image = np.repeat(source[:, :, None], 3, axis=2)
    elif source.ndim == 3 and source.shape[2] == 1:
        image = np.repeat(source, 3, axis=2)
    elif source.ndim == 3 and source.shape[2] in {3, 4}:
        image = source[:, :, :3]
    else:
        raise ImageProcessingError(
            "Image arrays must have height, width, and one, three, or four channels."
        )

    if image.dtype != np.uint8:
        try:
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
            if np.issubdtype(image.dtype, np.floating) and image.size:
                minimum = float(image.min())
                maximum = float(image.max())
                if minimum >= 0.0 and maximum <= 1.0:
                    image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ImageProcessingError("The image array contains unsupported values.") from exc
    return np.ascontiguousarray(image)


def load_image(source: Any, *, filename: str | Path | None = None) -> Any:
    """Decode a path, upload, bytes, PIL image, or NumPy array to RGB/BGR pixels.

    Images decoded with Pillow are RGB. Existing NumPy arrays keep their color
    order because webcam frames conventionally arrive as BGR and Ultralytics
    accepts that representation directly.
    """

    if filename is not None:
        validate_image_filename(filename)

    try:
        import numpy as np
        from PIL import Image
    except (ImportError, OSError) as exc:
        raise ImageProcessingError(
            "NumPy and Pillow are required to process images. Install project dependencies."
        ) from exc

    if isinstance(source, np.ndarray):
        return _normalize_array(source)
    if isinstance(source, Image.Image):
        try:
            source.load()
            return np.asarray(source.convert("RGB")).copy()
        except (OSError, ValueError) as exc:
            raise ImageProcessingError("The PIL image could not be decoded.") from exc
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _decode_bytes(bytes(source))
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        validate_image_filename(path.name)
        if not path.is_file():
            raise ImageProcessingError(f"Image file was not found: {path}")
        try:
            return _decode_bytes(path.read_bytes())
        except OSError as exc:
            raise ImageProcessingError(f"Could not read image file '{path}': {exc}") from exc

    read = getattr(source, "read", None)
    if callable(read):
        upload_name = filename or getattr(source, "name", None)
        if upload_name:
            validate_image_filename(str(upload_name))
        return _decode_bytes(_read_file_like(source))
    raise ImageProcessingError(
        "Unsupported image input. Provide JPG, JPEG, or PNG bytes, a path, a PIL image, "
        "or a NumPy array."
    )


def encode_image(
    image: Any,
    *,
    image_format: str = "PNG",
    input_is_bgr: bool = True,
) -> bytes:
    """Encode an image array for download.

    Ultralytics ``Result.plot()`` returns BGR pixels, so ``input_is_bgr`` is
    true by default. Set it to false for RGB arrays produced by ``load_image``.
    """

    try:
        from PIL import Image
    except (ImportError, OSError) as exc:
        raise ImageProcessingError("Pillow is required to encode images.") from exc

    pixels = _normalize_array(image)
    if input_is_bgr:
        pixels = pixels[:, :, ::-1]
    output = BytesIO()
    normalized_format = str(image_format).strip().upper()
    if normalized_format not in {"PNG", "JPEG"}:
        raise ImageProcessingError("Download image format must be PNG or JPEG.")
    try:
        Image.fromarray(pixels).save(output, format=normalized_format)
    except (OSError, ValueError) as exc:
        raise ImageProcessingError(f"Could not encode the annotated image: {exc}") from exc
    return output.getvalue()


class ImageProcessor:
    """Decode image inputs and pass them to an :class:`ObjectDetector`."""

    def __init__(self, detector: ObjectDetector) -> None:
        if detector is None or not callable(getattr(detector, "detect", None)):
            raise TypeError("detector must provide a detect(source) method.")
        self.detector = detector

    def process(
        self,
        source: Any,
        *,
        filename: str | Path | None = None,
    ) -> DetectionResult:
        image = load_image(source, filename=filename)
        return self.detector.detect(image)

    process_image = process


def process_image(
    detector: ObjectDetector,
    source: Any,
    *,
    filename: str | Path | None = None,
) -> DetectionResult:
    """Functional convenience wrapper around :class:`ImageProcessor`."""

    return ImageProcessor(detector).process(source, filename=filename)


__all__ = [
    "ImageProcessingError",
    "ImageProcessor",
    "encode_image",
    "load_image",
    "process_image",
]
