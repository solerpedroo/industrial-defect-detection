from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"


class ImageValidationError(ValueError):
    """Client-provided image input that cannot enter inference."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class DecodedImage:
    """Ephemeral RGB image normalized according to ADR-004."""

    image: np.ndarray
    width: int
    height: int
    source_format: str
    file_size_bytes: int


def _declared_format(data: bytes) -> str:
    if data.startswith(PNG_SIGNATURE):
        return "PNG"
    if data.startswith(JPEG_SIGNATURE):
        return "JPEG"
    raise ImageValidationError("Only PNG and JPEG image bytes are accepted.", 415)


def decode_upload(data: bytes) -> DecodedImage:
    """Validate and decode PNG/JPEG bytes to contiguous RGB uint8 pixels.

    The output has shape ``(height, width, 3)``, RGB channel order, no alpha,
    dtype ``uint8``, and values in ``[0, 255]``. It reads no filesystem state.
    """
    if not data:
        raise ImageValidationError("The uploaded image is empty.", 400)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("The uploaded image exceeds the 10 MiB limit.", 413)

    expected_format = _declared_format(data)
    try:
        with Image.open(BytesIO(data)) as probe:
            if probe.format != expected_format:
                raise ImageValidationError("Image bytes do not match their declared format.", 415)
            width, height = probe.size
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise ImageValidationError("Image dimensions exceed the 4096 × 4096 limit.", 413)
            probe.verify()

        with Image.open(BytesIO(data)) as source:
            source.load()
            rgb = source.convert("RGB")
            pixels = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    except ImageValidationError:
        raise
    except (OSError, UnidentifiedImageError) as error:
        raise ImageValidationError("The uploaded image is invalid or corrupted.", 415) from error

    return DecodedImage(
        image=pixels,
        width=width,
        height=height,
        source_format=expected_format,
        file_size_bytes=len(data),
    )
