from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
from PIL import Image


def _read_source_bytes(source: Any) -> tuple[bytes, str | None]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.read_bytes(), path.suffix.lower()

    if hasattr(source, "read"):
        try:
            current_position = source.tell()
        except Exception:
            current_position = None
        data = source.read()
        if current_position is not None:
            try:
                source.seek(current_position)
            except Exception:
                pass
        suffix = getattr(source, "name", None)
        suffix_text = Path(suffix).suffix.lower() if suffix else None
        return data, suffix_text

    if isinstance(source, (bytes, bytearray)):
        return bytes(source), None

    raise TypeError("Unsupported image source type")


def _is_pdf(data: bytes, suffix: str | None) -> bool:
    if suffix == ".pdf":
        return True
    return data[:4] == b"%PDF"


def _pixmap_to_bgr(pixmap: fitz.Pixmap) -> np.ndarray:
    samples = np.frombuffer(pixmap.samples, dtype=np.uint8)
    channels = pixmap.n
    image = samples.reshape(pixmap.height, pixmap.width, channels)
    if image.ndim == 2 or channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if pixmap.alpha:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def load_document_image(source: Any, dpi: int = 200) -> np.ndarray:
    data, suffix = _read_source_bytes(source)
    if _is_pdf(data, suffix):
        document = fitz.open(stream=data, filetype="pdf")
        try:
            page = document.load_page(0)
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            return _pixmap_to_bgr(pixmap)
        finally:
            document.close()

    image = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
