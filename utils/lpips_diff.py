from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def lpips_available() -> bool:
    try:
        import lpips  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def compute_lpips_distance(image_before: np.ndarray, image_after: np.ndarray) -> dict[str, Any]:
    try:
        import lpips
        import torch
    except Exception as exc:
        return {"available": False, "value": None, "message": f"LPIPS is unavailable: {exc}"}

    def _prepare(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 127.5 - 1.0
        return np.transpose(normalized, (2, 0, 1))

    tensor_before = torch.from_numpy(_prepare(image_before)).unsqueeze(0)
    tensor_after = torch.from_numpy(_prepare(image_after)).unsqueeze(0)
    model = lpips.LPIPS(net="alex")
    with torch.no_grad():
        value = float(model(tensor_before, tensor_after).item())
    return {"available": True, "value": value, "message": "LPIPS distance computed locally."}
