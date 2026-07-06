import cv2
import numpy as np
from typing import Optional

def _crop_padded(image: np.ndarray, bbox_xyxy: tuple[int, int, int, int], padding_pct: float = 0.3) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    h, w = image.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding_pct)
    pad_y = int(bh * padding_pct)

    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w, x2 + pad_x)
    y2_pad = min(h, y2 + pad_y)

    return image[y1_pad:y2_pad, x1_pad:x2_pad]

def generate_caption(before_image: np.ndarray, after_image: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> Optional[str]:
    try:
        import ollama
        import time
        import io
        from PIL import Image

        before_crop = _crop_padded(before_image, bbox_xyxy)
        after_crop = _crop_padded(after_image, bbox_xyxy)

        # Encode to jpg bytes
        _, before_encoded = cv2.imencode('.jpg', before_crop)
        _, after_encoded = cv2.imencode('.jpg', after_crop)

        prompt = "These two image crops show the same location in an architectural drawing, before and after a revision. In one short sentence, describe what changed. If nothing meaningful changed, say so. Do not guess measurements you cannot see."

        # This will fail fast if ollama service is not running locally
        response = ollama.chat(
            model='llava',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [before_encoded.tobytes(), after_encoded.tobytes()]
            }],
            options={'timeout': 15}
        )
        return response['message']['content'].strip()
    except Exception:
        return None
