from __future__ import annotations

from collections import Counter
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _box_near(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + gap < bx1 or bx2 + gap < ax1 or ay2 + gap < by1 or by2 + gap < ay1)


def _merge_boxes(boxes: list[dict[str, Any]], gap: int) -> list[dict[str, Any]]:
    merged = boxes[:]
    changed = True
    while changed:
        changed = False
        next_round: list[dict[str, Any]] = []
        while merged:
            current = merged.pop()
            current_box = current["bbox_xyxy"]
            merged_into_existing = False
            for existing in next_round:
                if _box_near(current_box, existing["bbox_xyxy"], gap):
                    x1 = min(current_box[0], existing["bbox_xyxy"][0])
                    y1 = min(current_box[1], existing["bbox_xyxy"][1])
                    x2 = max(current_box[2], existing["bbox_xyxy"][2])
                    y2 = max(current_box[3], existing["bbox_xyxy"][3])
                    existing["bbox_xyxy"] = (x1, y1, x2, y2)
                    existing["components"].extend(current["components"])
                    existing["area"] += current["area"]
                    existing["added_pixels"] += current["added_pixels"]
                    existing["removed_pixels"] += current["removed_pixels"]
                    existing["ssim_weighted_sum"] += current["ssim_weighted_sum"]
                    existing["ssim_weight"] += current["ssim_weight"]
                    merged_into_existing = True
                    changed = True
                    break
            if not merged_into_existing:
                next_round.append(current)
        merged = next_round
    return merged


def _quadrant_label(bbox_xyxy: tuple[int, int, int, int], image_shape: tuple[int, int]) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    width = image_shape[1]
    height = image_shape[0]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    horizontal = "left" if center_x < width / 2.0 else "right"
    vertical = "upper" if center_y < height / 2.0 else "lower"
    return f"{vertical}-{horizontal}"


def _score_region(ssim_map: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 1.0
    return float(np.clip(ssim_map[mask].mean(), 0.0, 1.0))


def analyze_differences(
    aligned_before: np.ndarray,
    after_image: np.ndarray,
    *,
    blur_kernel_size: int = 5,
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
    dilation_iterations: int = 2,
    cleanup_kernel_size: int = 3,
    min_region_area: int = 500,
    merge_gap: int = 12,
) -> dict[str, Any]:
    gray_before = _ensure_gray(aligned_before)
    gray_after = _ensure_gray(after_image)

    blur_kernel = _odd(blur_kernel_size)
    blurred_before = cv2.GaussianBlur(gray_before, (blur_kernel, blur_kernel), 0)
    blurred_after = cv2.GaussianBlur(gray_after, (blur_kernel, blur_kernel), 0)

    edges_before = cv2.Canny(blurred_before, canny_threshold1, canny_threshold2)
    edges_after = cv2.Canny(blurred_after, canny_threshold1, canny_threshold2)

    tolerance_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    expanded_before = cv2.dilate(edges_before, tolerance_kernel, iterations=dilation_iterations)
    expanded_after = cv2.dilate(edges_after, tolerance_kernel, iterations=dilation_iterations)

    removed_mask = cv2.bitwise_and(edges_before, cv2.bitwise_not(expanded_after))
    added_mask = cv2.bitwise_and(edges_after, cv2.bitwise_not(expanded_before))
    diff_mask = cv2.bitwise_or(removed_mask, added_mask)

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cleanup_kernel_size, cleanup_kernel_size))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)
    diff_mask = cv2.dilate(diff_mask, cleanup_kernel, iterations=1)

    ssim_score, ssim_map = structural_similarity(gray_before, gray_after, full=True, data_range=255)

    connected_count, labels, stats, _ = cv2.connectedComponentsWithStats(diff_mask, connectivity=8)
    components: list[dict[str, Any]] = []
    for label_index in range(1, connected_count):
        x, y, w, h, area = stats[label_index]
        bbox_area = int(w * h)
        if bbox_area < min_region_area:
            continue
        component_mask = labels == label_index
        added_pixels = int(np.count_nonzero(np.logical_and(component_mask, added_mask > 0)))
        removed_pixels = int(np.count_nonzero(np.logical_and(component_mask, removed_mask > 0)))
        if added_pixels == 0 and removed_pixels == 0:
            continue
        region_ssim = _score_region(ssim_map, component_mask)
        components.append(
            {
                "bbox_xyxy": (int(x), int(y), int(x + w), int(y + h)),
                "area": int(area),
                "bbox_area": bbox_area,
                "added_pixels": added_pixels,
                "removed_pixels": removed_pixels,
                "ssim_weighted_sum": region_ssim * int(area),
                "ssim_weight": int(area),
                "components": [label_index],
            }
        )

    merged_components = _merge_boxes(components, merge_gap)

    regions: list[dict[str, Any]] = []
    for component in merged_components:
        x1, y1, x2, y2 = component["bbox_xyxy"]
        area = int(component["area"])
        added_pixels = int(component["added_pixels"])
        removed_pixels = int(component["removed_pixels"])
        label = "added" if added_pixels >= removed_pixels else "removed"
        ssim_mean = float(component["ssim_weighted_sum"] / component["ssim_weight"]) if component["ssim_weight"] else 1.0
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
        regions.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": bbox_xywh,
                "area": area,
                "bbox_area": int(component.get("bbox_area", area)),
                "label": label,
                "quadrant": _quadrant_label(component["bbox_xyxy"], gray_before.shape),
                "ssim_mean": round(ssim_mean, 4),
                "added_pixels": added_pixels,
                "removed_pixels": removed_pixels,
            }
        )

    regions.sort(key=lambda item: (item["bbox_xyxy"][1], item["bbox_xyxy"][0]))

    total_changed_area = int(sum(region["area"] for region in regions))
    sheet_area = int(gray_before.shape[0] * gray_before.shape[1]) if gray_before.size else 0
    changed_area_pct = float((total_changed_area / sheet_area) * 100.0) if sheet_area else 0.0

    return {
        "edges_before": edges_before,
        "edges_after": edges_after,
        "removed_mask": removed_mask,
        "added_mask": added_mask,
        "diff_mask": diff_mask,
        "ssim_score": float(ssim_score),
        "ssim_map": ssim_map,
        "regions": regions,
        "region_count": len(regions),
        "total_changed_area": total_changed_area,
        "changed_area_pct": round(changed_area_pct, 4),
        "sheet_area": sheet_area,
        "label_counts": dict(Counter(region["label"] for region in regions)),
    }
