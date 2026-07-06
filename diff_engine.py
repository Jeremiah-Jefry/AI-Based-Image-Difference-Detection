from __future__ import annotations
from summary import build_summary

from PIL import Image
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from skimage.metrics import structural_similarity
from typing import Any
import cv2
import fitz
import io
import numpy as np


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


@dataclass
class AlignmentResult:
    warped_image: np.ndarray | None
    homography: np.ndarray | None
    inlier_count: int
    match_count: int
    keypoints_before: int
    keypoints_after: int
    success: bool
    message: str


def align_images(
    image_before: np.ndarray,
    image_after: np.ndarray,
    *,
    nfeatures: int = 5000,
    ransac_reproj_threshold: float = 5.0,
    min_inliers: int = 10,
) -> dict[str, Any]:
    gray_before = _ensure_gray(image_before)
    gray_after = _ensure_gray(image_after)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints_before, descriptors_before = orb.detectAndCompute(gray_before, None)
    keypoints_after, descriptors_after = orb.detectAndCompute(gray_after, None)

    if descriptors_before is None or descriptors_after is None:
        result = AlignmentResult(
            warped_image=None,
            homography=None,
            inlier_count=0,
            match_count=0,
            keypoints_before=len(keypoints_before or []),
            keypoints_after=len(keypoints_after or []),
            success=False,
            message="ORB could not find enough features to align the images.",
        )
        return result.__dict__

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(descriptors_before, descriptors_after), key=lambda match: match.distance)

    if len(matches) < 4:
        result = AlignmentResult(
            warped_image=None,
            homography=None,
            inlier_count=0,
            match_count=len(matches),
            keypoints_before=len(keypoints_before),
            keypoints_after=len(keypoints_after),
            success=False,
            message="Not enough feature matches to estimate a homography.",
        )
        return result.__dict__

    points_before = np.float32([keypoints_before[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    points_after = np.float32([keypoints_after[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(points_before, points_after, cv2.RANSAC, ransac_reproj_threshold)
    if homography is None or mask is None:
        result = AlignmentResult(
            warped_image=None,
            homography=None,
            inlier_count=0,
            match_count=len(matches),
            keypoints_before=len(keypoints_before),
            keypoints_after=len(keypoints_after),
            success=False,
            message="Homography estimation failed during RANSAC.",
        )
        return result.__dict__

    inlier_count = int(mask.ravel().sum())
    height_after, width_after = image_after.shape[:2]
    warped_before = cv2.warpPerspective(image_before, homography, (width_after, height_after))
    success = inlier_count >= min_inliers
    message = (
        f"Alignment succeeded with {inlier_count} inliers."
        if success
        else f"Alignment produced only {inlier_count} inliers, which is below the confidence threshold."
    )

    result = AlignmentResult(
        warped_image=warped_before,
        homography=homography,
        inlier_count=inlier_count,
        match_count=len(matches),
        keypoints_before=len(keypoints_before),
        keypoints_after=len(keypoints_after),
        success=success,
        message=message,
    )
    return result.__dict__


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

        # Classification: Likely real change vs Likely noise/artifact
        # Solid, coherent blob, area above threshold, isolated vs small, fragmented, clustered
        bbox_area = int(component.get("bbox_area", area))
        density = area / bbox_area if bbox_area > 0 else 0

        is_noise = False
        if area < 800 and density < 0.2:
             is_noise = True

        # We can pass is_noise along in the region dictionary.


        from measurement import get_shape_descriptor, get_change_type, get_quadrant

        region_dict = {
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": bbox_xywh,
            "area": area,
            "bbox_area": bbox_area,
            "label": label,
            "quadrant": get_quadrant((x1, y1, x2, y2), gray_before.shape),
            "ssim_mean": round(ssim_mean, 4),
            "added_pixels": added_pixels,
            "removed_pixels": removed_pixels,
            "is_noise": is_noise
        }

        region_dict["shape_descriptor"] = get_shape_descriptor((x1, y1, x2, y2), gray_before.shape)
        region_dict["change_type"] = get_change_type(region_dict)

        regions.append(region_dict)

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


def run_diff_engine(
    before_source: Any,
    after_source: Any,
    *,
    dpi: int = 200,
    llm_model: str = "gemma3",
) -> dict[str, Any]:
    before_image = load_document_image(before_source, dpi=dpi)
    after_image = load_document_image(after_source, dpi=dpi)

    alignment = align_images(before_image, after_image)
    if alignment["warped_image"] is None:
        return {
            "before_image": before_image,
            "after_image": after_image,
            "alignment": alignment,
            "diff": None,
            "stats": None,
            "summary": alignment["message"],
            "llm_model": llm_model,
        }

    diff = analyze_differences(alignment["warped_image"], after_image)
    stats = {
        "region_count": diff["region_count"],
        "total_changed_area": diff["total_changed_area"],
        "changed_area_pct": diff["changed_area_pct"],
        "sheet_area": diff["sheet_area"],
        "regions": diff["regions"],
        "label_counts": diff["label_counts"],
    }
    summary = build_summary(stats)

    return {
        "before_image": before_image,
        "after_image": after_image,
        "alignment": alignment,
        "diff": diff,
        "stats": stats,
        "summary": summary,
        "llm_model": llm_model,
        "ssim_map": diff.get("ssim_map"),
        "ssim_score": diff.get("ssim_score"),
    }
