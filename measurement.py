from typing import Any
import cv2
import numpy as np

def get_shape_descriptor(bbox_xyxy: tuple[int, int, int, int], image_shape: tuple[int, int]) -> str:
    """Returns a shape descriptor based on aspect ratio and bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1

    if w == 0 or h == 0:
        return "irregular region"

    aspect_ratio = w / h

    # Heuristics based on prompt
    img_h, img_w = image_shape[:2]
    cy = (y1 + y2) / 2

    # near grade = bottom 20% of image
    is_near_grade = cy > (img_h * 0.8)
    # near window band = middle 40-60% of image roughly
    is_near_window_band = (img_h * 0.3) < cy < (img_h * 0.7)

    if aspect_ratio > 3.0:
        if is_near_grade:
            return "sill/base line"
        return "thin linear element"
    elif aspect_ratio < 0.33:
        if is_near_window_band:
            return "opening"
        return "thin linear element"
    else:
        return "rectangular element"


def get_change_type(region: dict[str, Any]) -> str:
    """Returns string change type Added/Removed/Modified/Resized"""
    added_pixels = region.get("added_pixels", 0)
    removed_pixels = region.get("removed_pixels", 0)

    # Thresholding for classification
    if added_pixels > 0 and removed_pixels == 0:
        return "Added"
    elif removed_pixels > 0 and added_pixels == 0:
        return "Removed"
    elif added_pixels > 0 and removed_pixels > 0:
        # Check if the region has mostly modified edges inside
        return "Modified"
    return "Resized"

def get_quadrant(bbox_xyxy: tuple[int, int, int, int], image_shape: tuple[int, int]) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    width = image_shape[1]
    height = image_shape[0]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    # 3x3 grid
    col = "center"
    if center_x < width / 3.0:
        col = "left"
    elif center_x > 2 * width / 3.0:
        col = "right"

    row = "middle"
    if center_y < height / 3.0:
        row = "upper"
    elif center_y > 2 * height / 3.0:
        row = "lower"

    if row == "middle" and col == "center":
        return "middle-center"

    return f"{row}-{col}"
