from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


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


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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
