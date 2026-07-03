from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

from diff_engine import run_diff_engine
from utils.lpips_diff import lpips_available


st.set_page_config(page_title="CAD/Image Structural Diff Detector", layout="wide")


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _pil_from_bgr(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(_to_rgb(image_bgr))


def _overlay_regions(image_bgr: np.ndarray, regions: list[dict[str, object]]) -> np.ndarray:
    overlay = image_bgr.copy()
    for region in regions:
        x1, y1, x2, y2 = region["bbox_xyxy"]
        label = region["label"]
        color = (0, 200, 0) if label == "added" else (0, 0, 220)
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        caption = f"{label} | {region['quadrant']} | SSIM {region['ssim_mean']:.2f}"
        text_origin = (int(x1), max(18, int(y1) - 8))
        cv2.putText(overlay, caption, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return overlay


def _heatmap(diff_mask: np.ndarray, base_image_bgr: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(diff_mask, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(base_image_bgr, 0.7, colored, 0.3, 0.0)


def _write_temp_image(image_bgr: np.ndarray, filename: str) -> str:
    output_path = Path(__file__).resolve().parent / "outputs" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)
    return str(output_path)


def _read_upload(uploaded_file) -> io.BytesIO | str | None:
    if uploaded_file is None:
        return None
    return uploaded_file


st.title("CAD/Image Structural Diff Detector")
st.caption("Local ORB + homography alignment, edge-domain diffing, region extraction, and template-based summary.")

with st.sidebar:
    st.header("Inputs")
    before_file = st.file_uploader("Before file", type=["pdf", "png", "jpg", "jpeg"], key="before")
    after_file = st.file_uploader("After file", type=["pdf", "png", "jpg", "jpeg"], key="after")
    enable_lpips = st.checkbox(
        "Enable LPIPS if installed",
        value=False,
        disabled=not lpips_available(),
        help="Optional deep-feature distance; the app still runs without lpips installed.",
    )
    dpi = st.slider("Rasterization DPI", min_value=100, max_value=300, value=200, step=25)
    run_requested = st.button("Run comparison", type="primary")

if not lpips_available():
    st.info("LPIPS is not installed, so the optional deep-feature path remains inactive.")

if run_requested:
    if before_file is None or after_file is None:
        st.error("Upload both a before and after file before running the comparison.")
        st.stop()

    with st.spinner("Running alignment and structural diff analysis..."):
        try:
            results = run_diff_engine(_read_upload(before_file), _read_upload(after_file), dpi=dpi, enable_lpips=enable_lpips)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            st.stop()

    alignment = results["alignment"]
    diff = results["diff"]
    stats = results["stats"]
    summary = results["summary"]

    st.subheader("Alignment")
    st.metric("ORB/RANSAC inliers", alignment["inlier_count"])
    if not alignment["success"]:
        st.warning(alignment["message"])
    else:
        st.success(alignment["message"])

    if diff is None or stats is None:
        st.stop()

    aligned_before = results["alignment"]["warped_image"]
    after_image = results["after_image"]
    overlay = _overlay_regions(aligned_before, diff["regions"])
    heatmap = _heatmap(diff["diff_mask"], after_image)

    left, right = st.columns(2)
    with left:
        st.subheader("Aligned comparison")
        st.image([_pil_from_bgr(aligned_before), _pil_from_bgr(after_image)], caption=["Before (aligned)", "After"], use_container_width=True)
    with right:
        st.subheader("Change overlay")
        st.image(_pil_from_bgr(overlay), use_container_width=True)

    st.subheader("Before / After Slider")
    aligned_before_path = _write_temp_image(aligned_before, "aligned_before_slider.png")
    after_path = _write_temp_image(after_image, "after_slider.png")
    image_comparison(
        img1=aligned_before_path,
        img2=after_path,
        label1="Before (aligned)",
        label2="After",
        width=704,
        in_memory=False,
    )

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.subheader("Heatmap")
        st.image(_pil_from_bgr(heatmap), use_container_width=True)
    with lower_right:
        st.subheader("Summary")
        st.write(summary)
        st.metric("Changed area", f"{stats['changed_area_pct']:.2f}%")
        st.metric("Changed regions", stats["region_count"])

    st.subheader("Region stats")
    if stats["regions"]:
        st.dataframe(stats["regions"], use_container_width=True, hide_index=True)
    else:
        st.info("No surviving regions after filtering.")

    if results.get("lpips") is not None:
        st.subheader("LPIPS")
        st.write(results["lpips"]["message"])
        if results["lpips"]["value"] is not None:
            st.metric("LPIPS distance", f"{results['lpips']['value']:.4f}")

else:
    st.info("Upload a before/after pair and click Run comparison.")
