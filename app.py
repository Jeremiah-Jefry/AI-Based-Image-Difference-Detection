from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_image_comparison import image_comparison

from diff_engine import run_diff_engine
from summary import build_summary

# Load .env file for GROQ_API_KEY
load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CAD/Image Structural Diff Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark theme polish
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30,40,60,0.8), rgba(20,30,50,0.9));
        border: 1px solid rgba(100,140,255,0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: rgba(180,200,255,0.8) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 2rem !important;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Step headers */
    .step-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 24px 0 12px 0;
        padding: 12px 16px;
        background: linear-gradient(90deg, rgba(59,130,246,0.12), transparent);
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
    }
    .step-header .step-num {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
    .step-header .step-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    /* Report container */
    .report-container {
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,41,59,0.8));
        border: 1px solid rgba(100,140,255,0.15);
        border-radius: 16px;
        padding: 28px 32px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    /* Legend */
    .legend-container {
        display: flex;
        gap: 24px;
        padding: 10px 16px;
        background: rgba(15,23,42,0.6);
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid rgba(100,140,255,0.1);
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .legend-dot {
        width: 14px;
        height: 14px;
        border-radius: 3px;
    }
    .legend-dot.removed { background-color: #ef4444; }
    .legend-dot.added { background-color: #22c55e; }

    /* Severity badges */
    .severity-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .severity-none { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .severity-minor { background: rgba(234,179,8,0.15); color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
    .severity-moderate { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
    .severity-major { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

    /* Hide default header/footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(100,140,255,0.3), transparent);
        margin: 32px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _step_header(number: int, title: str) -> None:
    st.markdown(
        f'<div class="step-header">'
        f'<div class="step-num">{number}</div>'
        f'<div class="step-title">{title}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _pil_from_bgr(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(_to_rgb(image_bgr))


def _overlay_regions(image_bgr: np.ndarray, regions: list[dict[str, object]]) -> np.ndarray:
    overlay = image_bgr.copy()

    real_changes = [r for r in regions if not r.get("is_noise", False)]

    for i, region in enumerate(real_changes, 1):
        x1, y1, x2, y2 = region["bbox_xyxy"]
        label = region.get("change_type", "modified").lower()

        # Determine color
        if "added" in label or "new" in label:
            bg_color = (0, 200, 0)
        elif "removed" in label:
            bg_color = (0, 0, 220)
        else:
            bg_color = (200, 150, 0) # Modified/Resized

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(overlay, (cx, cy), 18, bg_color, -1)
        cv2.circle(overlay, (cx, cy), 18, (255,255,255), 2)

        text = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
        cv2.putText(overlay, text, (cx - int(tw/2), cy + int(th/2)), font, 0.6, (255,255,255), 2, cv2.LINE_AA)

    return overlay


def _heatmap_from_ssim(ssim_map: np.ndarray, base_image_bgr: np.ndarray) -> np.ndarray:
    """Create a smooth heatmap from the SSIM difference map (1 - SSIM)."""
    # SSIM map values are in [0, 1] — invert so differences are bright
    diff_intensity = (1.0 - ssim_map).clip(0, 1)
    # Scale to uint8
    diff_uint8 = (diff_intensity * 255).astype(np.uint8)
    # Apply colormap
    colored = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)
    # Resize to match base image if needed
    if colored.shape[:2] != base_image_bgr.shape[:2]:
        colored = cv2.resize(colored, (base_image_bgr.shape[1], base_image_bgr.shape[0]))
    return cv2.addWeighted(base_image_bgr, 0.55, colored, 0.45, 0.0)


def _heatmap_fallback(diff_mask: np.ndarray, base_image_bgr: np.ndarray) -> np.ndarray:
    """Fallback binary heatmap when SSIM map is unavailable."""
    normalized = cv2.normalize(diff_mask, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return cv2.addWeighted(base_image_bgr, 0.7, colored, 0.3, 0.0)


def _write_temp_image(image_bgr: np.ndarray, filename: str) -> str:
    output_path = Path(__file__).resolve().parent / "outputs" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)
    return str(output_path)


def _severity_from_stats(stats: dict) -> str:
    """Determine severity level from stats."""
    region_count = stats.get("region_count", 0)
    pct = stats.get("changed_area_pct", 0.0)
    if region_count == 0 or pct < 0.05:
        return "none"
    if region_count <= 3 and pct < 0.5:
        return "minor"
    if region_count <= 15 and pct < 1.5:
        return "moderate"
    return "major"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.markdown(
    '<h1 style="text-align:center; background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6); '
    '-webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem; '
    'font-weight: 800; margin-bottom: 4px;">🔍 Structural Diff Detector</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align:center; color: rgba(180,200,255,0.6); font-size: 0.85rem; margin-bottom: 32px;">'
    'AI-powered architectural drawing comparison • ORB alignment • Edge-domain diffing • LLM analysis</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 📂 Upload Files")
    before_file = st.file_uploader(
        "Before file", type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"], key="before"
    )
    after_file = st.file_uploader(
        "After file", type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"], key="after"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    dpi = st.slider("Rasterization DPI", min_value=100, max_value=300, value=200, step=25)

    st.markdown("---")


    st.markdown("---")
    run_requested = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if not run_requested:
    # Landing state
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📄 Upload")
        st.caption("Upload before & after PDF or image files of your architectural drawings.")
    with col2:
        st.markdown("#### 🔬 Analyze")
        st.caption("Analyze drawing differences.")
    with col3:
        st.markdown("#### 📊 Report")
        st.caption("AI-generated report with findings, severity assessment, and recommendations.")

    st.info("👈 Upload a before/after pair in the sidebar and click **Run Comparison** to get started.")
    st.stop()

if before_file is None or after_file is None:
    st.error("⚠️ Upload both a **before** and **after** file before running the comparison.")
    st.stop()

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
with st.spinner("🔄 Running alignment and structural diff analysis..."):
    try:
        results = run_diff_engine(
            before_file, after_file,
            dpi=dpi
        )
    except Exception as exc:
        st.error(f"❌ Processing failed: {exc}")
        st.stop()


if 'results' in locals():
    alignment = results["alignment"]
    diff = results["diff"]
    stats = results["stats"]
    summary = results["summary"]
    ssim_map = results.get("ssim_map")

    # ========================================================================
    # STEP 1: Alignment
    # ========================================================================
    _step_header(1, "Alignment Result")

    a_col1, a_col2, a_col3 = st.columns(3)
    with a_col1:
        st.metric("ORB/RANSAC Inliers", alignment["inlier_count"])
    with a_col2:
        st.metric("Keypoints (Before)", alignment["keypoints_before"])
    with a_col3:
        st.metric("Keypoints (After)", alignment["keypoints_after"])

    if not alignment["success"]:
        st.warning(f"⚠️ {alignment['message']}")
        st.stop()
    else:
        st.success(f"✅ {alignment['message']}")

    if diff is None or stats is None:
        st.stop()

    aligned_before = alignment["warped_image"]
    after_image = results["after_image"]

    real_changes = [r for r in stats["regions"] if not r.get("is_noise", False)]
    noise_changes = [r for r in stats["regions"] if r.get("is_noise", False)]

    # Verdict Banner
    st.markdown("---")
    st.markdown(f"### Cover Summary")
    st.markdown(f"**Verdict:** {summary}")

    if not real_changes:
        st.success("✅ No revisions detected between the two drawing versions.")
    else:
        # 2. Revision schedule
        st.markdown("### Revision Schedule")
        schedule_data = []

        from genai_caption import generate_caption

        for i, region in enumerate(real_changes, 1):
            desc = generate_caption(aligned_before, after_image, region["bbox_xyxy"])

            type_mapping = {
                "Added": "New element",
                "Removed": "Removed element",
                "Modified": "Revised element",
                "Resized": "Element resized"
            }

            ui_type = type_mapping.get(region.get("change_type", ""), region.get("change_type", "Unknown"))

            if not desc:
                desc = f"{ui_type} ({region.get('shape_descriptor', 'element')}) identified."

            schedule_data.append({
                "Item #": i,
                "Location": region["quadrant"],
                "Change Type": ui_type,
                "Description": desc,
                "Approx. Size": f"{region['area']} px"
            })

        st.table(schedule_data)

        # 3. Annotated drawings
        st.markdown("### Annotated Drawing")
        overlay = _overlay_regions(aligned_before, stats["regions"])
        st.image(_pil_from_bgr(overlay), use_container_width=True)

        # 4. Per-item detail cards
        st.markdown("### Detailed Callouts")
        for i, region in enumerate(real_changes, 1):
            with st.container():
                st.markdown(f"**Item #{i}: {region['quadrant']}**")

                type_mapping = {
                    "Added": "New element",
                    "Removed": "Removed element",
                    "Modified": "Revised element",
                    "Resized": "Element resized"
                }
                ui_type = type_mapping.get(region.get("change_type", ""), region.get("change_type", "Unknown"))

                st.markdown(f"*{ui_type} | {region['area']} px*")

                desc = generate_caption(aligned_before, after_image, region["bbox_xyxy"])
                if desc:
                    st.write(desc)
                else:
                    st.write(f"A {region.get('shape_descriptor', 'element')} was {ui_type.lower()}.")

                # Crops
                x1, y1, x2, y2 = region["bbox_xyxy"]

                pad = 20
                h, w = aligned_before.shape[:2]
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(w, x2+pad), min(h, y2+pad)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(_pil_from_bgr(aligned_before[cy1:cy2, cx1:cx2]), caption="Before")
                with c2:
                    st.image(_pil_from_bgr(after_image[cy1:cy2, cx1:cx2]), caption="After")
                st.markdown("---")

    # Slider
    st.markdown("### Full-sheet comparison")
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

    # 5. Technical appendix
    with st.expander("Technical appendix"):
        st.markdown("### Match Quality & Details")
        st.write(f"**Alignment Inliers:** {alignment['inlier_count']} / {alignment['match_count']}")

        if noise_changes:
            st.write(f"**Noise / Scan Variations:** {len(noise_changes)} regions filtered out.")

        st.markdown("**Parameters:** ORB alignment + RANSAC. Canny edge-domain structural differencing.")

    # 6. Limitation note
    st.caption("This report identifies and measures visual differences between drawing versions. It does not verify code compliance, structural adequacy, or dimensional accuracy against design intent — items should be confirmed against the drawing set by a qualified reviewer.")
