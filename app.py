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
from utils.llm_report import (
    GROQ_MODELS,
    DEFAULT_MODEL,
    generate_llm_report_stream,
    groq_available,
)
from utils.lpips_diff import lpips_available
from utils.summary import build_summary

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
    """Draw bounding boxes with semi-transparent fills and clean labels."""
    overlay = image_bgr.copy()
    fill_layer = image_bgr.copy()

    for region in regions:
        x1, y1, x2, y2 = region["bbox_xyxy"]
        label = region["label"]

        # Colors: red for removed, green for added (BGR)
        if label == "added":
            box_color = (0, 200, 0)
            fill_color = (0, 180, 0)
        else:
            box_color = (0, 0, 220)
            fill_color = (0, 0, 200)

        # Semi-transparent fill
        cv2.rectangle(fill_layer, (int(x1), int(y1)), (int(x2), int(y2)), fill_color, -1)

        # Solid border
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)

        # Label with background rectangle for readability
        caption = f"{label.upper()} | {region['quadrant']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(caption, font, font_scale, thickness)
        text_x = int(x1)
        text_y = max(text_h + 8, int(y1) - 8)

        # Background rectangle for text
        cv2.rectangle(
            overlay,
            (text_x, text_y - text_h - 6),
            (text_x + text_w + 8, text_y + 4),
            box_color, -1,
        )
        cv2.putText(
            overlay, caption,
            (text_x + 4, text_y - 2),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    # Blend the fill layer at 20% opacity
    overlay = cv2.addWeighted(fill_layer, 0.15, overlay, 0.85, 0)
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

    enable_lpips = st.checkbox(
        "Enable LPIPS",
        value=False,
        disabled=not lpips_available(),
        help="Optional deep-feature perceptual distance (requires lpips + PyTorch).",
    )

    st.markdown("---")
    st.markdown("### 🤖 AI Report (Groq)")

    llm_model = st.selectbox(
        "Model",
        options=GROQ_MODELS,
        index=0,
        help="Select the Groq model for report generation.",
    )

    _groq_ok = groq_available()
    if _groq_ok:
        st.success("✅ Groq API configured", icon="🟢")
    else:
        st.warning("Groq API key not set — template fallback will be used.", icon="⚠️")

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
        st.caption("ORB alignment, Canny edge diffing, SSIM scoring, and region extraction.")
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
            dpi=dpi,
            enable_lpips=enable_lpips,
            llm_model=llm_model,
        )
    except Exception as exc:
        st.error(f"❌ Processing failed: {exc}")
        st.stop()

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

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ========================================================================
# STEP 2: Visual Comparison
# ========================================================================
_step_header(2, "Visual Comparison")

overlay = _overlay_regions(aligned_before, diff["regions"])

tab_side, tab_overlay, tab_slider = st.tabs(["📐 Side by Side", "🎯 Change Overlay", "↔️ Slider"])

with tab_side:
    left, right = st.columns(2)
    with left:
        st.markdown("**Before (aligned)**")
        st.image(_pil_from_bgr(aligned_before), use_container_width=True)
    with right:
        st.markdown("**After**")
        st.image(_pil_from_bgr(after_image), use_container_width=True)

with tab_overlay:
    # Legend
    st.markdown(
        '<div class="legend-container">'
        '<div class="legend-item"><div class="legend-dot removed"></div> Removed</div>'
        '<div class="legend-item"><div class="legend-dot added"></div> Added</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.image(_pil_from_bgr(overlay), use_container_width=True)

with tab_slider:
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

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ========================================================================
# STEP 3: Heatmap Analysis
# ========================================================================
_step_header(3, "Heatmap Analysis")

if ssim_map is not None:
    heatmap = _heatmap_from_ssim(ssim_map, after_image)
    st.caption("Smooth SSIM-based heatmap — brighter/warmer colors indicate greater structural difference.")
else:
    heatmap = _heatmap_fallback(diff["diff_mask"], after_image)
    st.caption("Edge-based difference heatmap.")

h_col1, h_col2 = st.columns([2, 1])
with h_col1:
    st.image(_pil_from_bgr(heatmap), use_container_width=True)
with h_col2:
    severity = _severity_from_stats(stats)
    severity_labels = {
        "none": ("None", "severity-none", "No meaningful changes detected."),
        "minor": ("Minor", "severity-minor", "Small, localized changes."),
        "moderate": ("Moderate", "severity-moderate", "Notable changes warrant review."),
        "major": ("Major", "severity-major", "Extensive modifications detected."),
    }
    sev_label, sev_class, sev_desc = severity_labels[severity]
    st.markdown(f'**Severity**: <span class="severity-badge {sev_class}">{sev_label}</span>', unsafe_allow_html=True)
    st.markdown(f"*{sev_desc}*")
    st.metric("Changed Area", f"{stats['changed_area_pct']:.2f}%")
    st.metric("Change Regions", stats["region_count"])
    if stats.get("label_counts"):
        added = stats["label_counts"].get("added", 0)
        removed = stats["label_counts"].get("removed", 0)
        st.metric("Added Regions", added)
        st.metric("Removed Regions", removed)
    if results.get("ssim_score") is not None:
        st.metric("Global SSIM", f"{results['ssim_score']:.4f}")

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ========================================================================
# STEP 4: AI Analysis Report
# ========================================================================
_step_header(4, "AI Analysis Report")

_groq_ok = groq_available()

if _groq_ok:
    st.markdown(f"*Generating report with **{llm_model}** via Groq...*")
    report_placeholder = st.empty()
    report_chunks: list[str] = []

    try:
        for chunk in generate_llm_report_stream(stats, model=llm_model):
            report_chunks.append(chunk)
            report_placeholder.markdown("".join(report_chunks))

        # Check if the stream returned a fallback message
        full_report = "".join(report_chunks)
        if full_report.startswith("> ⚠️"):
            # Fallback was triggered — show template summary too
            st.markdown("---")
            st.markdown("**Template Summary (fallback):**")
            st.markdown(summary)
    except Exception as exc:
        st.warning(f"LLM report generation failed: {exc}")
        st.markdown("**Template Summary (fallback):**")
        st.markdown(summary)
else:
    st.info(
        "🤖 **Groq API key not configured** — showing template-based summary. "
        "Get a free key at [console.groq.com](https://console.groq.com) and enter it in the sidebar."
    )
    st.markdown(
        f'<div class="report-container">{summary}</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ========================================================================
# STEP 5: Detailed Region Data
# ========================================================================
_step_header(5, "Detailed Region Data")

if stats["regions"]:
    # Clean up the data for display
    display_regions = []
    for i, region in enumerate(stats["regions"], 1):
        display_regions.append({
            "#": i,
            "Type": region["label"].upper(),
            "Quadrant": region["quadrant"],
            "Area (px)": region["area"],
            "SSIM": f"{region['ssim_mean']:.4f}",
            "Added px": region["added_pixels"],
            "Removed px": region["removed_pixels"],
        })
    st.dataframe(display_regions, use_container_width=True, hide_index=True)
else:
    st.info("No surviving regions after filtering.")

# ========================================================================
# LPIPS (optional)
# ========================================================================
if results.get("lpips") is not None:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    _step_header(6, "LPIPS Perceptual Distance")
    st.write(results["lpips"]["message"])
    if results["lpips"]["value"] is not None:
        st.metric("LPIPS Distance", f"{results['lpips']['value']:.4f}")
