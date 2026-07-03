# CAD/Image Structural Diff Detector

## Setup

Install the local dependencies with `pip install -r requirements.txt`.

## Run

Launch the app with `streamlit run app.py`.

## Functional coverage

FR-1 is handled by the file uploaders in `app.py` and document rasterization in `utils/pdf_utils.py`.
FR-2 is handled by ORB feature alignment and RANSAC homography in `utils/align.py`.
FR-3 is handled by the Canny edge-domain diff, tolerant dilation, SSIM scoring, and region extraction in `utils/diff.py`.
FR-4 is handled by the overlay, heatmap, and slider visualization in `app.py`.
FR-5 is handled by the statistics assembly in `utils/diff.py` and formatting in `utils/summary.py`.
FR-6 is handled by deterministic template-based generation in `utils/summary.py` with no external API calls.
# AI-Based-Image-Difference-Detection