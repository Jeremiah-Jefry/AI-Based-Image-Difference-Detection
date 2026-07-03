# Structural Diff Detector

An advanced, AI-powered image comparison tool designed for architectural drawings, CAD exports, and engineering diagrams. This application automatically aligns, compares, and highlights structural differences between two versions of a document, generating a rich, user-friendly AI report to summarize the findings.

##  Features

- **Multi-Format Support**: Upload PDF, PNG, JPG, JPEG, TIFF, BMP, or WebP files.
- **Smart Alignment**: Uses ORB feature detection and RANSAC homography to automatically align drawings that might have been shifted, rotated, or scaled.
- **Structural Diffing**: Employs Canny edge-domain difference detection combined with SSIM (Structural Similarity Index) to eliminate noise and identify genuine additions and removals.
- **Interactive Visualizations**: 
  - **Change Overlay**: Highlights added regions in green and removed regions in red with bounding boxes.
  - **Structural Heatmap**: A continuous SSIM-based jet map showing the exact intensity of structural variations.
  - **Slider Tab**: A seamless drag-to-compare interactive slider.
- **AI-Powered Reporting**: Uses blazing-fast cloud inference via the **Groq API** to generate a comprehensive, actionable executive summary, severity assessment, and field recommendations.


##  Installation

1. Clone this repository.
2. Install the local dependencies:
   ```bash
   pip install -r requirements.txt
   ```

##  Configuration

To enable the AI Analysis Report, you need a free Groq API key:

1. Get an API key from the [Groq Console](https://console.groq.com).
2. Create a `.env` file in the root of the project.
3. Add your key to the file:
   ```env
   GROQ_API_KEY=gsk_your_api_key_here
   ```

##  Usage

Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will start on `http://localhost:8501`. 
1. Upload your "Before" and "After" documents in the sidebar.
2. Ensure the Groq status indicator shows **✅ Groq API configured**.
3. Click **Run Comparison** and explore the 5-step analysis report.

##  Architecture

- **Alignment**: `utils/align.py` (ORB + Homography)
- **Diffing**: `utils/diff.py` (Canny edge-domain + connected components)
- **LLM Engine**: `utils/llm_report.py` (Groq SDK + Streaming)
- **UI Framework**: `app.py` (Streamlit with custom CSS injections)
