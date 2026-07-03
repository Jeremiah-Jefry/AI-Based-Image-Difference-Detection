# agent.md — CAD/Image Structural Diff Detector

## What this project is
A Streamlit app that detects and visualizes differences between two versions
of the same document (architectural elevation drawings, exported as PDF or
image). It must NOT rely on pixel-by-pixel subtraction — that approach is
explicitly disqualified because scan skew, resolution mismatch, JPEG/PDF
recompression, and anti-aliasing all register as false "differences" even
when nothing meaningful changed. This has already been proven true on the
actual sample files in `sample_data/`.

## Validated approach (already tested — do not redesign, implement this)
This pipeline was built and run against `sample_data/v1-new.pdf` and
`sample_data/v2-new.pdf` (real architectural elevation sheets — same
drawing, two different export resolutions/compressions) with these
confirmed results:
- ORB + RANSAC homography alignment: 313/407 inlier matches, clean
  near-similarity transform (~0.49x scale) — confirms same content,
  different export scale.
- Edge-domain diff (Canny, not raw pixels) with a small dilation tolerance
  for residual misalignment.
- After noise filtering: 11 coherent regions survived, ~0.43% of total
  sheet area, clustered in dense window/balcony grid lines — the correct
  read on that pair is "structurally near-identical," which is itself a
  valid, honest output, not a pipeline failure.

**Pipeline stages, in order:**
1. **Ingest** — accept PDF or image uploads for "before" and "after".
   Rasterize PDFs with PyMuPDF (`fitz`), not `pdf2image`/poppler, to avoid
   an external binary dependency.
2. **Align** — grayscale both images, detect ORB keypoints, match with
   BFMatcher (Hamming, crossCheck), estimate homography with
   `cv2.findHomography(..., cv2.RANSAC)`, warp image A into image B's
   frame. Log inlier count — this is your alignment-confidence signal,
   surface it in the UI.
3. **Structural diff (the core "AI-based, not pixel-based" requirement)**
   - Primary: Canny edge maps on both aligned images (Gaussian blur first
     to suppress resample/compression noise), then a tolerant set
     difference between edge maps (dilate one edge map by a few px before
     comparing, to absorb sub-pixel jitter).
   - Secondary signal: `skimage.metrics.structural_similarity` (SSIM) —
     cheap, adds a confidence score per region, does not replace the
     primary detector.
   - Optional upgrade path (only if time allows): `lpips` for true
     pretrained deep-feature perceptual distance. Zero-shot, no training,
     lazy-imported so the app still runs without it installed.
4. **Region extraction** — morphological open/close on the diff mask,
   connected components, bounding boxes, merge boxes that are near each
   other, filter by a minimum area (tune against `sample_data` — ~500px²
   was the validated noise floor for this scale).
5. **Classify** each surviving region as `added` (in B, not A) or
   `removed` (in A, not B) by checking presence in each aligned edge map.
6. **Stats** — region count, total changed area, % of total sheet area,
   bounding box coordinates, rough quadrant (e.g. "upper-left").
7. **Summary (FR-6)** — generate a rich, structured natural-language report
   using a local Ollama LLM (e.g. llama3.1). The report includes an executive
   summary, detailed per-region findings, severity assessment, heatmap
   interpretation, and recommendations. Falls back to deterministic Python
   string templates when Ollama is unavailable. **No cloud LLM API calls** —
   Ollama runs entirely locally.
8. **Visualization** — side-by-side view, overlay with color-coded boxes
   (red = removed, green = added), a diff heatmap, and a draggable
   before/after slider via `streamlit-image-comparison`.

## Hard constraints — do not violate
- No pixel-by-pixel / raw subtraction diff as the primary detector.
- No model training or fine-tuning, anywhere.
- No external LLM API calls (Anthropic, OpenAI, or otherwise) for the
  summary or anything else.
- No custom object detector requiring annotated training data (a
  YOLO-style symbol detector was considered and explicitly ruled out for
  this scope — do not reintroduce it).
- Everything must run locally after `pip install -r requirements.txt`,
  with no required network calls at runtime (the optional `lpips` weight
  download happens once at install/first-run, not per-request).

## Tech stack
Streamlit, PyMuPDF, opencv-python, scikit-image, numpy, Pillow,
streamlit-image-comparison. Optional: lpips.

## Repo structure to produce
```
image-diff-detector/
├── agent.md
├── README.md
├── requirements.txt
├── app.py                  # Streamlit UI only — no pipeline logic here
├── diff_engine.py          # thin orchestrator calling utils/*, pure functions
├── utils/
│   ├── pdf_utils.py        # PyMuPDF rasterization
│   ├── align.py            # ORB + homography
│   ├── diff.py             # Canny diff, SSIM, region extraction, classification
│   ├── summary.py          # template-based NLG, stats formatting
│   └── lpips_diff.py       # optional, lazy-imported, must not break app if lpips is missing
├── sample_data/
│   ├── v1-new.pdf
│   └── v2-new.pdf
└── outputs/                 # gitignored — run artifacts land here
```

## Functional requirement → module mapping
| FR | Requirement | Module |
|---|---|---|
| FR-1 | Upload before/after PDF or image | `app.py`, `utils/pdf_utils.py` |
| FR-2 | Alignment/preprocessing | `utils/align.py` |
| FR-3 | Difference detection (non-pixel) | `utils/diff.py` |
| FR-4 | Visualization (boxes, heatmap, slider) | `app.py` |
| FR-5 | Statistics | `utils/diff.py`, `utils/summary.py` |
| FR-6 | AI summary, no external API | `utils/summary.py` |

## Acceptance test
Running `streamlit run app.py`, uploading `sample_data/v1-new.pdf` as
"before" and `sample_data/v2-new.pdf` as "after" must:
1. Complete without errors and without any network call.
2. Report alignment inlier count in the same ballpark as validated above
   (roughly 250–350 inliers — exact count will vary slightly with library
   versions).
3. Show a small number of surviving diff regions (single digits to low
   tens), not hundreds — if you get hundreds, your noise filtering is
   under-tuned, adjust blur/tolerance/min-area, don't change the
   algorithm.
4. Render a summary paragraph that correctly describes "few/minor
   differences, largely structurally identical" for this specific pair —
   if the summary claims major changes, something regressed.

## Out of scope for this pass
Track B (trained symbol/object detector for windows, doors, balconies as
discrete semantic classes) — documented as a valid future direction but
not to be implemented now; do not add a training loop or download a
COCO-pretrained detector and call it done, it won't detect architectural
classes meaningfully without fine-tuning.
