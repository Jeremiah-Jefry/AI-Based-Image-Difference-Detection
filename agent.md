# agent.md — CAD/Drawing Difference Detection App

## What this is
A Streamlit app that compares two versions of a drawing (PDF or image — architectural
elevations, CAD exports, scanned sheets) and tells a human, in plain language, **what
changed, where, and by how much** — without pixel-by-pixel diffing and without training
any model.

This spec is the source of truth. If Copilot is unsure what to build, it should re-read
this file, not invent behavior.

## Validated facts (already tested, not theoretical)
Run against the two real reference files (v1-new.pdf, v2-new.pdf):
- ORB + RANSAC homography alignment: 313/407 inliers, clean ~0.49x scale transform.
  Confirms these are the same sheet at different export resolutions.
- Edge-domain diff (Canny + tolerant edge-set difference) after alignment: 11 surviving
  regions after noise filtering, ~0.43% of sheet area — consistent with "same content,
  different compression/resolution," not a real design change.
- Conclusion baked into the app: the pipeline must be able to say "structurally
  identical" as a valid, confident output — not force-report a change that isn't there.

## Hard constraints
1. **No model training.** Every component is either classical CV or a pretrained model
   used zero-shot.
2. **No cloud/paid API.** No OpenAI/Anthropic/etc. calls. If GenAI text is used, it runs
   locally via Ollama.
3. **Must work fully offline as a fallback.** If Ollama isn't installed/running, the app
   degrades gracefully to template-based text — it must never crash or block on GenAI.
4. **Input:** PDF or image (PNG/JPG), any resolution, any orientation. Handle scale
   mismatch, rotation, and minor skew.
5. **Output must be understandable by a non-technical reviewer in under 30 seconds of
   looking at the screen.** This is a UX requirement, not a nice-to-have.

## Pipeline

### Phase 1 — Alignment + structural diff (core, must ship first)
1. Load both files. If PDF, rasterize each page with PyMuPDF (`fitz`) at a fixed DPI
   (300).
2. Convert to grayscale.
3. ORB keypoint detection + BFMatcher + RANSAC homography → warp image A onto image B's
   frame. Record `inliers`, `total_matches`, `scale_factor` from the homography.
4. Compute a valid-overlap mask (ignore black borders introduced by warping).
5. Canny edge maps on both aligned images (Gaussian blur first to suppress
   compression/resample noise — already tuned; don't re-derive from scratch, reuse the
   tuned kernel/threshold values from the validated run).
6. Tolerant edge-set difference: dilate each edge map by a small kernel (sub-pixel
   alignment slack) before comparing, so near-identical edges don't register as changed.
7. Morphological cleanup → connected components → bounding boxes.
8. Merge nearby boxes (distance threshold) into single regions.
9. Classify each region:
   - **Likely real change**: solid, coherent blob, area above threshold, isolated
     (not inside a dense repeating pattern like a window/railing grid).
   - **Likely noise/artifact**: small, fragmented, clustered inside a high edge-density
     area (repeating pattern signature).
10. Secondary confidence signal: compute SSIM (`skimage.metrics.structural_similarity`)
    over the same aligned pair, patch-wise, to corroborate/contradict the edge-diff
    regions.

### Phase 2 — Measurement + shape description (no training required)
For every region classified as a **likely real change**, compute and report:
- **Bounding box** in pixel coords (tucked into an expandable "technical details," not
  the headline).
- **Size**: width × height in pixels, converted to a physical estimate if the drawing
  has a scale bar or dimension line the user can input (optional field: "1 px = ___ ").
  If no scale is given, report pixel dimensions and percentage of sheet area only —
  never fabricate a physical unit.
- **Shape descriptor**: aspect ratio + contour approximation
  (`cv2.approxPolyDP`) to say whether the region is roughly rectangular, linear
  (a line/wall segment), or irregular. This gives "rectangular element" /
  "thin linear element" without needing a trained classifier.
- **Change type**: by comparing the region's presence/absence/edge-density between
  A and B:
  - present in B, absent in A → **Added**
  - present in A, absent in B → **Removed**
  - present in both, edges differ inside the box → **Modified**
  - same box, different size after alignment → **Resized** (report % change in
    width/height)
- **Location in plain language**: map the bounding box centroid to a 3×3 grid
  ("upper-left", "middle-center", "lower-right", etc.) for the headline view; keep
  exact pixel coords in the technical expander.

### Phase 3 — Local GenAI semantic description (adds "what it is", not just "where")
Goal: turn "rectangular element, 45×60px, upper-right, Modified" into something closer
to "a window opening appears to have been widened" — using a **local** vision-capable
model via **Ollama**, not a cloud API. This keeps the original "no external API"
constraint intact: no key, no cost, nothing leaves the machine.

Implementation:
1. Check if Ollama is installed and a vision model is available
   (e.g. `llava` or `llama3.2-vision`). If not, show a one-time setup hint in the UI
   (`ollama pull llava`) and **fall back to Phase 2's template output** — never block
   the app on this.
2. For each "likely real change" region, crop a padded window (region + ~30% margin)
   from **both** aligned before/after images.
3. Send both crops (not the whole sheet — keeps it fast and focused) to the local model
   with a fixed, constrained prompt, e.g.:
   > "These two image crops show the same location in an architectural drawing, before
   > and after a revision. In one short sentence, describe what changed. If nothing
   > meaningful changed, say so. Do not guess measurements you cannot see."
4. Store the model's sentence alongside the Phase 2 structured data — GenAI output is
   an *additional* caption, never a replacement for the measured facts. If the model's
   description contradicts the measured Added/Removed/Modified/Resized label, show both
   and let the geometric label win in the headline (GenAI text is supplementary, not
   authoritative — it can hallucinate; the geometry can't).
5. Timeout per region (e.g. 15s); on timeout/error, silently fall back to template text
   for that region only, don't fail the whole report.

### Output / UX layer (applies across all phases)
- **Verdict banner at the top**, before any stats table:
  - ✅ "Structurally identical" (no regions survive filtering, or only noise-classified
    regions)
  - ⚠️ "Minor differences detected" (1+ small/medium real-change regions)
  - 🔴 "Significant differences detected" (large area % or multiple large regions)
- **Region cards**, one per detected real change, each containing:
  - Zoomed before/after crop, side by side (this is the single most important UX
    element — a human can *see* the change even if the algorithm can't name it)
  - Change type badge (Added / Removed / Modified / Resized)
  - Plain-language location (3×3 grid) + size in pixels (+ physical unit if scale given)
  - Shape descriptor ("rectangular element", "thin linear element", "irregular region")
  - GenAI caption if available, otherwise omit that line (don't show a placeholder)
- **Noise/artifact regions** shown separately, collapsed by default, labeled clearly as
  "likely compression/resampling artifacts, not design changes" — never mixed into the
  headline count.
- **Stats panel**: total regions by type, % area changed, alignment confidence
  (High/Medium/Low, derived from inlier ratio — not a raw number in the headline view).
- **Legend on every visual** (color = change type).
- Full-resolution side-by-side view + `streamlit-image-comparison` draggable slider for
  the whole sheet, in addition to the per-region cards.

## File structure
```
project/
  app.py                 # Streamlit UI — wires everything below together
  diff_engine.py          # Phase 1: alignment, edge-diff, SSIM, region extraction, classification
  measurement.py          # Phase 2: shape descriptors, change-type logic, plain-language location
  genai_caption.py        # Phase 3: Ollama integration, graceful fallback
  summary.py               # Template-based fallback text generation (always available)
  requirements.txt
  README.md
```

## Libraries (all free/local, no training)
- `pymupdf` (fitz) — PDF → image
- `opencv-python` — ORB, homography, Canny, contours
- `scikit-image` — SSIM
- `numpy`
- `streamlit`, `streamlit-image-comparison`
- `Pillow`
- `ollama` (Python client) — optional, guarded by availability check

## Acceptance test
Given `v1-new.pdf` and `v2-new.pdf` as input, the app must:
1. Report alignment confidence as **High** (inlier ratio ~77%, matching the validated
   313/407 run).
2. Show verdict banner as ✅ or ⚠️ (not 🔴) — matches the validated finding of
   ~0.43% changed area.
3. Separate the ~11 raw candidate regions into noise vs. real-change buckets, with the
   dense window/railing-grid clusters landing in the noise bucket.
4. Run end-to-end with Ollama **not installed** and still produce a complete report
   (Phase 3 silently skipped, Phase 1+2 output intact) — this is the pass/fail check for
   the "never block on GenAI" constraint.
5. Complete in under ~20 seconds per comparison on a normal laptop (no GPU required).

## Professional report requirements (audience: civil engineer / architect, not a developer)
The default report language leaked implementation detail (edge maps, SSIM, inlier
counts, "region," "artifact"). That's wrong for this audience. The report is a **drawing
revision comparison**, and should read like one — closer to how a set of revision
clouds + a revision schedule reads on a real drawing sheet, not like a CV pipeline log.

**Rewrite every user-facing string according to this vocabulary map:**

| Internal term (never shown to user) | Report term (shown to user) |
|---|---|
| Region / bounding box | Change / Item / Callout |
| Added / Removed / Modified / Resized | New element / Removed element / Revised element / Element resized |
| Alignment confidence, inlier ratio | Drawing match quality (or omit entirely if High) |
| Noise / artifact region | Not shown, or (only if asked) "No revision — scan/print variation" |
| Edge-diff, SSIM, ORB, homography, Canny | Never shown anywhere in the report |
| "Structurally identical" | "No revisions detected between the two drawing versions" |
| 3x3 grid label | Keep — "upper-left / middle-center" etc. is already plain language |
| Shape descriptor ("rectangular element") | Reframe using drawing vocabulary where possible: an element that's tall/narrow near a window band → "opening"; wide and horizontal near grade → "sill/base line"; otherwise fall back to a plain phrase ("a rectangular element"), never a technical one |

**Report must read top to bottom as:**
1. **Cover summary** (1–2 lines, plain language): drawing names/versions compared, date
   of comparison, overall verdict ("No revisions detected" / "Minor revisions detected"
   / "Significant revisions detected").
2. **Revision schedule** (the centerpiece — architects expect a table like this on real
   drawings): one row per confirmed change —

   | Item # | Location | Change Type | Description | Approx. Size |
   |---|---|---|---|---|

   Item # is a simple running number (1, 2, 3...) referenced on the drawing image itself
   via matching numbered callout markers — not raw coordinates.
3. **Annotated drawings**: before/after (or overlay) image with numbered callout
   markers matching the schedule's Item #, instead of colored boxes alone. Color coding
   can stay as a secondary aid, but the numbered marker is primary — standard
   architectural drawing convention, instantly familiar to this audience.
4. **Per-item detail cards**: zoomed before/after crop per item, labeled with its Item #,
   plain-language location, size, and the description in one clean sentence — no
   confidence scores, no pixel coordinates inline (those move to the appendix).
5. **Technical appendix (collapsed/optional)**: alignment method, match quality,
   parameters — everything currently in the "Stats panel" moves here, opt-in only.
6. **Limitation note**: rephrase for this audience — e.g. "This report identifies and
   measures visual differences between drawing versions. It does not verify code
   compliance, structural adequacy, or dimensional accuracy against design intent —
   items should be confirmed against the drawing set by a qualified reviewer."

**Units and precision:** default to architectural convention (feet-inches or mm,
matching the drawing's likely convention/title block) wherever a scale is available.
If no scale is given, say so plainly ("size shown in pixels; provide a scale to convert
to real-world units") rather than showing bare pixel counts unexplained.

**Tone check for every sentence in the report:** would a project architect or civil
engineer, with no CV/software background, read this sentence and immediately know what
to do with it? If not, rewrite it.

## Known limitation (be upfront about this)
This system localizes and measures changes and gives a best-effort local-AI caption of
what changed — it does not have a trained architectural-symbol classifier, so it cannot
guarantee correct identification of *what kind* of element changed (window vs. vent vs.
railing) with certainty. That would require the trained-detector path (Track B), which
was explicitly ruled out for this timeline. State this limitation in the README and,
ideally, once in the app UI footer.
