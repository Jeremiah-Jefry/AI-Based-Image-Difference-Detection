from __future__ import annotations

import json
import os
from typing import Any, Generator


# Groq-available models (fast inference)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def groq_available(api_key: str | None = None) -> bool:
    """Check if Groq API key is configured and the SDK is installed."""
    try:
        from groq import Groq  # noqa: F401
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        return bool(key.strip())
    except ImportError:
        return False


def _get_client(api_key: str | None = None):
    """Create a Groq client instance."""
    from groq import Groq
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    return Groq(api_key=key)


_SYSTEM_PROMPT = """\
You are an expert architectural and engineering drawing analyst. You are given \
structured data from an automated comparison between two versions of the same \
architectural document (elevation drawings, floor plans, or similar CAD exports).

Your job is to produce a **rich, user-friendly analysis report** in Markdown format. \
The report must be insightful, professional, and actionable. Write as if you are \
briefing a project manager or architect who needs to understand what changed between \
the two versions.

Structure your report EXACTLY with these sections:

## Executive Summary
2-3 sentences summarizing the overall comparison result. State whether the documents \
are nearly identical, have minor changes, or have significant modifications. Mention \
the total number of change regions and the percentage of area affected.

## Detailed Findings
For each detected change region, describe:
- **Location**: Which part of the drawing (quadrant) the change is in
- **Type**: Whether elements were added or removed
- **Magnitude**: The area affected and the SSIM similarity score for that region
- **Possible interpretation**: What kind of architectural element might be involved \
(based on location and size — e.g., window modifications, structural changes, \
annotation updates, etc.)

Number each finding (Finding 1, Finding 2, etc.).

## Severity Assessment
Rate the overall severity as one of:
- **None**: No meaningful changes detected
- **Minor**: Small, localized changes that don't affect the overall design intent
- **Moderate**: Notable changes in specific areas that warrant review
- **Major**: Extensive modifications across the drawing

Justify your rating in 1-2 sentences.

## Heatmap Interpretation
Describe what the spatial distribution of changes reveals. Are changes clustered \
in one area or spread across the sheet? What does this pattern suggest about the \
nature of the revision?

## Recommendations
Provide 2-4 actionable recommendations based on the findings. Examples:
- Areas that need closer manual review
- Whether a design review meeting is warranted
- Specific regions to verify in the field
- Whether the changes appear to be cosmetic vs. structural

Keep the tone professional but accessible. Use bullet points and bold text for \
emphasis. Do NOT use technical jargon about the detection algorithm — speak in \
terms the end user (architect/engineer/PM) would understand.\
"""


def _build_user_prompt(stats: dict[str, Any]) -> str:
    """Build the user prompt from computed diff statistics."""
    regions = stats.get("regions", [])
    region_details = []
    for i, region in enumerate(regions, 1):
        region_details.append({
            "region_number": i,
            "location_quadrant": region.get("quadrant", "unknown"),
            "label": region.get("label", "unknown"),
            "bounding_box": region.get("bbox_xywh", region.get("bbox_xyxy", [])),
            "pixel_area": region.get("area", 0),
            "bounding_box_area": region.get("bbox_area", 0),
            "ssim_similarity": region.get("ssim_mean", 1.0),
            "added_pixels": region.get("added_pixels", 0),
            "removed_pixels": region.get("removed_pixels", 0),
        })

    data = {
        "total_change_regions": stats.get("region_count", 0),
        "total_changed_area_pixels": stats.get("total_changed_area", 0),
        "sheet_area_pixels": stats.get("sheet_area", 0),
        "changed_area_percentage": stats.get("changed_area_pct", 0.0),
        "label_counts": stats.get("label_counts", {}),
        "regions": region_details,
    }

    return (
        "Here is the structured comparison data between two versions of an "
        "architectural drawing:\n\n"
        f"```json\n{json.dumps(data, indent=2)}\n```\n\n"
        "Please analyze this data and produce the full report."
    )


def generate_llm_report_stream(
    stats: dict[str, Any],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> Generator[str, None, None]:
    """Stream LLM-generated report tokens via Groq. Yields text chunks."""
    try:
        client = _get_client(api_key)
    except Exception as exc:
        yield _fallback_message(f"Groq SDK error: {exc}")
        return

    user_prompt = _build_user_prompt(stats)

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            temperature=0.4,
            max_tokens=2048,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except Exception as exc:
        yield _fallback_message(f"Groq API error: {exc}")


def generate_llm_report(
    stats: dict[str, Any],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> str:
    """Generate the full LLM report as a single string (non-streaming)."""
    parts: list[str] = []
    for chunk in generate_llm_report_stream(stats, model=model, api_key=api_key):
        parts.append(chunk)
    return "".join(parts)


def _fallback_message(reason: str) -> str:
    return (
        f"> ⚠️ **LLM report unavailable**: {reason}\n\n"
        "The template-based summary is shown below instead. "
        "To enable AI-powered reports, make sure your Groq API key is configured "
        "in the `.env` file or entered in the sidebar."
    )
