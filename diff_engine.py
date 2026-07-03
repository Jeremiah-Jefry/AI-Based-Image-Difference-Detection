from __future__ import annotations

from typing import Any

from utils.align import align_images
from utils.diff import analyze_differences
from utils.pdf_utils import load_document_image
from utils.summary import build_summary


def run_diff_engine(
    before_source: Any,
    after_source: Any,
    *,
    dpi: int = 200,
    llm_model: str = "gemma3",
) -> dict[str, Any]:
    before_image = load_document_image(before_source, dpi=dpi)
    after_image = load_document_image(after_source, dpi=dpi)

    alignment = align_images(before_image, after_image)
    if alignment["warped_image"] is None:
        return {
            "before_image": before_image,
            "after_image": after_image,
            "alignment": alignment,
            "diff": None,
            "stats": None,
            "summary": alignment["message"],
            "llm_model": llm_model,
        }

    diff = analyze_differences(alignment["warped_image"], after_image)
    stats = {
        "region_count": diff["region_count"],
        "total_changed_area": diff["total_changed_area"],
        "changed_area_pct": diff["changed_area_pct"],
        "sheet_area": diff["sheet_area"],
        "regions": diff["regions"],
        "label_counts": diff["label_counts"],
    }
    summary = build_summary(stats)

    return {
        "before_image": before_image,
        "after_image": after_image,
        "alignment": alignment,
        "diff": diff,
        "stats": stats,
        "summary": summary,
        "llm_model": llm_model,
        "ssim_map": diff.get("ssim_map"),
        "ssim_score": diff.get("ssim_score"),
    }
