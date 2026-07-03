from __future__ import annotations

from collections import Counter
from typing import Any


def _pluralize(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural or f"{singular}s"


def _format_quadrants(regions: list[dict[str, Any]]) -> str:
    if not regions:
        return "the sheet"
    counts = Counter(region["quadrant"] for region in regions)
    ordered = [item[0] for item in counts.most_common()]
    if len(ordered) == 1:
        return f"the {ordered[0]} quadrant"
    if len(ordered) == 2:
        return f"the {ordered[0]} and {ordered[1]} quadrants"
    return f"the {ordered[0]}, {ordered[1]}, and {ordered[2]} quadrants"


def _describe_change_balance(added_count: int, removed_count: int) -> str:
    parts: list[str] = []
    if added_count:
        parts.append(f"{added_count} added {_pluralize(added_count, 'region')}")
    if removed_count:
        parts.append(f"{removed_count} removed {_pluralize(removed_count, 'region')}")
    if not parts:
        return "no surviving change regions"
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} and {parts[1]}"


def build_summary(stats: dict[str, Any]) -> str:
    region_count = int(stats.get("region_count", 0))
    changed_area_pct = float(stats.get("changed_area_pct", 0.0))
    regions = list(stats.get("regions", []))
    labels = Counter(region.get("label", "unknown") for region in regions)
    added_count = labels.get("added", 0)
    removed_count = labels.get("removed", 0)
    quadrants = _format_quadrants(regions)

    if region_count == 0 or changed_area_pct < 0.05:
        return (
            "No significant structural differences were detected after alignment. "
            "The two sheets appear largely identical, with only negligible edge noise."
        )

    if region_count <= 3 and changed_area_pct < 0.5:
        balance = _describe_change_balance(added_count, removed_count)
        return (
            f"A few minor structural differences were detected across {region_count} "
            f"{_pluralize(region_count, 'region')}, concentrated in {quadrants}. "
            f"The surviving changes are mainly {balance}. "
            "Overall, the drawings remain structurally very similar."
        )

    if region_count <= 15 and changed_area_pct < 1.5:
        balance = _describe_change_balance(added_count, removed_count)
        return (
            f"{region_count} localized change {_pluralize(region_count, 'region')} were detected, "
            f"covering about {changed_area_pct:.2f}% of the sheet area and clustering in {quadrants}. "
            f"The balance of changes is {balance}. "
            "The sheets are still largely structurally aligned."
        )

    balance = _describe_change_balance(added_count, removed_count)
    return (
        f"The comparison found {region_count} change {_pluralize(region_count, 'region')} across about {changed_area_pct:.2f}% "
        f"of the sheet area, with activity concentrated in {quadrants}. "
        f"The change balance is {balance}. "
        "The sheets remain related, but the differences are no longer minor."
    )


def build_llm_context(stats: dict[str, Any]) -> dict[str, Any]:
    """Format the stats into a clean payload suitable for LLM prompting."""
    regions = list(stats.get("regions", []))
    return {
        "region_count": int(stats.get("region_count", 0)),
        "changed_area_pct": float(stats.get("changed_area_pct", 0.0)),
        "total_changed_area": int(stats.get("total_changed_area", 0)),
        "sheet_area": int(stats.get("sheet_area", 0)),
        "label_counts": dict(stats.get("label_counts", {})),
        "regions": [
            {
                "quadrant": r.get("quadrant", "unknown"),
                "label": r.get("label", "unknown"),
                "area": r.get("area", 0),
                "bbox_area": r.get("bbox_area", 0),
                "ssim_mean": r.get("ssim_mean", 1.0),
                "added_pixels": r.get("added_pixels", 0),
                "removed_pixels": r.get("removed_pixels", 0),
            }
            for r in regions
        ],
    }
