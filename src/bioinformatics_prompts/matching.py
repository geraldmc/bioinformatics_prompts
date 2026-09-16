"""Matching a predicted research-area name against available templates.

Pure dictionary logic with no DSPy involvement. It lives here rather than in
`dspy_modules/router.py` so that it stays importable in a core-only install,
without the `routing` extra.
"""

from typing import Dict, List, Optional


def match_area(predicted: str, areas: List[Dict]) -> Optional[Dict]:
    """
    Match a predicted research_area string against a list of template dicts.

    Tries, in order:
      1. Exact match, case-insensitive.
      2. Substring match in either direction, case-insensitive.
    Returns the matching dict, or None if nothing matches.
    """
    if not predicted:
        return None

    predicted_lower = predicted.strip().lower()

    for area in areas:
        if area["research_area"].strip().lower() == predicted_lower:
            return area

    substring_matches = [
        area
        for area in areas
        if (area_lower := area["research_area"].strip().lower()) in predicted_lower
        or predicted_lower in area_lower
    ]
    if substring_matches:
        # Prefer the most specific (longest) name, since a shorter area
        # name can itself be a substring of a longer one (e.g. "Genomics"
        # is a substring of "Single-Cell Genomics").
        return max(substring_matches, key=lambda area: len(area["research_area"]))

    return None
