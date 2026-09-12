"""DSPy-based router that picks a research-area template from a user query."""

from typing import Dict, List, Optional

import dspy


class SelectResearchArea(dspy.Signature):
    """Select the research area template that best matches the user's question."""

    question: str = dspy.InputField(desc="The user's bioinformatics question")
    available_areas: str = dspy.InputField(
        desc="Newline-separated list of '<research_area>: <description>' entries"
    )
    research_area: str = dspy.OutputField(
        desc="The research_area name copied verbatim from available_areas"
    )


class TemplateRouter(dspy.Module):
    """Wraps a dspy.Predict(SelectResearchArea) call."""

    def __init__(self):
        super().__init__()
        self.select = dspy.Predict(SelectResearchArea)

    def forward(self, question: str, areas: List[Dict]) -> dspy.Prediction:
        available_areas = "\n".join(
            f"{area['research_area']}: {area.get('description', '')}" for area in areas
        )
        return self.select(question=question, available_areas=available_areas)


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
