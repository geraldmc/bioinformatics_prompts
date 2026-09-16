"""DSPy-based router that picks a research-area template from a user query.

Importing this module requires the `routing` extra. The DSPy-free half of
routing — matching a predicted name against the available templates — lives in
`bioinformatics_prompts.matching` so it stays reachable without the extra.
"""

from typing import Dict, List

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
