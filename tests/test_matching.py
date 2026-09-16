"""Tests for research-area name matching.

Plain-dict logic with no DSPy involvement, so this module deliberately imports
nothing from `dspy_modules` and runs in a core-only install without the
`routing` extra. It lived in test_router.py until match_area moved out of the
dspy-importing module (#11).
"""

from bioinformatics_prompts.matching import match_area

AREAS = [
    {"research_area": "Genomics", "description": "DNA sequencing and assembly"},
    {"research_area": "Single-Cell Genomics", "description": "Single-cell RNA-seq analysis"},
]


def test_match_area_exact_match():
    assert match_area("Genomics", AREAS) == AREAS[0]


def test_match_area_case_insensitive():
    assert match_area("genomics", AREAS) == AREAS[0]


def test_match_area_substring_fallback():
    # Predicted string contains extra text around a listed area name.
    assert match_area("The best match is Single-Cell Genomics.", AREAS) == AREAS[1]


def test_match_area_no_match_returns_none():
    assert match_area("Astrophysics", AREAS) is None


def test_match_area_prefers_the_longest_name():
    """A shorter area name can be a substring of a longer one."""
    assert match_area("Single-Cell Genomics", AREAS) == AREAS[1]


def test_match_area_empty_prediction_returns_none():
    assert match_area("", AREAS) is None
