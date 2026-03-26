"""
Parameterized tests for the validate_annotations utility.

Covers:
- Text mismatch (raw_text vs entity text)
- Span length validation
- Overlap detection
- Invalid boundaries
- Sorting and persistence
"""

import json
from pathlib import Path

import pytest

from mdner_llm.utils.validate_annotations import validate_annotations


def write_json(tmp_path: Path, data: dict) -> Path:
    """Write a JSON file in a temporary directory.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.
    data : dict
        Data to write to the JSON file.

    Returns
    -------
    Path
        Path to the created JSON file.
    """
    file_path = tmp_path / "test.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return file_path


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        # Text mismatch
        (
            {
                "raw_text": "abcdef",
                "entities": [
                    {"text": "xyz", "start": 0, "end": 3, "label": "MOL"},
                ],
            },
            {"text": 1, "span": 0, "overlap": 0, "invalid": 0, "label": 0},
        ),
        # Span mismatch
        (
            {
                "raw_text": "Hello",
                "entities": [
                    {"text": "Hello", "start": 0, "end": 10, "label": "MOL"},
                ],
            },
            {"text": 0, "span": 1, "overlap": 0, "invalid": 0, "label": 0},
        ),
        # Overlap
        (
            {
                "raw_text": "abcdef",
                "entities": [
                    {"text": "abc", "start": 0, "end": 3, "label": "MOL"},
                    {"text": "bcd", "start": 1, "end": 4, "label": "MOL"},
                ],
            },
            {"text": 0, "span": 0, "overlap": 1, "invalid": 0, "label": 0},
        ),
        # Invalid boundaries
        (
            {
                "raw_text": " hello",
                "entities": [
                    {"text": " hello", "start": 0, "end": 6, "label": "MOL"},
                ],
            },
            {"text": 0, "span": 0, "overlap": 0, "invalid": 1, "label": 0},
        ),
        # Wrong labels
        (
            {
                "raw_text": "abcdef",
                "entities": [
                    {"text": "ab", "start": 0, "end": 2, "label": "WRONG"},
                ],
            },
            {"text": 0, "span": 0, "overlap": 0, "invalid": 0, "label": 1},
        ),
        # Clean case
        (
            {
                "raw_text": "abcdef",
                "entities": [
                    {"text": "ab", "start": 0, "end": 2, "label": "MOL"},
                ],
            },
            {"text": 0, "span": 0, "overlap": 0, "invalid": 0, "label": 0},
        ),
    ],
)
def test_validation_counts(tmp_path: Path, data: dict, expected: dict):
    """Test validation counters using parameterized inputs."""
    file_path = write_json(tmp_path, data)

    count_errors = validate_annotations("docs/entities_config.yaml", str(file_path))

    assert count_errors["text_mismatches"] == expected["text"]
    assert count_errors["span_mismatches"] == expected["span"]
    assert count_errors["overlaps"] == expected["overlap"]
    assert count_errors["invalid_boundaries"] == expected["invalid"]
    assert count_errors["unknown_labels"] == expected["label"]


def test_sorting_and_persistence(tmp_path: Path):
    """Ensure entities are sorted and persisted to disk."""
    data = {
        "raw_text": "abcdef",
        "entities": [
            {"text": "cd", "start": 2, "end": 4, "label": "MOL"},
            {"text": "ab", "start": 0, "end": 2, "label": "MOL"},
        ],
    }

    file_path = write_json(tmp_path, data)

    validate_annotations("docs/entities_config.yaml", str(file_path))

    with file_path.open("r", encoding="utf-8") as f:
        updated = json.load(f)

    starts = [ent["start"] for ent in updated["entities"]]
    assert starts == sorted(starts)


def test_multiple_issues_same_entity(tmp_path: Path):
    """Test an entity triggering multiple validation errors."""
    data = {
        "raw_text": "abcdef",
        "entities": [
            {
                "text": " xyz",  # invalid boundary + mismatch
                "start": 0,
                "end": 3,
                "label": "MOL",
            },
        ],
    }

    file_path = write_json(tmp_path, data)
    count_errors = validate_annotations("docs/entities_config.yaml", str(file_path))

    assert count_errors["text_mismatches"] == 1
    assert count_errors["invalid_boundaries"] == 1
    assert count_errors["span_mismatches"] == 1
    assert count_errors["overlaps"] == 0
    assert count_errors["unknown_labels"] == 0
