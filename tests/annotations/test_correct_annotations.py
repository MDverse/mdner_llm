"""Tests for mdner_llm.utils.correct_annotations."""

import json
from pathlib import Path

import pytest

from mdner_llm.annotations.correct_annotations import (
    clean_annotation_file_temperatures,
    clean_trailing_dot,
)


def test_clean_trailing_dot() -> None:
    """Ensure trailing dots not followed by digits are removed."""
    assert clean_trailing_dot("310K.") == "310K"
    assert clean_trailing_dot("340 k.") == "340 k"
    assert clean_trailing_dot("value 42 c.") == "value 42 c"


def test_dot_followed_by_digit_not_removed() -> None:
    """Ensure dots before digits are preserved."""
    assert clean_trailing_dot("310.5K") == "310.5K"
    assert clean_trailing_dot("273.15") == "273.15"


@pytest.mark.parametrize(
    ("raw_text", "entity", "expected_text", "expected_end"),
    [
        (
            "A 300K.",
            {"category": "TEMP", "text": "300K.", "start": 2, "end": 7},
            "300K",
            6,
        ),
        (
            "310 Kelvin.",
            {"category": "TEMP", "text": "310 Kelvin", "start": 0, "end": 11},
            "310 Kelvin",
            10,
        ),
        (
            "Temperature is 293.15K.",
            {"category": "TEMP", "text": "293.15K.", "start": 15, "end": 24},
            "293.15K",
            22,
        ),
    ],
)
def test_clean_temperature_parametrized(
    tmp_path: Path,
    raw_text: str,
    entity: dict,
    expected_text: str,
    expected_end: int,
) -> None:
    """Ensure TEMP entities are cleaned correctly (parametrized cases)."""
    file_path = tmp_path / "test.json"

    data = {
        "raw_text": raw_text,
        "entities": [entity],
    }

    file_path.write_text(json.dumps(data), encoding="utf-8")

    clean_annotation_file_temperatures(file_path)

    updated = json.loads(file_path.read_text(encoding="utf-8"))
    ent = updated["entities"][0]

    assert ent["text"] == expected_text
    assert ent["end"] == expected_end


def test_clean_annotation_no_change(tmp_path: Path) -> None:
    """Ensure no change when TEMP has no trailing dot."""
    file_path = tmp_path / "test.json"

    data = {
        "raw_text": "Simulation at 310K",
        "entities": [
            {"category": "TEMP", "text": "310K", "start": 14, "end": 18},
        ],
    }

    file_path.write_text(json.dumps(data), encoding="utf-8")

    clean_annotation_file_temperatures(file_path)

    updated = json.loads(file_path.read_text(encoding="utf-8"))
    assert updated == data


def test_misaligned_entity_text(tmp_path: Path) -> None:
    """Ensure function corrects end even if annotation text mismatches raw_text."""
    file_path = tmp_path / "test.json"

    data = {
        "raw_text": "Temp is 300K.",
        "entities": [
            # incorrect annotation span (too long)
            {"category": "TEMP", "text": "300K.", "start": 8, "end": 20},
        ],
    }

    file_path.write_text(json.dumps(data), encoding="utf-8")
    clean_annotation_file_temperatures(file_path)

    updated = json.loads(file_path.read_text(encoding="utf-8"))
    ent = updated["entities"][0]

    assert ent["text"] == "300K"
    assert ent["end"] == ent["start"] + len("300K")
