"""Tests for mdner_llm.utils.correct_annotations."""

import json
from pathlib import Path

from mdner_llm.utils.correct_annotations import (
    clean_annotation_file_temperatures,
    clean_trailing_dot,
    split_ffm_entities,
)


def test_clean_trailing_dot() -> None:
    """Ensure trailing dots not followed by digits are removed."""
    assert clean_trailing_dot("310K.") == "310K"
    assert clean_trailing_dot("340 k.") == "340 k"
    assert clean_trailing_dot("293.15") == "293.15"
    assert clean_trailing_dot("value 42.") == "value 42"


def test_clean_annotation_file_temperatures(tmp_path: Path) -> None:
    """Ensure TEMP entities are cleaned and end index updated."""
    file_path = tmp_path / "test.json"

    data = {
        "raw_text": "Simulation at 310K.",
        "entities": [
            {"label": "TEMP", "text": "310K.", "start": 14, "end": 19},
        ],
    }

    file_path.write_text(json.dumps(data), encoding="utf-8")

    clean_annotation_file_temperatures(file_path)

    updated = json.loads(file_path.read_text(encoding="utf-8"))

    assert updated["entities"][0]["text"] == "310K"
    assert updated["entities"][0]["end"] == 18


def test_split_ffm_entities(tmp_path: Path) -> None:
    """Ensure FFMNAME is split into FFMNAME + FFMVERS correctly."""
    annotation_file = tmp_path / "ann.json"
    md_file = tmp_path / "ffm.md"

    md_file.write_text(
        """
        # Controlled list of force field or model for annotation (FFMNAME)

        SPC/E
        SPC
        CHARMM
        """,
        encoding="utf-8",
    )

    data = {
        "raw_text": "We used CHARMM36 and SPC/E water model.",
        "entities": [
            {"label": "FFMNAME", "text": "CHARMM36", "start": 8, "end": 17},
            {"label": "FFMNAME", "text": "SPC/E", "start": 22, "end": 27},
        ],
    }

    annotation_file.write_text(json.dumps(data), encoding="utf-8")

    split_ffm_entities(annotation_file, md_file)

    updated = json.loads(annotation_file.read_text(encoding="utf-8"))

    labels = [e["label"] for e in updated["entities"]]
    texts = [e["text"] for e in updated["entities"]]

    assert "FFMVERS" in labels
    assert "36" in texts
    assert "SPC/E" in texts
