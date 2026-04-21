"""
Test suite for entity extraction pipeline.

This module validates:
- Serialization / deserialization round-trips
- Pydantic model reconstruction
- Normalization of LLM outputs
- File persistence (JSON + TXT)
- Mocked ChatCompletion handling
"""

import json
from pathlib import Path
from typing import Any

import pytest

from mdner_llm.common import serialize_response
from mdner_llm.core.extract_entities_with_llm import (
    save_formated_response_with_metadata_to_json,
)
from mdner_llm.models.entities import ListOfEntities

# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1},
        {"entities": []},
        {"nested": {"x": [1, 2, 3]}},
    ],
)
def test_roundtrip_generic(data: dict[str, Any]) -> None:
    """Ensure generic JSON-serializable objects survive roundtrip."""
    serialized = serialize_response(data)
    deserialized = json.loads(serialized)

    assert deserialized == data


def test_roundtrip_entities() -> None:
    """Ensure ListOfEntities survives serialization roundtrip."""
    obj = ListOfEntities(
        entities=[
            {"category": "MOL", "text": "Cholesterol"},
            {"category": "SOFTNAME", "text": "GROMACS"},
        ]
    )
    serialized = serialize_response(obj)
    reconstructed = ListOfEntities(**json.loads(serialized))
    assert reconstructed == obj


def test_serialize_none() -> None:
    """Ensure None serialization is handled."""
    serialized = serialize_response(None)
    assert serialized is not None


def test_empty_entities_model() -> None:
    """Ensure empty entity list is handled correctly."""
    obj = ListOfEntities(entities=[])

    serialized = serialize_response(obj)
    reconstructed = ListOfEntities(**json.loads(serialized))
    assert reconstructed.entities == []


def test_double_serialization_stability() -> None:
    """Ensure serialization is stable across multiple passes."""
    data = {"entities": [{"category": "X", "text": "Y"}]}

    first = serialize_response(data)
    second = serialize_response(json.loads(first))
    assert json.loads(first) == json.loads(second)


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------


def test_save_and_load_json(tmp_path: Path) -> None:
    """Ensure JSON output is saved and reloadable."""
    output_path = tmp_path / "output.json"

    data = {
        "text": "example",
        "llm_response": {"entities": []},
    }

    save_formated_response_with_metadata_to_json(output_path, data)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == data
