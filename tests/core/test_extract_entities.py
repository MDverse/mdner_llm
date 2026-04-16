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
from types import SimpleNamespace
from typing import Any

import pytest

from mdner_llm.core.extract_entities import (
    extract_content,
    normalize_llm_output,
    save_json_output,
    save_txt_output,
)
from mdner_llm.models.entities import ListOfEntities
from mdner_llm.utils.common import serialize_response

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
# LLM output normalization
# ---------------------------------------------------------------------------


def test_extract_content_valid() -> None:
    """Ensure extract_content retrieves message content correctly."""
    fake = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))]
    )
    result = extract_content(fake)
    assert result == "hello world"


def test_extract_content_missing_choices() -> None:
    """Ensure extract_content raises when choices are missing."""
    fake = SimpleNamespace(choices=[])

    with pytest.raises(ValueError, match="ChatCompletion has no choices"):
        extract_content(fake)


def test_normalize_chatcompletion() -> None:
    """Ensure ChatCompletion-like objects are normalized."""
    fake = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"entities": []}'))]
    )

    result = normalize_llm_output(fake)
    assert result == '{"entities": []}'


def test_normalize_passthrough() -> None:
    """Ensure non-ChatCompletion objects are returned unchanged."""
    data = {"a": 1}

    result = normalize_llm_output(data)
    assert result == data


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------


def test_save_and_load_json(tmp_path: Path) -> None:
    """Ensure JSON output is saved and reloadable."""
    output_path = tmp_path / "output.json"

    data = {
        "text": "example",
        "llm_response": serialize_response({"entities": []}),
    }

    save_json_output(output_path, data)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == data


def test_save_txt_output(tmp_path: Path) -> None:
    """Ensure text output is saved correctly."""
    output_path = tmp_path / "output.txt"
    content = "raw model response"

    save_txt_output(output_path, content)
    loaded = output_path.read_text(encoding="utf-8")
    assert loaded == content
