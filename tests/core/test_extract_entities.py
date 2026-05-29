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
    add_guidelines_and_examples_to_prompt,
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


# ---------------------------------------------------------------------------
# Prompt construction normalization
# ---------------------------------------------------------------------------


def test_add_guidelines_and_examples_normalization(tmp_path: Path) -> None:
    """Ensure guideline headers are correctly normalized."""
    # Setup mock prompt template with Jinja2 placeholders
    prompt_template = (
        "System prompt.\n\n## Guidelines\n{{ guidelines }}"
        "\n\n## Examples\n{{ examples }}"
    )
    # Setup guidelines file with a header to strip and headers to shift
    guidelines_content = (
        "This text should be stripped.\n"
        "## Core Rules\n"
        "Some rule here.\n"
        "### Sub Rule\n"
        "Some sub rule here."
    )
    guidelines_file = tmp_path / "guidelines.md"
    guidelines_file.write_text(guidelines_content, encoding="utf-8")
    # Setup examples file
    examples_content = "### Example 1\nInput/Output"
    examples_file = tmp_path / "examples.md"
    examples_file.write_text(examples_content, encoding="utf-8")
    # Execute the prompt construction
    rendered_prompt = add_guidelines_and_examples_to_prompt(
        prompt=prompt_template,
        guidelines_path=guidelines_file,
        examples_path=examples_file,
    )

    # Check that text before the first ## was stripped
    assert "This text should be stripped." not in rendered_prompt
    # Check that headers were shifted (## becomes ###, and ### becomes ####)
    assert "### Core Rules" in rendered_prompt
    assert "#### Sub Rule" in rendered_prompt
    # Check that examples were correctly injected
    assert "### Example 1" in rendered_prompt


def test_add_guidelines_and_examples_with_no_examples(tmp_path: Path) -> None:
    """Ensure that providing examples_path=None renders an empty string for examples."""
    prompt_template = "Guidelines:\n{{ guidelines }}\nExamples:\n{{ examples }}"

    guidelines_file = tmp_path / "guidelines.md"
    guidelines_file.write_text("## Strict Rules", encoding="utf-8")
    rendered_prompt = add_guidelines_and_examples_to_prompt(
        prompt=prompt_template, guidelines_path=guidelines_file, examples_path=None
    )
    assert "### Strict Rules" in rendered_prompt
    # The Examples section in the template should be followed by nothing
    assert "Examples:\n" in rendered_prompt
    assert "{{ examples }}" not in rendered_prompt
