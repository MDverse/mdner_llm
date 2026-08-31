"""Tests for GLiNER and GLiNER2 entity inference and extraction utilities."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mdner_llm.gliner.extract_entities_with_gliner_all_texts import (
    extract_entities_with_gliner,
    load_metadata,
    load_model,
    load_sample,
    run_gliner,
)
from mdner_llm.models.entities import ListOfEntities


@pytest.fixture
def mock_logger() -> MagicMock:
    """Provide a mock logger to silence logging during unit tests.

    Returns
    -------
        MagicMock: A mock logger instance.
    """
    return MagicMock()


@pytest.fixture
def sample_metadata_file(tmp_path: Path) -> Path:
    """Create a temporary TSV file mapping JSON file paths to source URLs.

    Returns
    -------
        Path: The path to the created temporary TSV file.
    """
    metadata_content = (
        "/data/annotations/sample_01.json\thttps://example.org/entry/1\n"
        "/data/annotations/sample_02.json\thttps://example.org/entry/2\n"
    )
    metadata_file = tmp_path / "metadata.tsv"
    metadata_file.write_text(metadata_content, encoding="utf-8")
    return metadata_file


@pytest.fixture
def sample_jsonl_file(tmp_path: Path) -> Path:
    """Create a temporary JSONL dataset containing input texts and entity ground truth.

    Returns
    -------
        Path: The path to the created temporary JSONL file.
    """
    first_row = {
        "input": "Simulation of lipid bilayer using CHARMM36 force field.",
        "output": {
            "entities": {"FFM": ["CHARMM36"]},
            "entity_descriptions": {"FFM": "Force field or water model."},
        },
    }
    second_row = {
        "input": "Water model TIP3P equilibrated in GROMACS.",
        "output": {
            "entities": {"FFM": ["TIP3P"], "SOFTNAME": ["GROMACS"]},
            "entity_descriptions": {
                "FFM": "Force field or water model.",
                "SOFTNAME": "MD simulation or analysis software.",
            },
        },
    }
    jsonl_file = tmp_path / "dataset.jsonl"
    jsonl_file.write_text(
        f"{json.dumps(first_row)}\n{json.dumps(second_row)}\n",
        encoding="utf-8",
    )
    return jsonl_file


def test_load_metadata(sample_metadata_file: Path) -> None:
    """Verify parsing of tab-separated metadata rows into path and URL tuples."""
    entries = load_metadata(sample_metadata_file)

    assert len(entries) == 2
    assert entries[0] == (
        Path("/data/annotations/sample_01.json"),
        "https://example.org/entry/1",
    )
    assert entries[1] == (
        Path("/data/annotations/sample_02.json"),
        "https://example.org/entry/2",
    )


def test_load_metadata_empty_and_corrupt_lines(tmp_path: Path) -> None:
    """Verify that empty lines are skipped and malformed lines raise ValueError."""
    corrupted_file = tmp_path / "corrupted_metadata.tsv"
    corrupted_file.write_text(
        "\n/valid/path.json\thttps://example.org\n\ninvalid_line_without_tab\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match=r"not enough values to unpack \(expected 2, got 1\)"
    ):
        load_metadata(corrupted_file)


def test_load_sample(
    sample_jsonl_file: Path,
    sample_metadata_file: Path,
    mock_logger: MagicMock,
) -> None:
    """Verify loading and alignment of JSONL records with corresponding TSV metadata."""
    samples = load_sample(sample_jsonl_file, sample_metadata_file, logger=mock_logger)

    assert len(samples) == 2
    first_text, first_desc, first_groundtruth, first_path, first_url = samples[0]

    assert first_text == "Simulation of lipid bilayer using CHARMM36 force field."
    assert first_desc == {"FFM": "Force field or water model."}
    assert isinstance(first_groundtruth, ListOfEntities)
    assert len(first_groundtruth.entities) == 1
    assert first_groundtruth.entities[0].category == "FFM"
    assert first_groundtruth.entities[0].text == "CHARMM36"
    assert first_path == Path("/data/annotations/sample_01.json")
    assert first_url == "https://example.org/entry/1"


@patch(
    "mdner_llm.gliner.extract_entities_with_gliner_all_texts.AutoExtractor.from_pretrained"
)
def test_load_model_gliner2_with_adapter(
    mock_auto_extractor: MagicMock, mock_logger: MagicMock
) -> None:
    """Verify that GLiNER2 models load and attach adapters when requested."""
    mock_model = MagicMock()
    mock_auto_extractor.return_value = mock_model

    loaded_model = load_model(
        model_path="fastgliner/gliner2-base",
        adapter_path="/weights/lora_adapter",
        logger=mock_logger,
    )

    assert loaded_model == mock_model
    mock_auto_extractor.assert_called_once_with("fastgliner/gliner2-base")
    mock_model.load_adapter.assert_called_once_with("/weights/lora_adapter")


@patch("mdner_llm.gliner.extract_entities_with_gliner_all_texts.GLiNER.from_pretrained")
@patch(
    "mdner_llm.gliner.extract_entities_with_gliner_all_texts.AutoExtractor.from_pretrained",
    side_effect=OSError("Not a GLiNER2 model"),
)
def test_load_model_fallback_to_gliner_v1(
    mock_auto_extractor: MagicMock,
    mock_gliner_v1: MagicMock,
    mock_logger: MagicMock,
) -> None:
    """Verify graceful fallback to GLiNER v1 when GLiNER2 loader raises OSError."""
    mock_model = MagicMock(spec=[])
    mock_gliner_v1.return_value = mock_model

    loaded_model = load_model(
        model_path="urchade/gliner_base",
        adapter_path=None,
        logger=mock_logger,
    )

    assert loaded_model == mock_model
    mock_auto_extractor.assert_called_once()
    mock_gliner_v1.assert_called_once_with("urchade/gliner_base")


def test_run_gliner_v2_api() -> None:
    """Verify extraction workflow using GLiNER v<2 extract_entities method."""
    mock_model = MagicMock()
    mock_model.extract_entities.return_value = {
        "entities": {"FFM": [{"text": "AMBER99SB", "confidence": 0.95}]}
    }
    text = "Equilibrated with AMBER99SB."
    entity_descriptions = {"FFM": "Force field or water model."}
    predictions, elapsed_time, in_tokens, out_tokens = run_gliner(
        mock_model, text, entity_descriptions
    )
    mock_model.extract_entities.assert_called_once_with(
        text, entity_descriptions, include_confidence=True
    )
    assert "entities" in predictions
    # Check that token counts are positive.
    assert in_tokens > 0
    assert out_tokens > 0
    assert elapsed_time >= 0.0


def test_run_gliner_v1_fallback_api() -> None:
    """Verify legacy GLiNER v1 entity extraction using predict_entities."""
    mock_model = MagicMock(spec=["predict_entities"])
    mock_model.predict_entities.return_value = [
        {"label": "SOFTNAME", "text": "NAMD", "score": 0.88}
    ]
    text = "Simulated on NAMD cluster."
    entity_descriptions = {"SOFTNAME": "MD simulation or analysis software."}
    predictions, elapsed_time, in_tokens, out_tokens = run_gliner(
        mock_model, text, entity_descriptions
    )
    # Different assertion for GLiNER v1
    # it uses predict_entities instead of extract_entities.
    mock_model.predict_entities.assert_called_once_with(
        text, ["SOFTNAME"], threshold=0.5
    )
    assert predictions == {
        "entities": {"SOFTNAME": [{"text": "NAMD", "confidence": 0.88}]}
    }
    # Check that token counts are positive.
    assert in_tokens > 0
    assert out_tokens > 0
    assert elapsed_time >= 0.0


def test_extract_entities_with_gliner_writes_json(
    tmp_path: Path, mock_logger: MagicMock
) -> None:
    """Verify end-to-end single document inference serialization."""
    mock_model = MagicMock()
    mock_model.extract_entities.return_value = {
        "entities": {"FFM": [{"text": "CHARMM36m", "confidence": 0.99}]}
    }
    groundtruth_model = ListOfEntities.model_validate(
        {"entities": [{"category": "FFM", "text": "CHARMM36m"}]}
    )
    source_path = Path("/annotations/run_01.json")
    extract_entities_with_gliner(
        model=mock_model,
        model_name_id="gliner2-md",
        text="Simulation with CHARMM36m.",
        entity_desc={"FFM": "Force field or water model."},
        groundtruth=groundtruth_model,
        text_path=source_path,
        url="https://example.org/run/1",
        output_dir=tmp_path,
        logger=mock_logger,
    )
    output_files = list(tmp_path.glob("*.json"))
    assert len(output_files) == 1
    payload = json.loads(output_files[0].read_text(encoding="utf-8"))
    assert payload["model_name"] == "gliner2-md"
    assert payload["framework_name"] == "noframework"
    assert payload["status"] == "ok"
    # Validate formatted entities through ListOfEntities to avoid schema mismatch.
    extracted_model = ListOfEntities.model_validate(payload["formatted_response"])
    assert len(extracted_model.entities) == 1
    assert extracted_model.entities[0].category == "FFM"
    assert extracted_model.entities[0].text == "CHARMM36m"
