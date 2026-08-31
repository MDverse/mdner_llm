"""Tests for GLiNER2 dataset preparation and training utilities."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from gliner2.training.data import InputExample, TrainingDataset
from pydantic import BaseModel

from mdner_llm.gliner.train_gliner import (
    build_example,
    build_train_dataset,
    check_alignment,
    k_fold_split,
    load_config,
    save_dataset_to_jsonl,
    save_loss_points,
)


class MockDataConfig(BaseModel):
    """Mock data configuration for GLiNER training."""

    cv_folds: int = 2
    seed: int = 42
    train_ratio: float = 0.5
    val_ratio: float = 0.5


class MockGLiNERConfig(BaseModel):
    """Mock configuration for GLiNER training, including data settings."""

    data: MockDataConfig = MockDataConfig()


@pytest.fixture
def sample_annotation_file(tmp_path: Path) -> Path:
    """Create a temporary valid JSON annotation file.

    Returns
    -------
        Path: The path to the created JSON file.
    """
    annotation_payload = {
        "raw_text": "Simulations were run using CHARMM36 force field.",
        "url": "https://example.org/dataset/1",
        "entities": [
            {"text": "CHARMM36", "category": "FORCE_FIELD"},
            {"text": "CHARMM36", "category": "FORCE_FIELD"},
        ],
    }
    json_path = tmp_path / "annotation_01.json"
    json_path.write_text(json.dumps(annotation_payload), encoding="utf-8")
    return json_path


def test_load_config_missing_file(tmp_path: Path) -> None:
    """Verify that loading a non-existent configuration file returns None."""
    non_existent_path = tmp_path / "missing_config.yaml"
    loaded_config = load_config(non_existent_path)
    assert loaded_config is None


def test_build_example_deduplication(sample_annotation_file: Path) -> None:
    """Check deduplication of entities and extraction of text and URL."""
    descriptions = {"FORCE_FIELD": "Molecular mechanics force field"}
    example, url = build_example(sample_annotation_file, descriptions)

    assert isinstance(example, InputExample)
    assert example.text == "Simulations were run using CHARMM36 force field."
    assert example.entities == {"FORCE_FIELD": ["CHARMM36"]}
    assert example.entity_descriptions == descriptions
    assert url == "https://example.org/dataset/1"


def test_build_train_dataset(tmp_path: Path, sample_annotation_file: Path) -> None:
    """Ensure training dataset is constructed correctly from a directory of JSONs."""
    dataset, paths, urls = build_train_dataset(tmp_path)

    assert isinstance(dataset, TrainingDataset)
    assert len(dataset) == 1
    assert paths == [sample_annotation_file]
    assert urls == ["https://example.org/dataset/1"]


def test_check_alignment(sample_annotation_file: Path) -> None:
    """Verify alignment detection when raw text matches and when it diverges."""
    mock_logger = MagicMock()
    valid_example = InputExample(
        text="Simulations were run using CHARMM36 force field.",
        entities={},
    )
    valid_dataset = TrainingDataset([valid_example])

    # Case 1: Aligned dataset and annotation file.
    mismatches = check_alignment(
        valid_dataset, [sample_annotation_file], logger=mock_logger
    )
    assert mismatches is None

    # Case 2: Misaligned text.
    invalid_example = InputExample(text="Unrelated text content.", entities={})
    invalid_dataset = TrainingDataset([invalid_example])
    mismatches = check_alignment(
        invalid_dataset, [sample_annotation_file], logger=mock_logger
    )

    assert mismatches is not None
    assert len(mismatches) == 1
    assert mismatches[0]["index"] == 0
    assert mismatches[0]["path"] == sample_annotation_file


def test_save_dataset_to_jsonl(tmp_path: Path) -> None:
    """Test serialization of TrainingDataset to JSONL format."""
    target_path = tmp_path / "output" / "data.jsonl"
    example = InputExample(
        text="GROMACS run.",
        entities={"SOFTWARE": ["GROMACS"]},
        entity_descriptions={"SOFTWARE": "Simulation engine"},
    )
    dataset = TrainingDataset([example])

    save_dataset_to_jsonl(dataset, target_path)

    assert target_path.exists()
    lines = target_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["input"] == "GROMACS run."
    assert record["output"]["entities"]["SOFTWARE"] == ["GROMACS"]


def test_k_fold_split(tmp_path: Path) -> None:
    """Ensure K-fold splitting distributes samples across folds correctly."""
    mock_logger = MagicMock()
    examples = [
        InputExample(text=f"Sample text sequence number {index}", entities={})
        for index in range(4)
    ]
    dataset = TrainingDataset(examples)
    paths = [tmp_path / f"file_{index}.json" for index in range(4)]
    urls = [f"https://example.org/{index}" for index in range(4)]

    # Create dummy files for alignment check.
    for index, current_path in enumerate(paths):
        payload = {"raw_text": f"Sample text sequence number {index}"}
        current_path.write_text(json.dumps(payload), encoding="utf-8")

    mock_config = MockGLiNERConfig()
    folds = k_fold_split(
        dataset=dataset,
        paths=paths,
        urls=urls,
        cfg=mock_config,
        output_dir=tmp_path / "folds_out",
        logger=mock_logger,
    )

    assert len(folds) == 2
    assert (tmp_path / "folds_out" / "fold_1" / "data" / "train.jsonl").exists()
    assert (tmp_path / "folds_out" / "fold_1" / "data" / "val.jsonl").exists()


def test_save_loss_points(tmp_path: Path) -> None:
    """Test serializing train and validation loss points across folds to JSON."""
    results_list = [
        {
            "train_metrics_history": [{"epoch": 1, "loss": 0.45}],
            "eval_metrics_history": [{"epoch": 1, "eval_loss": 0.40}],
        }
    ]
    save_loss_points(results_list, tmp_path)
    output_file = tmp_path / "loss_points.json"

    assert output_file.exists()
    saved_data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(saved_data) == 1
    assert saved_data[0]["fold"] == 1
    assert saved_data[0]["train_loss"] == pytest.approx(0.45)
    assert saved_data[0]["eval_loss"] == pytest.approx(0.40)
