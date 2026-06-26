"""Unit tests for the consensus aggregation core logic."""

import pytest

from mdner_llm.core.aggregate_consensus_entities import (
    build_aggregated_metadata,
    build_consensus_output,
    compute_consensus,
)
from mdner_llm.models.entities import (
    ForceFieldModel,
    ListOfEntities,
    Molecule,
    SimulationTime,
)


@pytest.fixture
def sample_annotations() -> list[dict]:
    """Fixture providing a base list of model annotations with MD-specific entities.

    Returns
    -------
    list[dict]
    """
    return [
        {
            "model_name": "gpt-4o",
            "temperature": 0.0,
            "inference_time_sec": 1.5,
            "input_tokens": 100,
            "output_tokens": 50,
            "inference_cost_usd": 0.003,
            "timestamp": "2025-06-01T12:00:00Z",
            "formatted_response": ListOfEntities(
                entities=[
                    Molecule(text="POPC", category="MOL"),
                    ForceFieldModel(text="AMBER14SB", category="FFM"),
                    SimulationTime(text="100 ns", category="STIME"),
                ]
            ),
        },
        {
            "model_name": "claude-3-5-sonnet",
            "temperature": 0.0,
            "inference_time_sec": 2.5,
            "input_tokens": 120,
            "output_tokens": 60,
            "inference_cost_usd": 0.005,
            "timestamp": "2025-06-01T12:00:00Z",
            "formatted_response": ListOfEntities(
                entities=[
                    Molecule(text="POPC", category="MOL"),
                    ForceFieldModel(text="AMBER14SB", category="FFM"),
                    # This second model missed the '100 ns' simulation time
                ]
            ),
        },
    ]


def test_compute_consensus_empty_input() -> None:
    """Verify that an empty list of annotations returns empty dicts safely."""
    consensus, entity_objects = compute_consensus([])
    assert consensus == {}
    assert entity_objects == {}


def test_compute_consensus_valid_votes(sample_annotations: list[dict]) -> None:
    """Verify that consensus scores+model voting presence are calculated accurately."""
    consensus, entity_objects = compute_consensus(sample_annotations)

    # Both models extracted the "POPC" lipid molecule -> Score = 1.0
    popc_key = ("POPC", "MOL")
    assert popc_key in consensus
    assert consensus[popc_key]["score"] == pytest.approx(1.0)
    assert consensus[popc_key]["responses"][0]["found"] is True
    assert consensus[popc_key]["responses"][1]["found"] is True

    # Only the first model extracted the simulation time "100 ns" -> Score = 0.5
    time_key = ("100 ns", "STIME")
    assert time_key in consensus
    assert consensus[time_key]["score"] == pytest.approx(0.5)
    assert consensus[time_key]["responses"][0]["found"] is True
    assert consensus[time_key]["responses"][1]["found"] is False

    # Check that entity_objects holds the original structured objects
    assert popc_key in entity_objects
    assert time_key in entity_objects


def test_build_aggregated_metadata_metrics(
    sample_annotations: list[dict],
) -> None:
    """Verify that numerical performance metrics are summed and names are formatted."""
    metadata = build_aggregated_metadata(sample_annotations)

    assert metadata["model_name"] == "consensus_claude-3-5-sonnet_gpt-4o_t_0.0"
    assert metadata["temperature"] == ["0.0"]
    assert metadata["inference_time_sec"] == pytest.approx(4.0)
    assert metadata["input_tokens"] == 220
    assert metadata["output_tokens"] == 110
    assert metadata["inference_cost_usd"] == pytest.approx(0.008)


def test_build_aggregated_metadata_custom_fields(
    sample_annotations: list[dict],
) -> None:
    """Verify custom metadata field merging and deduplication logic."""
    sample_annotations[0]["tag"] = "_run1"
    sample_annotations[1]["tag"] = "_run2"
    metadata = build_aggregated_metadata(sample_annotations)

    assert metadata["timestamp"] == "2025-06-01T12:00:00Z"
    assert sorted(metadata["tag"]) == ["_run1", "_run2"]


def test_build_consensus_output_threshold_filtering(
    sample_annotations: list[dict],
) -> None:
    """Verify that entities falling below the consensus threshold are filtered out."""
    consensus, entity_objects = compute_consensus(sample_annotations)

    # A strict threshold of 0.6 filters out "100 ns" (0.5 agreement)
    output = build_consensus_output(
        sample_annotations, consensus, entity_objects, threshold=0.6
    )
    entities = output["formatted_response"]["entities"]

    # Only "POPC" and "AMBER14SB" should survive
    assert len(entities) == 2
    assert any(entity["text"] == "POPC" for entity in entities)
    assert any(entity["text"] == "AMBER14SB" for entity in entities)
    assert not any(entity["text"] == "100 ns" for entity in entities)


def test_build_consensus_output_inclusive_threshold(
    sample_annotations: list[dict],
) -> None:
    """Verify that MD entities exactly equal to the threshold are preserved."""
    consensus, entity_objects = compute_consensus(sample_annotations)

    # An inclusive threshold of 0.5 retains the partial agreement on "100 ns"
    output = build_consensus_output(
        sample_annotations, consensus, entity_objects, threshold=0.5
    )
    entities = output["formatted_response"]["entities"]

    assert len(entities) == 3
    assert any(
        entity["text"] == "100 ns"
        and entity["category"] == "STIME"
        and entity["score"] == pytest.approx(0.5)
        for entity in entities
    )
