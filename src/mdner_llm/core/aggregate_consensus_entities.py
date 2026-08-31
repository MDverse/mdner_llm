"""Aggregate entities extracted by multiple LLMs into a consensus list."""

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import click
import loguru
from pydantic import BaseModel, ValidationError

from mdner_llm.common import ensure_dir
from mdner_llm.logger import create_logger
from mdner_llm.models.entities import ListOfEntities


def parse_annotation_file(
    path: Path, logger: loguru.Logger = loguru.logger
) -> dict[str, object] | None:
    """Read and validate an annotation JSON file.

    Returns
    -------
    dict[str, object] | None
        Parsed annotation payload with validated 'formatted_response',
        or None if parsing fails.
    """
    logger.debug(f"Reading {path.name}.")
    # Load the JSON content from the file.
    try:
        with path.open(encoding="utf-8") as file_handler:
            annotation = json.load(file_handler)
    except (OSError, json.JSONDecodeError) as error:
        logger.error(f"Cannot read or parse {path.name}: {error}")
        return None
    # Get the predicted entities
    raw_response = annotation.get("formatted_response")
    if raw_response is None:
        logger.warning(f"'formatted_response' missing in {path.name}, skipped.")
        return None
    # Validate it against the ListOfEntities model.
    try:
        annotation["formatted_response"] = ListOfEntities.model_validate(raw_response)
    except ValidationError as error:
        logger.warning(f"Cannot parse 'formatted_response' in {path.name}: {error}")
        return None
    return annotation


def compute_consensus(
    annotations: list[dict[str, object]],
) -> tuple[
    dict[tuple[str, str], dict[str, object]],
    dict[tuple[str, str], BaseModel],
]:
    """Calculate agreement scores for entities across annotators.

    Returns
    -------
    tuple[dict[tuple[str, str], dict[str, object]], dict[tuple[str, str], BaseModel]]
        Mapping of entity keys to scores/responses and original entity models.

    Examples
    --------
    >>> # Scenario: 2 LLMs annotate a molecular dynamics dataset description.
    >>> # LLM 0 finds ("CHARMM36", "FORCE_FIELD") and ("GROMACS", "SOFTWARE").
    >>> # LLM 1 finds ("CHARMM36", "FORCE_FIELD") and ("TIP3P", "WATER_MODEL").
    >>> # Result for ("CHARMM36", "FORCE_FIELD"): score = 2 / 2 = 1.0 (found by both).
    >>> # Result for ("GROMACS", "SOFTWARE"): score = 1 / 2 = 0.5 (found by LLM 0 only).
    """
    total_annotations = len(annotations)
    annotator_profiles = [
        {
            "model_name": str(ann.get("model_name", "unknown")),
            "temperature": ann.get("temperature"),
        }
        for ann in annotations
    ]
    # Map (text, category) to voter indices and original entity instances.
    # Example:
    # votes = {
    #     ("CHARMM36", "FORCE_FIELD"): {0, 1},
    #     ("GROMACS", "SOFTWARE"): {0},
    #     ("TIP3P", "WATER_MODEL"): {1},
    # }
    votes = defaultdict(set)
    entity_objects = {}
    for annotator_index, annotation in enumerate(annotations):
        response_model = annotation["formatted_response"]
        for entity in response_model.entities:
            entity_key = (entity.text, entity.category)
            votes[entity_key].add(annotator_index)
            entity_objects.setdefault(entity_key, entity)

    # Compute individual consensus ratio for each candidate entity.
    # Example output for ("CHARMM36", "FORCE_FIELD"):
    # {
    #     "text": "CHARMM36",
    #     "category": "FORCE_FIELD",
    #     "score": 1.0,
    #     "responses": [
    #         {"model_name": "gpt-4o", "temperature": 0.0, "found": True},
    #         {"model_name": "claude-3-5-sonnet", "temperature": 0.2, "found": True},
    #     ],
    # }
    consensus = {}
    for (text, category), voter_set in votes.items():
        score = len(voter_set) / total_annotations
        consensus[text, category] = {
            "text": text,
            "category": category,
            "score": round(score, 4),
            "responses": [
                {
                    "model_name": annotator_profiles[index]["model_name"],
                    "temperature": annotator_profiles[index]["temperature"],
                    "found": index in voter_set,
                }
                for index in range(total_annotations)
            ],
        }
    return consensus, entity_objects


def build_aggregated_metadata(
    annotations: list[dict[str, object]],
) -> dict[str, object]:
    """Merge and aggregate metadata fields across all run annotations.

    Returns
    -------
    dict[str, object]
        Combined metadata dictionary containing summed metrics and run info.
    """
    # Extract distinct model names.
    model_names = sorted(
        {
            str(annotation.get("model_name")).replace("/", "_")
            for annotation in annotations
        }
    )
    # Collect sorted unique temperatures and providers.
    temperatures = sorted(
        {
            float(annotation["temperature"])
            for annotation in annotations
            if annotation.get("temperature") is not None
        }
    )
    providers = sorted(
        {
            str(annotation["provider"])
            for annotation in annotations
            if annotation.get("provider") is not None
        }
    )
    tags = sorted(
        {
            str(annotation["tag"])
            for annotation in annotations
            if annotation.get("tag") is not None
        }
    )
    temperatures_identifier = "_".join(str(temp) for temp in temperatures)
    # Aggregate run identifiers and sum numerical metrics across annotations.
    aggregated = {
        "model_name": f"consensus_{'_'.join(model_names)}_t_{temperatures_identifier}",
        "tag": tags,
        "temperature": temperatures,
        "provider": providers,
        "inference_time_sec": sum(
            float(annotation.get("inference_time_sec", 0.0))
            for annotation in annotations
        ),
        "input_tokens": sum(
            int(annotation.get("input_tokens", 0)) for annotation in annotations
        ),
        "output_tokens": sum(
            int(annotation.get("output_tokens", 0)) for annotation in annotations
        ),
        "inference_cost_usd": sum(
            float(annotation.get("inference_cost_usd", 0.0))
            for annotation in annotations
        ),
    }
    # Retain remaining custom metadata from the first annotation.
    excluded_keys = set(aggregated.keys()) | {
        "formatted_response",
        "normalized_entities",
    }
    aggregated.update(
        {
            key: value
            for key, value in annotations[0].items()
            if key not in excluded_keys
        }
    )
    return aggregated


def build_consensus_output(
    annotations: list[dict[str, object]],
    consensus: dict[tuple[str, str], dict[str, object]],
    entity_objects: dict[tuple[str, str], BaseModel],
    threshold: float,
) -> dict[str, object]:
    """Filter agreed entities and construct the final JSON-serializable structure.

    Returns
    -------
    dict[str, object]
        Final merged document payload matching the target schema.
    """
    metadata = build_aggregated_metadata(annotations)
    # Collect entities satisfying the voting threshold.
    qualified_entities = []
    for key, entity_detail in consensus.items():
        if float(entity_detail["score"]) >= threshold and key in entity_objects:
            dumped_entity = entity_objects[key].model_dump()
            dumped_entity["score"] = entity_detail["score"]
            qualified_entities.append(dumped_entity)
    # Re-validate structure through Pydantic container model.
    validated_response = ListOfEntities.model_validate(
        {"entities": qualified_entities}
    ).model_dump()
    # Ensure scores persist in output.
    for entity_item, source_item in zip(
        validated_response["entities"], qualified_entities, strict=True
    ):
        entity_item["score"] = source_item["score"]
    return {**metadata, "formatted_response": validated_response}


def write_json(
    path: Path, data: dict[str, object], logger: loguru.Logger = loguru.logger
) -> None:
    """Write data dictionary to a formatted JSON file."""
    try:
        with path.open("w", encoding="utf-8") as file_handler:
            json.dump(data, file_handler, ensure_ascii=False, indent=2)
        logger.success(f"Saved to {path} successfully.")
    except OSError as error:
        logger.error(f"Failed to write {path}: {error}")


def write_consensus_details_csv(
    path: Path,
    consensus: dict[tuple[str, str], dict[str, object]],
    logger: loguru.Logger = loguru.logger,
) -> None:
    """Export consensus score breakdown to CSV format."""
    fieldnames = [
        "text",
        "category",
        "consensus_score",
        "model_name",
        "temperature",
        "found",
    ]
    try:
        with path.open("w", encoding="utf-8", newline="") as file_handler:
            csv_writer = csv.DictWriter(file_handler, fieldnames=fieldnames)
            csv_writer.writeheader()
            # Write each entity's consensus details row by row.
            for detail in consensus.values():
                for response in detail["responses"]:
                    csv_writer.writerow(
                        {
                            "text": detail["text"],
                            "category": detail["category"],
                            "consensus_score": detail["score"],
                            "model_name": response["model_name"],
                            "temperature": response["temperature"],
                            "found": response["found"],
                        }
                    )
        logger.success(f"Saved to {path} successfully.")
    except OSError as error:
        logger.error(f"Failed to write {path}: {error}")


def aggregate_consensus_entities(
    inferences_dir: Path,
    threshold: float,
    output_dir: Path,
    logger: loguru.Logger = loguru.logger,
) -> None:
    """Group, evaluate, and save consensus results for inference collections."""
    # Retrieve all JSON files in the specified directory.
    json_paths = sorted(inferences_dir.glob("*.json"))
    if not json_paths:
        logger.error(f"No JSON files found in {inferences_dir}. Exiting.")
        return
    logger.info(f"Found {len(json_paths)} JSON files in {inferences_dir}.")
    # Group valid annotations by dataset stem name.
    grouped_annotations = defaultdict(list)
    for json_path in json_paths:
        parsed = parse_annotation_file(json_path, logger)
        if parsed is None:
            continue
        raw_source_path = parsed.get("input_json_path")
        group_key = (
            Path(str(raw_source_path)).stem if raw_source_path else json_path.stem
        )
        grouped_annotations[group_key].append(parsed)
    logger.info(f"Identified {len(grouped_annotations)} dataset groups.")
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
    # Compute and persist consensus annotations for each source dataset.
    for source_identifier, annotations in sorted(grouped_annotations.items()):
        logger.info(f"Processing '{source_identifier}' ({len(annotations)} files).")
        consensus, entity_objects = compute_consensus(annotations)
        # Count candidate entities meeting agreement threshold.
        matching_count = sum(
            1 for detail in consensus.values() if float(detail["score"]) >= threshold
        )
        logger.info(
            f"{len(annotations)} JSON aggregated | "
            f"{matching_count}/{len(consensus)} entities above threshold {threshold}."
        )
        # Build and write final JSON entity output.
        output_payload = build_consensus_output(
            annotations, consensus, entity_objects, threshold
        )
        json_target = output_dir / f"{source_identifier}_{timestamp}_consensus.json"
        write_json(json_target, output_payload, logger)
        # Write detailed voter matrix CSV output.
        csv_target = (
            output_dir / f"{source_identifier}_{timestamp}_consensus_details.csv"
        )
        write_consensus_details_csv(csv_target, consensus, logger)
    logger.success("Successfully completed consensus aggregation.")


@click.command()
@click.option(
    "--inferences-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    help="Directory containing the per-run LLM inference JSON files.",
)
@click.option(
    "--threshold",
    default=0.5,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum consensus score [0-1] to include an entity in the output.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path),
    help="Directory where consensus outputs will be written.",
    callback=ensure_dir,
)
def run_main_from_cli(inferences_dir: Path, threshold: float, output_dir: Path) -> None:
    """CLI entry point for consensus aggregation."""
    log_file_path = (
        f"logs/aggregate_{datetime.now(UTC).strftime('%Y-%m-%d_%Hh%Mm%Ss')}.log"
    )
    logger = create_logger(log_file_path)
    logger.info("Starting consensus aggregation.")
    aggregate_consensus_entities(
        inferences_dir=inferences_dir,
        threshold=threshold,
        output_dir=output_dir,
        logger=logger,
    )


if __name__ == "__main__":
    run_main_from_cli()
