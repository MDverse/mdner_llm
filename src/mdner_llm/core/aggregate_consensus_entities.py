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

# Name of the single CSV file gathering the votes of every dataset.
DETAILS_CSV_NAME = "consensus_details.csv"
# Columns of the details CSV, in the order they are written.
DETAILS_FIELDNAMES = [
    "entity_predicted_by_model",
    "category",
    "consensus_score",
    "model_name",
    "temperature",
    "found_by_model",
    "consensus_file_path",
    "input_json_path",
]


def extract_temperature(annotation: dict[str, object]) -> float | None:
    """Return the temperature of one annotation as a single float.

    Returns
    -------
    float | None
        The temperature, 1.0 when it is missing, or None when it is an empty list.
    """
    # Read the raw value, usually stored as a list such as [1.0].
    temperature = annotation.get("temperature")
    # A missing temperature means the provider default, which is 1.0.
    if temperature is None:
        return 1.0
    # An empty list carries no information, so callers ignore it.
    if temperature == []:
        return None
    # Unwrap single-element lists such as [1.0], and accept plain numbers too.
    value = temperature[0] if isinstance(temperature, list) else temperature
    return float(value)


def parse_annotation_file(
    path: Path, logger: "loguru.Logger" = loguru.logger
) -> dict[str, object] | None:
    """Read and validate an annotation JSON file.

    Returns
    -------
    dict[str, object] | None
        Parsed annotation payload with validated 'formatted_response',
        or None if parsing fails.
    """
    # Log which file is being read (debug level).
    logger.debug(f"Reading {path.name}.")
    # Open the file and load its JSON content.
    try:
        with path.open(encoding="utf-8") as file_handler:
            annotation = json.load(file_handler)
    # Unreadable file or invalid JSON: log the error and skip this file.
    except (OSError, json.JSONDecodeError) as error:
        logger.error(f"Cannot read or parse {path.name}: {error}")
        return None
    # Get the entities predicted by the LLM (still a raw dictionary).
    raw_response = annotation.get("formatted_response")
    # A file without predictions cannot be used, so skip it.
    if raw_response is None:
        logger.warning(f"'formatted_response' missing in {path.name}, skipped.")
        return None
    # Replace the raw dictionary by a validated ListOfEntities object.
    try:
        annotation["formatted_response"] = ListOfEntities.model_validate(raw_response)
    # Predictions that do not follow the expected schema are skipped.
    except ValidationError as error:
        logger.warning(f"Cannot parse 'formatted_response' in {path.name}: {error}")
        return None
    # Return the annotation with its validated predictions.
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
    # Number of annotators (one per model and temperature), i.e. the number of votes.
    total_annotations = len(annotations)
    # Keep the model name and temperature of each annotator, in the same order.
    annotator_profiles = [
        {
            "model_name": str(ann.get("model_name", "unknown")),
            "temperature": extract_temperature(ann),
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
    # Loop over the annotators, keeping the position of each one as its identifier.
    for annotator_index, annotation in enumerate(annotations):
        # Entities predicted by this annotator (already validated).
        response_model = annotation["formatted_response"]
        for entity in response_model.entities:
            # An entity is identified by its text and its category.
            entity_key = (entity.text, entity.category)
            # Record that this annotator voted for the entity.
            votes[entity_key].add(annotator_index)
            # Keep the first entity object seen, to dump it later.
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
        # Score = share of annotators that found the entity (between 0 and 1).
        score = len(voter_set) / total_annotations
        consensus[text, category] = {
            "text": text,
            "category": category,
            "score": round(score, 4),
            # One response per annotator, saying whether it found the entity.
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
    # Extract distinct model names, with "/" replaced to keep the name file-safe.
    model_names = sorted(
        {
            str(annotation.get("model_name")).replace("/", "_")
            for annotation in annotations
        }
    )
    # Collect the distinct temperatures (missing means 1.0, empty lists are ignored).
    unique_temperatures = {
        temperature
        for annotation in annotations
        if (temperature := extract_temperature(annotation)) is not None
    }
    # Sort temperatures in ascending order.
    temperatures_sorted = sorted(unique_temperatures)
    # Extract unique provider names from single-element lists.
    unique_providers = set()
    for annotation in annotations:
        provider_list = annotation.get("provider")
        # Keep the provider only if the list is not empty and not None.
        if provider_list and provider_list[0] is not None:
            unique_providers.add(str(provider_list[0]))
    # Sort the unique providers alphabetically.
    providers = sorted(unique_providers)
    # Extract unique tag names.
    tags = sorted(
        {
            str(annotation["tag"])
            for annotation in annotations
            if annotation.get("tag") is not None
        }
    )
    # Build the temperature part of the model name, e.g. "1.0_2.0".
    temperatures_identifier = "_".join(str(temp) for temp in temperatures_sorted)
    # Find the earliest timestamp across all runs.
    timestamps = [
        annotation["timestamp"]
        for annotation in annotations
        if annotation.get("timestamp")
    ]
    earliest_timestamp = min(timestamps)
    # Define the set of metric keys to aggregate and the set of handled keys.
    metric_keys = {
        "inference_time_sec",
        "input_tokens",
        "output_tokens",
        "inference_cost_usd",
    }
    handled_keys = metric_keys | {
        "model_name",
        "timestamp",
        "tag",
        "temperature",
        "provider",
        "formatted_response",
        "normalized_entities",
    }
    # Inherit non-aggregated fields from the first annotation.
    aggregated = {
        key: value for key, value in annotations[0].items() if key not in handled_keys
    }
    # Inject correctly computed sums and parameters.
    aggregated.update(
        {
            # Name of the consensus, e.g. "consensus_google_gemma_qwen_qwen3_t_1.0".
            "model_name": (
                f"consensus_{'_'.join(model_names)}_t_{temperatures_identifier}"
            ),
            "models": model_names,
            "timestamp": earliest_timestamp,
            "tag": tags,
            "temperature": temperatures_sorted,
            "provider": providers,
            # Time, tokens and cost are summed because every run is paid and executed.
            "inference_time_sec": round(
                sum(
                    float(annotation.get("inference_time_sec") or 0.0)
                    for annotation in annotations
                ),
                4,
            ),
            "input_tokens": sum(
                int(annotation.get("input_tokens") or 0) for annotation in annotations
            ),
            "output_tokens": sum(
                int(annotation.get("output_tokens") or 0) for annotation in annotations
            ),
            "inference_cost_usd": round(
                sum(
                    float(annotation.get("inference_cost_usd") or 0.0)
                    for annotation in annotations
                ),
                8,
            ),
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
    # Build the metadata block (model name, temperatures, summed costs...).
    metadata = build_aggregated_metadata(annotations)
    # Collect entities satisfying the voting threshold.
    qualified_entities = []
    for key, entity_detail in consensus.items():
        # Keep the entity if enough annotators found it.
        if float(entity_detail["score"]) >= threshold and key in entity_objects:
            # Convert the Pydantic entity to a dictionary and attach its score.
            dumped_entity = entity_objects[key].model_dump()
            dumped_entity["score"] = entity_detail["score"]
            qualified_entities.append(dumped_entity)
    # Re-validate structure through Pydantic container model.
    validated_response = ListOfEntities.model_validate(
        {"entities": qualified_entities}
    ).model_dump()
    # Ensure scores persist in output (validation may drop unknown fields).
    for entity_item, source_item in zip(
        validated_response["entities"], qualified_entities, strict=True
    ):
        entity_item["score"] = source_item["score"]
    return {**metadata, "formatted_response": validated_response}


def write_json(
    path: Path, data: dict[str, object], logger: "loguru.Logger" = loguru.logger
) -> None:
    """Write data dictionary to a formatted JSON file."""
    try:
        # Open the target file in write mode with UTF-8 encoding.
        with path.open("w", encoding="utf-8") as file_handler:
            # Keep accents readable (ensure_ascii=False) and indent for humans.
            json.dump(data, file_handler, ensure_ascii=False, indent=2)
        logger.success(f"Saved to {path} successfully.")
    # Log the error instead of crashing when the file cannot be written.
    except OSError as error:
        logger.error(f"Failed to write {path}: {error}")


def build_details_rows(
    consensus: dict[tuple[str, str], dict[str, object]],
    consensus_file_path: Path,
    input_json_path: str,
) -> list[dict[str, object]]:
    """Flatten the votes of one dataset into CSV rows (one row per entity and annotator).

    Returns
    -------
    list[dict[str, object]]
        Rows whose keys match DETAILS_FIELDNAMES.
    """
    rows = []
    # One block of rows per candidate entity.
    for detail in consensus.values():
        # One row per annotator (model and temperature) for this entity.
        for response in detail["responses"]:
            rows.append(
                {
                    "entity_predicted_by_model": detail["text"],
                    "category": detail["category"],
                    "consensus_score": detail["score"],
                    "model_name": response["model_name"],
                    "temperature": response["temperature"],
                    "found_by_model": response["found"],
                    # Path of the consensus annotation file written for this dataset.
                    "consensus_file_path": str(consensus_file_path),
                    # Path of the ground truth file the LLMs were run on.
                    "input_json_path": input_json_path,
                }
            )
    return rows


def write_consensus_details_csv(
    path: Path,
    rows: list[dict[str, object]],
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Export the consensus score breakdown of all datasets to a single CSV file."""
    try:
        # newline="" avoids blank lines between rows on Windows.
        with path.open("w", encoding="utf-8", newline="") as file_handler:
            # Write the rows using the fixed column order.
            csv_writer = csv.DictWriter(file_handler, fieldnames=DETAILS_FIELDNAMES)
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        logger.success(f"Saved to {path} successfully.")
    # Log the error instead of crashing when the file cannot be written.
    except OSError as error:
        logger.error(f"Failed to write {path}: {error}")


def aggregate_consensus_entities(
    inferences_dir: Path,
    threshold: float,
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
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
        # Skip files that could not be read or validated.
        if parsed is None:
            continue
        # The ground truth file the LLM was run on identifies the dataset.
        raw_source_path = parsed.get("input_json_path")
        # Use its stem as group key, or the annotation file stem as a fallback.
        group_key = (
            Path(str(raw_source_path)).stem if raw_source_path else json_path.stem
        )
        grouped_annotations[group_key].append(parsed)
    logger.info(f"Identified {len(grouped_annotations)} dataset groups.")
    # Rows of the single details CSV, filled while looping over the datasets.
    all_details_rows = []
    # Compute and persist consensus annotations for each source dataset.
    for source_identifier, annotations in sorted(grouped_annotations.items()):
        logger.info(f"Processing '{source_identifier}' ({len(annotations)} files).")
        # Votes of every annotator for every candidate entity.
        consensus, entity_objects = compute_consensus(annotations)
        # Count candidate entities meeting agreement threshold.
        matching_count = sum(
            1 for detail in consensus.values() if float(detail["score"]) >= threshold
        )
        logger.info(
            f"{len(annotations)} JSON aggregated | "
            f"{matching_count}/{len(consensus)} entities above threshold {threshold}."
        )
        # Build the final JSON entity output.
        output_payload = build_consensus_output(
            annotations, consensus, entity_objects, threshold
        )
        # File name without timestamp, so reruns overwrite the same output files.
        json_target = output_dir / f"{source_identifier}_consensus.json"
        write_json(json_target, output_payload, logger)
        # Path of the ground truth file, shared by all annotations of the group.
        input_json_path = str(annotations[0].get("input_json_path") or "")
        # Add the votes of this dataset to the rows of the single CSV.
        all_details_rows.extend(
            build_details_rows(consensus, json_target, input_json_path)
        )
    # Write the single details CSV once, after all datasets are processed.
    write_consensus_details_csv(output_dir / DETAILS_CSV_NAME, all_details_rows, logger)
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
    # The log file name keeps a timestamp, only the output data files do not.
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
