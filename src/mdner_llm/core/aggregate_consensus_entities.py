"""Aggregate entities extracted by multiple LLMs into a consensus list."""

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import click
import loguru
from pydantic import ValidationError as PydanticValidationError

from mdner_llm.common import ensure_dir
from mdner_llm.logger import create_logger
from mdner_llm.models.entities import ListOfEntities


def parse_annotation_file(
    path: Path, logger: "loguru.Logger" = loguru.logger
) -> dict | None:
    """Read and validate a single annotation JSON file, returning None on any error.

    Returns
    -------
    dict | None
        The parsed annotation dict with validated 'formatted_response',
        or None if any error occurs.
    """
    logger.debug(f"Reading {path.name}.")
    # Read the JSON file
    try:
        with path.open(encoding="utf-8") as file_handle:
            annotation = json.load(file_handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.error(f"Cannot read or parse {path.name}: {error}")
        return None
    # Extract the raw 'formatted_response'
    raw_response = annotation.get("formatted_response")
    if raw_response is None:
        logger.warning(f"'formatted_response' missing in {path.name}, skipped.")
        return None
    # Validate and parse 'formatted_response' into ListOfEntities model
    try:
        annotation["formatted_response"] = ListOfEntities.model_validate(raw_response)
    except PydanticValidationError as error:
        logger.warning(f"Cannot parse 'formatted_response' in {path.name}: {error}")
        return None
    return annotation


def compute_consensus(
    annotations: list[dict],
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], object]]:
    """Compute per-entity consensus scores across all annotations.

    Returns
    -------
    tuple[dict[tuple[str, str], dict], dict[tuple[str, str],
    object]]
        A tuple of (consensus dict, entity objects dict), where:
        - consensus dict maps (text, category) to score and responses.
        - entity objects dict maps (text, category) to the original entity object
        from the first annotation that found it.
    """
    total_annotations = len(annotations)
    if total_annotations == 0:
        logger.warning("No annotations to compute consensus on.")
        return {}, {}
    # Extract annotator profiles
    annotators = [
        {
            "model_name": ann.get("model_name", "unknown"),
            "temperature": ann.get("temperature"),
        }
        for ann in annotations
    ]
    # Map entity unique keys to annotator indices and original objects
    votes: dict[tuple[str, str], set[int]] = defaultdict(set)
    entity_objects = {}
    # Iterate over annotations and their entities to tally votes
    # for each unique (text, category)
    for idx, ann in enumerate(annotations):
        # Iterate over entities in this annotation's formatted_response
        for entity in ann["formatted_response"].entities:
            # Use (text, category) as the unique key for consensus voting
            key = (entity.text, entity.category)
            # Add this annotator's vote for the entity
            votes[key].add(idx)
            # Store the original entity object from the first annotation that found it
            entity_objects.setdefault(key, entity)
    # Calculate final agreement ratio and build metadata
    consensus = {}
    for (text, category), voter_set in votes.items():
        score = len(voter_set) / total_annotations
        consensus[text, category] = {
            "text": text,
            "category": category,
            "score": round(score, 4),
            "responses": [
                {
                    "model_name": annotators[idx]["model_name"],
                    "temperature": annotators[idx]["temperature"],
                    "found": idx in voter_set,
                }
                for idx in range(total_annotations)
            ],
        }
    return consensus, entity_objects


def build_aggregated_metadata(annotations: list[dict]) -> dict:
    """Merge metadata across annotations by summing/grouping metrics.

    Returns
    -------
    dict
        A dict with summed performance metrics, formatted consensus name,
        and merged custom metadata attributes.
    """
    # Extract and format model names and temperatures
    models = sorted(
        {str(a.get("model_name", "unknown")).replace("/", "_") for a in annotations}
    )
    temps = sorted({str(a.get("temperature")) for a in annotations})
    # Initialize accumulated metrics and reserved keys
    res = {
        "model_name": f"consensus_{'_'.join(models)}_t_{'_'.join(temps)}",
        "temperature": temps,
        "inference_time_sec": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "inference_cost_usd": 0.0,
    }
    # Process all annotations
    for ann in annotations:
        for key, val in ann.items():
            if key in res and key not in {"model_name", "temperature"}:
                # Sum performance metrics safely
                if isinstance(val, (int, float)):
                    res[key] += val
            elif key not in {"model_name", "temperature", "formatted_response"}:
                # Merge custom metadata keys into single values or deduplicated lists
                if key not in res:
                    res[key] = val
                elif res[key] != val:
                    current = res[key] if isinstance(res[key], list) else [res[key]]
                    if val not in current:
                        current.append(val)
                    res[key] = current
    return res


def build_consensus_output(
    annotations: list[dict],
    consensus: dict[tuple[str, str], dict],
    entity_objects: dict[tuple[str, str], object],
    threshold: float,
) -> dict:
    """Build the aggregated output dict with score embedded in each entity.

    Returns
    -------
    dict
        A dict containing aggregated metadata and a formatted_response with entities
        whose score >= threshold, ready for JSON serialisation.
    """
    meta = build_aggregated_metadata(annotations)
    # Filter entities matching threshold and copy original content
    entities_with_scores = []
    for key, details in consensus.items():
        if details["score"] >= threshold and key in entity_objects:
            dumped = entity_objects[key].model_dump()
            dumped["score"] = details["score"]
            entities_with_scores.append(dumped)
    # Strip score property temporarily to pass baseline Pydantic schema validation
    filtered = ListOfEntities.model_validate(
        {
            "entities": [
                {key: value for key, value in entity.items() if key != "score"}
                for entity in entities_with_scores
            ]
        }
    )
    # Append scores back post-validation
    formatted_response = filtered.model_dump()
    for entity_dict, score_dict in zip(
        formatted_response["entities"], entities_with_scores, strict=False
    ):
        entity_dict["score"] = score_dict["score"]
    return {**meta, "formatted_response": formatted_response}


def write_json(path: Path, data: dict, logger: "loguru.Logger" = loguru.logger) -> None:
    """Save a dict to JSON."""
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.success(f"Saved to {path} successfully.")
    except OSError as exc:
        logger.error(f"Failed to write {path}: {exc}")


def write_consensus_details_csv(
    path: Path,
    consensus: dict[tuple[str, str], dict],
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Save consensus details to CSV, one row per entity/model response."""
    fieldnames = [
        "text",
        "category",
        "consensus_score",
        "model_name",
        "temperature",
        "found",
    ]

    try:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            for detail in consensus.values():
                for response in detail["responses"]:
                    writer.writerow(
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
    except OSError as exc:
        logger.error(f"Failed to write {path}: {exc}")


def aggregate_consensus_entities(
    annotations_dir: Path,
    threshold: float,
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Compute consensus score by source and save as one consensus annotation."""
    # Load all JSON annotation files
    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        logger.error(f"No JSON files found in {annotations_dir}. Exiting.")
        return
    else:
        logger.info(f"Found {len(json_files)} JSON files in {annotations_dir}.")
    # Group files by source name
    groups = defaultdict(list)
    for path in json_files:
        try:
            with path.open(encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            raw = {}
        raw = raw.get("input_json_path")
        if raw:
            groups[Path(raw).stem].append(path)
        else:
            logger.warning(
                f"'input_json_path' missing in {path.name}, falling back to filename."
            )
            groups[path.stem].append(path)
    logger.info(f"Identified {len(groups)} dataset annotations.")

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
    for source, paths in sorted(groups.items()):
        logger.info(f"Processing '{source}' ({len(paths)} files).")
        annotations = []
        # Parse each annotation file, skipping any that fail
        for path in paths:
            ann = parse_annotation_file(path, logger)
            if ann is not None:
                annotations.append(ann)
            logger.debug(f"{len(annotations)}/{len(paths)} parsed ({path.name}).")
        if not annotations:
            logger.warning(f"No valid annotations for '{source}', skipped.")
            continue
        # Compute consensus and filter by threshold
        consensus, entity_objects = compute_consensus(annotations, logger)
        n_above = sum(1 for d in consensus.values() if d["score"] >= threshold)
        logger.info(
            f"{len(annotations)} JSON aggregated | "
            f"{n_above}/{len(consensus)} entities above threshold {threshold}."
        )
        # Save consensus json output (entities + metadata)
        output = build_consensus_output(
            annotations, consensus, entity_objects, threshold
        )
        write_json(output_dir / f"{source}_{timestamp}_consensus.json", output, logger)
        # Save details of consensus scores and responses for all entities
        write_consensus_details_csv(
            output_dir / f"{source}_{timestamp}_consensus_details.csv",
            consensus,
            logger,
        )
    logger.success("Successfully completed consensus aggregation.")


@click.command()
@click.option(
    "--annotations-dir",
    required=True,
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path),
    help="Directory containing the per-run LLM annotation JSON files.",
    callback=ensure_dir,
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
def run_main_from_cli(
    annotations_dir: Path, threshold: float, output_dir: Path
) -> None:
    """CLI entry point for consensus aggregation."""
    logger = create_logger(
        f"logs/aggregate_{datetime.now(UTC).strftime('%Y-%m-%d_%Hh%Mm%Ss')}.log"
    )
    logger.info("Starting consensus aggregation.")
    aggregate_consensus_entities(
        annotations_dir=annotations_dir,
        threshold=threshold,
        output_dir=output_dir,
        logger=logger,
    )


if __name__ == "__main__":
    run_main_from_cli()
