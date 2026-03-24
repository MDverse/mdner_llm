"""
Validate and clean JSON annotations for named entities.

This script checks a single JSON annotation file for:
- Overlapping entities (start/end indices)
- Span length consistency (end - start == len(text))
- Sorts annotations in text order
"""

import json
import operator
from datetime import UTC, datetime
from pathlib import Path

import click
import loguru

from mdner_llm.core.logger import create_logger


def has_invalid_boundaries(text: str) -> bool:
    """Check if an entity has invalid leading or trailing characters.

    Parameters
    ----------
    text : str
        The entity text to check.

    Returns
    -------
    bool
        True if the text has invalid leading or trailing characters, False otherwise.
    """
    invalid_chars = {" ", ".", ",", "(", ")", ";"}
    return len(text) > 0 and (text[0] in invalid_chars or text[-1] in invalid_chars)


def validate_annotations(
    json_path: str, logger: "loguru.Logger" = loguru.logger
) -> tuple[int, int, int]:
    """Validate named entity annotations in a JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing annotations.
        The JSON should have a structure like:
        {
            "text": "Some text to annotate.",
            "entities": [
                {"start": 0, "end": 4, "text": "Some", "label": "O"},
                {"start": 5, "end": 9, "text": "text", "label": "O"},
                ...
            ]
        }
    logger : loguru.Logger, optional
        Logger instance for logging warnings and info. If None, a default logger will be used.

    Returns
    -------
    tuple[int, int, int]
        A tuple containing:
        - count_span_mismatches (int): Number of entities with span length mismatches.
        - count_overlaps (int): Number of overlapping entity pairs.
        - count_invalid_boundaries (int): Number of entities with invalid leading/trailing characters.
    """
    # Load JSON data
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Extract entities
    entities = data.get("entities", [])

    count_span_mismatches = 0
    count_invalid_boundaries = 0
    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        text = ent.get("text", "")
        # Check span length consistency
        if end - start != len(text):
            count_span_mismatches += 1
            logger.warning(
                f"Span length mismatch: '{text}' [{start}, {end}] len(text)={len(text)}"
                f"({path})"
            )
        # Check for invalid leading/trailing characters in entity text
        if has_invalid_boundaries(text):
            count_invalid_boundaries += 1
            logger.warning(
                f"Invalid entity boundaries: '{text}' [{start}, {end}]({path})"
            )

    # Check for overlapping entities
    count_overlaps = 0
    sorted_entities = sorted(entities, key=operator.itemgetter("start"))
    for i in range(1, len(sorted_entities)):
        prev = sorted_entities[i - 1]
        curr = sorted_entities[i]
        if curr["start"] < prev["end"]:
            count_overlaps += 1
            logger.warning(
                f"Overlapping entities: '{prev['text']}' "
                f"[{prev['start']},{prev['end']}] "
                f"and '{curr['text']}' [{curr['start']},{curr['end']}]"
                f"({path})"
            )

    # Sort entities in text order (by start index)
    data["entities"] = sorted_entities
    # Save the JSON file with sorted entities
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return count_span_mismatches, count_overlaps, count_invalid_boundaries


def validate_all_annotations_from_dir(
    annotations_dir: str, log_path: str | None = None
):
    """Validate all JSON annotation files in a directory.

    Parameters
    ----------
    annotations_dir : str
        Path to the directory containing JSON annotation files.
    log_path : str, optional
        Path to save the validation log. If None, logs will be printed to console.
    """
    logger = create_logger(log_path)
    logger.info(f"Validating all annotations in directory: {annotations_dir}")
    # Find all JSON files in the directory
    annotation_files = list(Path(annotations_dir).glob("*.json"))
    logger.info(f"Found {len(annotation_files)} JSON files to validate.")
    # Initialize counters
    total_span_mismatches = 0
    total_overlaps = 0
    total_invalid_boundaries = 0

    # Validate each file and accumulate counts
    for json_file in annotation_files:
        span_mismatches, overlaps, invalid_boundaries = validate_annotations(
            str(json_file), logger
        )
        total_span_mismatches += span_mismatches
        total_overlaps += overlaps
        total_invalid_boundaries += invalid_boundaries
    # Log summary
    logger.info("Validation complete.")
    logger.info(f"Total span mismatches: {total_span_mismatches}")
    logger.info(f"Total overlapping entities: {total_overlaps}")
    logger.info(f"Total entities with invalid boundaries: {total_invalid_boundaries}")


@click.command()
@click.option("--json-path", type=click.Path(exists=True))
@click.option("--annotations-dir", type=click.Path(exists=True))
def run_main_from_cli(json_path: str, annotations_dir: str):
    """Run the annotation validation from the command line."""
    if json_path:
        # Initialize logger
        logger = create_logger()
        logger.info(f"Validating annotations in {json_path}...")
        validate_annotations(json_path, logger)
        logger.success(f"Sorted entities saved to {json_path} successfully.")
    if annotations_dir:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        log_path = f"logs/validate_annotations_{timestamp}.log"
        validate_all_annotations_from_dir(annotations_dir, log_path=log_path)


if __name__ == "__main__":
    run_main_from_cli()
