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

UNWANTED_TEXTS = {
    "water",
    "waters",
    "lipid",
    "lipids",
    "protein",
    "membrane",
}


def remove_unwanted_entities(
    entities: list[dict],
    unwanted_texts: set[str],
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[list[dict], int]:
    """Remove entities whose text matches unwanted terms.

    Matching is case-insensitive.

    Parameters
    ----------
    entities : list[dict]
        List of entity dictionaries.
    unwanted_texts : set[str]
        Set of unwanted entity texts (lowercase).
    logger : loguru.Logger, optional
        Logger instance for logging removed entities.
        If None, a default logger will be used.

    Returns
    -------
    list[dict]
        Filtered list of entities.
    int
        Count of removed entities.
    """
    filtered_entities = []
    count_removed = 0
    for ent in entities:
        text_lower = ent.get("text", "").lower()
        if text_lower in unwanted_texts:
            count_removed += 1
            logger.debug(
                f"Removed entity '{ent.get('text')}' [{ent.get('start')}, "
                f"{ent.get('end')}]"
            )
        else:
            filtered_entities.append(ent)
    return filtered_entities, count_removed


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


def fix_text_mismatch_local(
    ent: dict,
    raw_text: str,
    window: int = 10,
) -> dict | None:
    """Fix entity span using local search around original indices.

    Parameters
    ----------
    ent : dict
        Entity dictionary with 'text', 'start', 'end'.
    raw_text : str
        Full raw text.
    window : int, optional
        Number of characters before and after to include in search window.

    Returns
    -------
    dict | None
        Updated entity if a unique match is found, otherwise None.
    """
    text = ent["text"]
    start = ent["start"]
    end = ent["end"]

    # Define local search window
    left = max(0, start - window)
    right = min(len(raw_text), end + window)
    local_text = raw_text[left:right]

    matches = []
    search_start = 0

    while True:
        idx = local_text.find(text, search_start)
        if idx == -1:
            break
        global_start = left + idx
        global_end = global_start + len(text)
        matches.append((global_start, global_end))
        search_start = idx + 1

    # Unique match → safe correction
    if len(matches) == 1:
        new_start, new_end = matches[0]
        ent["start"] = new_start
        ent["end"] = new_end
        return ent

    # No match or ambiguous → do nothing
    return None


def validate_annotations(
    json_path: str, logger: "loguru.Logger" = loguru.logger
) -> tuple[int, int, int, int, int]:
    """Validate named entity annotations in a JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing annotations.
        The JSON should have a structure like:
        {
            "raw_text": "Some text to annotate.",
            "entities": [
                {"start": 0, "end": 4, "text": "Some", "label": "O"},
                {"start": 5, "end": 9, "text": "text", "label": "O"},
                ...
            ]
        }
    logger : loguru.Logger, optional
        Logger instance for logging warnings and info.
        If None, a default logger will be used.

    Returns
    -------
    tuple[int, int, int, int, int]
        A tuple containing:
        - count_text_mismatches (int): Number of entities with text mismatches.
        - count_span_mismatches (int): Number of entities with span length mismatches.
        - count_overlaps (int): Number of overlapping entity pairs.
        - count_invalid_boundaries (int):
          Number of entities with invalid leading/trailing characters.
        - count_removed (int): Number of entities removed due to unwanted text.
    """
    # Load JSON data
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Extract entities
    entities = data.get("entities", [])
    # Extract raw text
    raw_text = data.get("raw_text", "")
    # Remove unwanted entities
    entities, count_removed = remove_unwanted_entities(entities, UNWANTED_TEXTS, logger)

    count_text_mismatches = 0
    count_span_mismatches = 0
    count_invalid_boundaries = 0
    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        text = ent.get("text", "")
        # Check if the text matches the span in raw_text
        if raw_text[start:end] != text:
            count_text_mismatches += 1
            logger.warning(
                f"Text mismatch: '{text}' != raw_text[{start}:{end}]="
                f"'{raw_text[start:end]}' ({path})"
            )
            # Try to fix it using local search
            fixed = fix_text_mismatch_local(ent, raw_text)
            if fixed:
                logger.info(
                    f"Fixed span: '{text}' [{start},{end}] -> "
                    f"[{fixed['start']},{fixed['end']}] ({path})"
                )
                ent = fixed
            else:
                logger.warning(
                    f"Unresolved mismatch: '{text}' [{start},{end}] "
                    f"vs '{raw_text[start:end]}' ({path})"
                )
        # Check span length consistency
        if end - start != len(text):
            count_span_mismatches += 1
            logger.warning(
                f"Span length mismatch: '{text}' [{start}, {end}] len(text)={len(text)}"
                f" ({path})"
            )
        # Check for invalid leading/trailing characters in entity text
        if has_invalid_boundaries(text):
            count_invalid_boundaries += 1
            logger.warning(
                f"Invalid entity boundaries: '{text}' [{start}, {end}] ({path})"
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
                f" ({path})"
            )

    # Sort entities in text order (by start index)
    data["entities"] = sorted_entities
    # Save the JSON file with sorted entities
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return (
        count_text_mismatches,
        count_span_mismatches,
        count_overlaps,
        count_invalid_boundaries,
        count_removed,
    )


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
    total_text_mismatches = 0
    total_span_mismatches = 0
    total_overlaps = 0
    total_invalid_boundaries = 0
    total_removed = 0

    # Validate each file and accumulate counts
    for json_file in annotation_files:
        (
            text_mismatches,
            span_mismatches,
            overlaps,
            invalid_boundaries,
            count_removed,
        ) = validate_annotations(str(json_file), logger)
        total_text_mismatches += text_mismatches
        total_span_mismatches += span_mismatches
        total_overlaps += overlaps
        total_invalid_boundaries += invalid_boundaries
        total_removed += count_removed
    # Log summary
    logger.info("Validation complete.")
    logger.info(f"Total text mismatches: {total_text_mismatches}")
    logger.info(f"Total span mismatches: {total_span_mismatches}")
    logger.info(f"Total overlapping entities: {total_overlaps}")
    logger.info(f"Total removed entities: {total_removed}")
    logger.info(f"Total entities with invalid boundaries: {total_invalid_boundaries}")


@click.command()
@click.option("--json-path", type=click.Path(exists=True))
@click.option("--annotations-dir", type=click.Path(exists=True))
def run_main_from_cli(json_path: str | None, annotations_dir: str | None):
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
