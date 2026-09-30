"""Validate that model inference JSON outputs do not contain duplicate entity texts."""

import json
import sys
from collections import Counter
from pathlib import Path

import click
import loguru

from mdner_llm.logger import create_logger


def inspect_entity_duplicates(file_path: Path) -> list[dict[str, str | int]]:
    """Scan an inference JSON file and return duplicated entity texts found globally.

    Returns
    -------
    list[dict[str, str | int]]
        A list of dictionaries, each containing the duplicated entity text
        and its frequency.
    """
    with file_path.open("r", encoding="utf-8") as file_stream:
        payload = json.load(file_stream)

    formatted_response = payload.get("formatted_response", {})
    entities = formatted_response.get("entities", [])

    # Extract all non-empty entity texts globally across all categories
    entity_texts = [
        str(entity.get("text", "")).strip() for entity in entities if entity.get("text")
    ]

    frequency_counter = Counter(entity_texts)
    return [
        {"text": text, "count": count}
        for text, count in frequency_counter.items()
        if count > 1
    ]


def validate_directory_inferences(
    inferences_dir: Path, logger: "create_logger" = loguru.logger
) -> bool:
    """Validate all inference JSON files in a directory for entity text uniqueness.

    Returns
    -------
    bool
        True if all files are valid (no duplicates), False otherwise.
    """
    json_paths = sorted(inferences_dir.rglob("*.json"))

    if not json_paths:
        logger.warning(f"No JSON files found in {inferences_dir}.")
        return True

    logger.info(f"Validating entity uniqueness across {len(json_paths)} files...")

    faulty_files_count = 0
    total_duplicate_entries = 0

    for json_file in json_paths:
        try:
            duplicates = inspect_entity_duplicates(json_file)
            if duplicates:
                faulty_files_count += 1
                total_duplicate_entries += len(duplicates)
                logger.warning(
                    f"Found {len(duplicates)} duplicate text(s) in {json_file.name}:"
                )
                for item in duplicates:
                    logger.debug(f"  '{item['text']}' appeared {item['count']} times")
        except json.JSONDecodeError as decode_error:
            logger.error(f"Could not parse JSON file {json_file.name}: {decode_error}")
            faulty_files_count += 1

    if faulty_files_count > 0:
        logger.critical(
            f"Validation failed: {faulty_files_count}/{len(json_paths)} files have"
            f" duplicates({total_duplicate_entries} duplicate groups total)."
        )
        return False

    logger.success(f"All {len(json_paths)} files contain unique entity texts.")
    return True


@click.command()
@click.option(
    "--inferences-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing inference JSON files to inspect.",
)
def run_cli(inferences_dir: Path) -> None:
    """CLI entrypoint to check entity text uniqueness across model predictions."""
    log_path = Path("logs/check_inference_duplicates.log")
    logger = create_logger(log_path, level="INFO")
    is_valid = validate_directory_inferences(
        inferences_dir=inferences_dir, logger=logger
    )
    if not is_valid:
        logger.warning("LLM can produce duplicate entity texts in its predictions.")
        logger.info(f"Please review the logs at {log_path}.")
        sys.exit(1)


if __name__ == "__main__":
    run_cli()
