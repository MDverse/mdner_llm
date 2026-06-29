"""Normalize extracted entities from JSON inferences."""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import click
from loguru import logger
from pydantic import ValidationError

from mdner_llm.models.entities import ListOfEntities, ListOfEntitiesNormalized


def load_json_data(file_path: Path) -> dict[str, Any] | None:
    """
    Safely load a JSON file.

    Returns
    -------
        dict[str, Any] | None: The parsed data or None if a IO/JSON error occurs.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, JSONDecodeError) as e:
        logger.error(f"Failed to read or parse JSON file {file_path.name}: {e}")
        return None


def normalize_json_content(data: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extract and normalize entities using Pydantic validation context.

    Returns
    -------
        dict[str, Any] | None: Updated data dictionary or None if validation fails.
    """
    try:
        original_list = ListOfEntities.model_validate(
            data.get("formatted_response", {})
        )
        normalized_list = ListOfEntitiesNormalized.model_validate(
            {"entities": original_list.entities},
            context={
                "source_text": data.get("text", ""),
                "url": data.get("url", "N/A"),
            },
        )
        data["normalized_entities"] = normalized_list.model_dump()
        return data
    except ValidationError as e:
        logger.error(f"Pydantic validation failed: {e}")
        return None


def save_json_data(data: dict[str, Any], output_path: Path) -> bool:
    """
    Safely save the updated dictionary to a destination path.

    Returns
    -------
        bool: True if saving succeeded, False otherwise.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except OSError as e:
        logger.error(f"Failed to write file {output_path.name}: {e}")
        return False


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing the input inference JSON files.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where normalized JSON files will be saved.",
)
def main(input_dir: Path, output_dir: Path) -> None:
    """Load JSON files, normalize their entities, and save the updated data."""
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load all JSON files from the input directory
    json_files = list(input_dir.glob("*.json"))
    total_files = len(json_files)
    if total_files == 0:
        logger.warning(f"No JSON files found in {input_dir}")
        return
    logger.info(f"Found {total_files} JSON file(s) to process from {input_dir}.")

    processed_count = 0
    for file_path in json_files:
        # Load the json file
        data = load_json_data(file_path)
        if data is None:
            continue
        # Normalize the entities in the JSON content
        updated_data = normalize_json_content(data)
        if updated_data is None:
            continue
        # Save the updated data to the output directory
        output_path = output_dir / f"normalized_{file_path.name}"
        if save_json_data(updated_data, output_path):
            processed_count += 1
            # Log progress
            percent_done = (processed_count / total_files) * 100
            logger.info(
                f"Processed {processed_count}/{total_files} files ({percent_done:.1f}%)"
            )

    logger.success("Normalization processing complete.")
    logger.success(f"Total successfully processed: {processed_count}/{total_files}")


if __name__ == "__main__":
    main()
