"""Build a vocabulary of named entities from JSON annotation files.

This script scans a directory of JSON annotation files, aggregates named
entities by class, normalizes entity text to lowercase, counts total
occurrences across all files, and generates one vocabulary file per entity class.

Each output file contains:
- A header reporting the number of unique entities for that class.
- One normalized entity per line with its total occurrence count.
"""

import json
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from mdner_llm.utils.common import list_json_files_from_txt
from mdner_llm.utils.logger import create_logger


def collect_entities(
    texts_path: Path,
) -> list[dict]:
    """
    Collect normalized entity counts per class from annotation files.

    Parameters
    ----------
    texts_path : Path
        Path to a text file containing a list of JSON annotation files.

    Returns
    -------
    list[dict]
        List of entities.
    """
    logger = create_logger()
    logger.info("Collecting entities.")
    entities_list = []
    json_files = list_json_files_from_txt(texts_path=texts_path, logger=logger)

    if json_files == []:
        logger.warning(f"No JSON files found in {texts_path}")
    # Handle relative paths if the text file is located in a different directory
    if str(texts_path).startswith("../../"):
        json_files = [Path("../../") / json_file for json_file in json_files]

    for json_file in json_files:
        try:
            with json_file.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse JSON file {json_file.name}: {exc}")
            continue

        for entity in data.get("entities", []):
            # Extract category and text
            category = entity.get("category")
            text = entity.get("text")
            # Skip if either category or text is missing
            if not category or not text:
                continue
            # Create entity dictionnary
            entity_dict = {
                "entity": text.lower(),
                "category": category,
                "json_file": Path(json_file).name,
            }
            entities_list.append(entity_dict)
    logger.success(f"Collected {len(entities_list)} entities")
    return entities_list


def write_inventory(
    entities: list[dict],
    out_path: Path,
) -> None:
    """
    Write a single TSV file containing all entity counts.

    Parameters
    ----------
    entities : list[dict]
        Lit of all entities as dictionnaries.
    out_path : Path
        Path where the output TSV file will be written.
    """
    logger.info("Writing entity inventory TSV file.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Create the dataframe
    df = pd.DataFrame(entities)
    # Write to TSV
    df.to_csv(out_path, sep="\t", index=False)
    logger.success(f"Saved entity inventoryin: {out_path}")


@click.command()
@click.option(
    "--annotation-path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    required=True,
    help="Text file containing the list of JSON files with annotations.",
)
@click.option(
    "--out-path",
    type=click.Path(file_okay=True, path_type=Path),
    required=True,
    help="Path of the TSV file with the entities.",
)
def run_cli(
    annotation_path: Path,
    out_path: Path,
) -> None:
    """
    Run the QC entity inventory process.

    Parameters
    ----------
    annotation_path : Path
        Text file containing the list of JSON files with annotations.
    out_path : Path
        Path of the TSV file with the entities..
    """
    logger = create_logger()
    logger.info("Starting entity inventory.")
    entities = collect_entities(annotation_path)
    write_inventory(entities, out_path)
    logger.success("Entity inventory completed successfully!")


if __name__ == "__main__":
    run_cli()
