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
import loguru
import pandas as pd

from mdner_llm.logger import create_logger


def collect_entities(
    texts_path: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> pd.DataFrame:
    """
    Collect normalized entity counts per class from annotation files.

    Returns
    -------
    pd.DataFrame
        DataFrame of entities.
    """
    logger.info("Collecting entities.")
    entities_list = []
    # Scan the directory for JSON files.
    json_files = list(texts_path.glob("*.json"))
    logger.success(f"Found {len(json_files)} JSON files successfully.")
    # Handle relative paths.
    if str(texts_path).startswith("../../"):
        json_files = [Path("../../") / json_file for json_file in json_files]
    # Process each JSON file.
    for json_file in json_files:
        try:
            with json_file.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse JSON file {json_file.name}: {exc}")
            continue
        # Extract entities and normalize them.
        for entity in data.get("entities", []):
            # Extract category and text
            category = entity.get("category")
            text = entity.get("text")
            # Create entity dictionnary
            entity_dict = {
                "entity": text,
                "entity_normalized": text.lower().strip(),
                "category": category,
                "json_file": Path(json_file).name,
            }
            entities_list.append(entity_dict)
    logger.success(f"Collected {len(entities_list)} entities.")
    return pd.DataFrame(entities_list)


@click.command()
@click.option(
    "--annotations-path",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    required=True,
    help="Folder containing the list of JSON files with annotations.",
)
@click.option(
    "--out-path",
    type=click.Path(file_okay=True, path_type=Path),
    required=True,
    help="Path of the TSV file with the entities.",
)
def run_cli(
    annotations_path: Path,
    out_path: Path,
) -> None:
    """Run the QC entity inventory process."""
    logger = create_logger()
    logger.info("Starting entity inventory.")
    df_entities = collect_entities(annotations_path, logger=logger)
    # Write to TSV.
    df_entities.to_csv(out_path, sep="\t", index=False)
    logger.success(f"Saved entity inventory in: {out_path}")
    logger.success("Entity inventory completed successfully!")


if __name__ == "__main__":
    run_cli()
