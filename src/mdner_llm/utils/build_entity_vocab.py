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

from mdner_llm.core.logger import create_logger
from mdner_llm.utils.common import list_json_files_from_txt


def collect_entity_counts(
    texts_path: Path,
) -> dict[str, dict[str, dict[str, int | list[str]]]]:
    """
    Collect normalized entity counts per class from annotation files.

    Parameters
    ----------
    texts_path : Path
        Path to a text file containing a list of JSON annotation files.

    Returns
    -------
    dict[str, dict[str, EntityInfo]]
        Nested dictionary mapping class labels to entity occurrence counts.
    """
    logger = create_logger()
    logger.info("Collecting entity counts.")
    entity_counts = {}
    json_files = list_json_files_from_txt(texts_path=texts_path, logger=logger)

    if json_files == []:
        logger.warning(f"No JSON annotation files found in {texts_path}")
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
            # Extract label and text
            label = entity.get("label")
            text = entity.get("text")
            # Skip if either label or text is missing
            if not label or not text:
                continue
            # Normalize entity text to lowercase for consistent counting
            normalized_text = text.lower()
            # Initialize nested dictionaries if not already present
            if label not in entity_counts:
                entity_counts[label] = {}
            # Initialize count and file list for this entity if not already present
            if normalized_text not in entity_counts[label]:
                entity_counts[label][normalized_text] = {"count": 0, "files": []}
            # Increment count and record file name
            entity_counts[label][normalized_text]["count"] += 1
            entity_counts[label][normalized_text]["files"].append(str(json_file.name))

    logger.success(
        f"Collected entity counts for {len(entity_counts)} classes: "
        f"{', '.join(sorted(entity_counts.keys()))}!"
    )
    return entity_counts


def write_inventory_tsv(
    entity_counts: dict[str, dict[str, dict[str, int | list[str]]]],
    out_path: Path,
) -> None:
    """
    Write a single TSV file containing all entity counts.

    Parameters
    ----------
    entity_counts : dict
        Aggregated entity counts per class.
    out_path : Path
        Path where the output TSV file will be written.
    """
    logger.info("Writing global entity TSV file.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, entities in entity_counts.items():
        for entity_text, info in entities.items():
            # Remove duplicates and sort file names
            file_names = sorted({Path(f).name for f in info["files"]})
            rows.append(
                {
                    "class": label,
                    "entity": entity_text,
                    "occurrence": info["count"],
                    "annotation_files": ",".join(file_names),
                }
            )
    # Create the dataframe
    df = pd.DataFrame(rows)
    # Sort by occurrence count in descending order
    if not df.empty:
        df = df.sort_values("occurrence", ascending=False).reset_index(drop=True)
    # Write to TSV
    df.to_csv(out_path, sep="\t", index=False)
    logger.success(f"Saved global entity TSV file to {out_path} successfully!")


def main(
    annot_folder: Path,
    out_folder: Path,
) -> dict[str, dict[str, dict[str, int | list[str]]]]:
    """
    Run the QC entity inventory process.

    Parameters
    ----------
    annot_folder : Path
        Directory containing annotation JSON files.
    out_folder : Path
        Path where the QC inventory TSV file will be saved.

    Returns
    -------
    dict[str, dict[str, dict[str, int | list[str]]]]
        Nested dictionary mapping class labels to entity occurrence counts.
    """
    logger = create_logger()
    logger.info("Starting QC entity inventory.")

    entity_counts = collect_entity_counts(annot_folder)
    write_inventory_tsv(entity_counts, out_folder)

    logger.success("QC entity inventory completed successfully!")
    return entity_counts


def build_entity_inventory(entity_counts):
    """
    Create a structured inventory of entities from the collected counts.

    Parameters
    ----------
    entity_counts : dict
        Dictionary structured as:
        {
            "CLASS": {
                "entity_text": {
                    "count": int,
                    "files": list[str]
                }
            }
        }

    Returns
    -------
    dict
        Dictionary mapping each entity class to a DataFrame
        with columns: ["entity", "count", "annotation_paths"].
    """
    entity_data = {}

    for entity_class, entities in entity_counts.items():
        rows = []
        for entity_text, data in entities.items():
            rows.append(
                {
                    "entity": entity_text,
                    "count": data["count"],
                    # Remove duplicates and sort file paths
                    "annotation_paths": ",".join(sorted(set(data["files"]))),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("count", ascending=False).reset_index(drop=True)
        entity_data[entity_class] = df

    return entity_data


def process_qc_entity_inventory(texts_path: Path, out_path: Path) -> None:
    """
    Run the QC entity inventory process.

    Parameters
    ----------
    texts_path : Path
        Text file containing annotation JSON files.
    out_path : Path
        Path where the QC inventory TSV file will be saved.

    Returns
    -------
    dict
        Dictionary mapping each entity class to a pandas DataFrame
        with columns: ["class", "entity", "count", "annotation_paths"].
    """
    entity_counts = main(texts_path, out_path)
    return build_entity_inventory(entity_counts)


@click.command()
@click.option(
    "--texts-path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    required=True,
    help="Text file containing annotation JSON files.",
)
@click.option(
    "--out-path",
    type=click.Path(file_okay=True, path_type=Path),
    default=Path("results/qc_annotations/entities.tsv"),
    show_default=True,
    help="Output path for the QC inventory TSV file.",
)
def run_main_from_cli(
    texts_path: Path,
    out_path: Path,
) -> None:
    """
    Run the QC entity inventory process.

    Parameters
    ----------
    texts_path : Path
        Text file containing annotation JSON files.
    out_path : Path
        Path where the QC inventory TSV file will be saved.
    """
    main(texts_path, out_path)


if __name__ == "__main__":
    run_main_from_cli()
