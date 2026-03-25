"""Build a vocabulary of named entities from JSON annotation files.

This script scans a directory of JSON annotation files, aggregates named
entities by class, normalizes entity text to lowercase, counts total
occurrences across all files, and generates one vocabulary file per entity class.

Each output file contains:
- A header reporting the number of unique entities for that class.
- One normalized entity per line with its total occurrence count.
"""

import json
from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from mdner_llm.core.logger import create_logger
from mdner_llm.utils.count_entities import list_json_files


def collect_entity_counts(
    annot_folder: Path,
) -> dict[str, dict[str, dict[str, int | list[str]]]]:
    """
    Collect normalized entity counts per class from annotation files.

    Parameters
    ----------
    annot_folder : Path
        Directory containing JSON annotation files.

    Returns
    -------
    dict[str, dict[str, EntityInfo]]
        Nested dictionary mapping class labels to entity occurrence counts.
    """
    logger = create_logger()
    logger.info("Collecting entity counts.")
    entity_counts = {}
    json_files = list_json_files(annot_folder)
    logger.debug(f"Found {len(json_files)} JSON files.")

    if not json_files:
        logger.warning(f"No JSON annotation files found in {annot_folder}")

    for json_file in json_files:
        try:
            with json_file.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse JSON file {json_file.name}: {exc}")
            continue

        for entity in data.get("entities", []):
            label = entity.get("label")
            text = entity.get("text")
            if not label or not text:
                continue

            normalized_text = text.lower()

            if label not in entity_counts:
                entity_counts[label] = {}

            if normalized_text not in entity_counts[label]:
                entity_counts[label][normalized_text] = {"count": 0, "files": []}

            entity_counts[label][normalized_text]["count"] += 1
            entity_counts[label][normalized_text]["files"].append(str(json_file))

    logger.success(
        f"Collected entity counts for {len(entity_counts)} classes: "
        f"{', '.join(sorted(entity_counts.keys()))}!"
    )
    return entity_counts


def write_inventory_files(
    entity_counts: defaultdict[str, defaultdict[str, dict[str, object]]],
    out_folder: Path,
) -> None:
    """
    Write QC inventory files for each entity class.

    Parameters
    ----------
    entity_counts : defaultdict[str, defaultdict[str, int]]
        Aggregated entity counts per class.
    out_folder : Path
        Directory where output files will be written.
    """
    out_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving QC inventory files to {out_folder}.")

    for label, entities in entity_counts.items():
        output_path = out_folder / f"{label}.txt"
        total_occurrences = sum(e["count"] for e in entities.values())

        sorted_entities = sorted(
            entities.items(),
            key=lambda item: (-item[1]["count"], item[0]),
        )

        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("# QC ENTITY INVENTORY\n")
            handle.write(f"# Class: {label}\n")
            handle.write(f"# Unique entities: {len(entities)}\n")
            handle.write(f"# Total occurrences: {total_occurrences}\n")
            handle.write("# ----------------------------------------\n")

            for entity_text, info in sorted_entities:
                files_str = ", ".join(info["files"])
                handle.write(f"{entity_text}\t{info['count']}\t[{files_str}]\n")

        logger.debug(f"Written file {output_path.name}")


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
        Directory where QC inventory files will be saved.

    Returns
    -------
    dict[str, dict[str, dict[str, int | list[str]]]]
        Nested dictionary mapping class labels to entity occurrence counts.
    """
    logger = create_logger()
    logger.info("Starting QC entity inventory.")
    logger.debug(f"Annotation folder: {annot_folder}.")
    logger.debug(f"Output folder: {out_folder}.")

    entity_counts = collect_entity_counts(annot_folder)
    write_inventory_files(entity_counts, out_folder)

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


def process_qc_entity_inventory(
    annot_folder: Path,
    out_folder: Path,
) -> None:
    """
    Run the QC entity inventory process.

    Parameters
    ----------
    annot_folder : Path
        Directory containing annotation JSON files.
    out_folder : Path
        Directory where QC inventory files will be saved.

    Returns
    -------
    dict
        Dictionary mapping each entity class to a pandas DataFrame
        with columns: ["entity", "count", "annotation_paths"].
    """
    entity_counts = main(annot_folder, out_folder)
    return build_entity_inventory(entity_counts)


@click.command()
@click.option(
    "--annot-folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("annotations/v3"),
    show_default=True,
    help="Folder containing JSON annotation files.",
)
@click.option(
    "--out-folder",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results/qc_annotations"),
    show_default=True,
    help="Output folder for QC inventory files.",
)
def run_main_from_cli(
    annot_folder: Path,
    out_folder: Path,
) -> None:
    """
    Run the QC entity inventory process.

    Parameters
    ----------
    annot_folder : Path
        Directory containing annotation JSON files.
    out_folder : Path
        Directory where QC inventory files will be saved.
    """
    main(annot_folder, out_folder)


if __name__ == "__main__":
    run_main_from_cli()
