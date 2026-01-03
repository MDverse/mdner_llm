"""Quality control inventory of named entities from annotation files.

This script scans a directory of JSON annotation files, aggregates named
entities by class, normalizes entity text to lowercase, counts global
occurrences across all annotations, and produces one vocabulary file
per entity class.

Each output file contains:
- A header reporting the number of unique entities for the class
- One normalized entity per line with its total occurrence count

Usage
-----
uv run src/qc_entity_inventory.py --annot-folder PATH --out-folder PATH

Arguments
---------
--annot-folder : str
    Directory containing JSON annotation files.
--out-folder : str
    Output directory where QC inventory files will be written.

Examples
--------
uv run src/qc_entity_inventory.py \
    --annot-folder annotations/v2 \
    --out-folder results/qc_annotations
"""

import json
from collections import defaultdict
from pathlib import Path

import click
from loguru import logger

EntityInfo = dict[str, int | list[str]]


def collect_entity_counts(
    annot_folder: Path,
) -> dict[str, dict[str, EntityInfo]]:
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
    logger.info("Collecting entity count...")
    entity_counts: dict[str, dict[str, EntityInfo]] = {}
    json_files = list(annot_folder.glob("*.json"))
    logger.debug(f"Found {len(json_files)} JSON files")

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

            entity_counts[label][normalized_text]["count"] += 1  # ty:ignore[unsupported-operator]
            entity_counts[label][normalized_text]["files"].append(str(json_file))  # ty:ignore[possibly-missing-attribute]

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
    logger.info(f"Saving QC inventory files to {out_folder}...")

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


@click.command()
@click.option(
    "--annot-folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("annotations/v2"),
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
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR"],
        case_sensitive=False,
    ),
    default="INFO",
    show_default=True,
    help="Logging verbosity level.",
)
def main(
    annot_folder: Path,
    out_folder: Path,
    log_level: str,
) -> None:
    """
    Run the QC entity inventory process.

    Parameters
    ----------
    annot_folder : Path
        Directory containing annotation JSON files.
    out_folder : Path
        Directory where QC inventory files will be saved.
    log_level : str
        Logging verbosity level.
    """
    logger.info("Starting QC entity inventory...")
    logger.debug(f"Annotation folder: {annot_folder}")
    logger.debug(f"Output folder: {out_folder} \n")

    entity_counts = collect_entity_counts(annot_folder)
    write_inventory_files(entity_counts, out_folder)

    logger.success("QC entity inventory completed successfully!")


if __name__ == "__main__":
    main()
