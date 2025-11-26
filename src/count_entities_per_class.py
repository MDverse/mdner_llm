"""Count the number of entities per class for each annotation.

This script processes all JSON annotation files in a specified directory
and counts the number of entities for each class defined in the annotation file.
The script outputs a TSV file containing the filename, annotated text length
and the number of entities per class.

Usage :
=======
    uv run src/count_entities_per_class.py

"""

__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "AGPL-3.0 license"
__date__ = "2025"
__version__ = "1.0.0"


import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ANNOTATION_DIR = Path("annotations/v2")
RESULTS_DIR = Path("results")
CLASSES = ["TEMP", "SOFTNAME", "SOFTVERS", "STIME", "MOL", "FFM"]


def setup_logger(logger) -> None:
    """Update logger configuration."""
    logger.remove()
    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss}"
        "| <level>{level:<8}</level> "
        "| <level>{message}</level>"
    )
    logger.add(
        sys.stdout,
        format=fmt,
        level="DEBUG",
    )


def list_json_files(directory: Path) -> list[Path]:
    """
    Retrieve all JSON files from a given directory.

    Parameters
    ----------
        directory (Path): The path to the directory containing JSON files.

    Returns
    -------
    files: List[Path]
        A list of JSON file paths.
    """
    files = list(directory.rglob("*.json"))
    logger.success(f"Found {len(files)} JSON annotation files.")
    return files


def load_json(json_file_path: Path) -> dict:
    """
    Load a JSON file and return its content as a dictionary.

    Parameters
    ----------
        json_file_path (Path): The full path to the JSON file.

    Returns
    -------
        Dict: Parsed JSON data.
    """
    data = {}
    try:
        with open(json_file_path, encoding="utf-8") as json_file:
            data = json.load(json_file)
    except ValueError as e:
        logger.error(f"Failed to load {json_file_path}: {e}")
    except OSError as e:
        logger.error(f"Cannot open {json_file_path}: {e}")
    return data


def count_entities_per_class(data: dict, classes: list[str]) -> dict:
    """
    Count the number of entities per class in a JSON annotation.

    Parameters
    ----------
        data (Dict): The JSON data loaded from an annotation file.
        classes (List[str]): List of entity classes to count.

    Returns
    -------
        Dict: Updated dictionary with the count of entities per class.
    """
    # Create empty dictionary.
    record = dict.fromkeys(classes, 0)
    # Count entities per class.
    for entity in data["entities"]:
        record[entity["label"]] += 1
    return record


def aggregate(counts_list: list[dict], classes: list[str]) -> pd.DataFrame:
    """
    Aggregate a list of entity count dictionaries into a DataFrame.

    Parameters
    ----------
    counts_list: List[Dict]
        A list of dictionaries with entity counts.
    classes: List[str]
        List of entity classes.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the aggregated counts.
    """
    df = pd.DataFrame(counts_list)
    columns = {}
    for cls in classes:
        columns[cls] = f"NB_{cls}"
    df = df.rename(columns=columns)
    df = df[["filename", "length", *list(columns.values())]]
    return df


def display_stats(df: pd.DataFrame, classes: list[str]) -> None:
    """
    Display statistics of entity counts per class.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame containing the entity counts.
    classes: List[str]
        List of entity classes.
    """
    total_entities = df[[f"NB_{cls}" for cls in classes]].sum()
    logger.info(f"Total number of entities: {total_entities.sum()}")
    for cls in classes:
        class_total = df[f"NB_{cls}"].sum()
        logger.info(f"Number of entities for class '{cls}': {class_total}")


def export_to_tsv(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Export the entity counts into a TSV file.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame containing the aggregated entity counts.
    output_dir: Path
        Directory where the TSV file will be saved.
    """
    # Ensure output directory exists.
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    tsv_file_path = output_dir / "all_annotations_entities_count.tsv"
    try:
        df.to_csv(tsv_file_path, sep="\t", index=False)
    except OSError as e:
        logger.error(f"Failed to write TSV file: {e}")
    else:
        logger.success(f"Count results saved in {tsv_file_path}")


if __name__ == "__main__":
    setup_logger(logger)
    logger.info("Searching for JSON files...")
    json_files = list_json_files(ANNOTATION_DIR)
    all_counts = []
    logger.info("Counting entities...")
    columns = ["filename", "length"] + [f"NB_{cls}" for cls in CLASSES]
    for filepath in json_files:
        json_data = load_json(filepath)
        if not json_data or "raw_text" not in json_data:
            logger.warning(f"File {filepath} is missing 'raw_text' or failed to load. Skipping.")
            continue
        counts = count_entities_per_class(json_data, CLASSES)
        counts["filename"] = filepath.name
        counts["length"] = len(json_data["raw_text"])
        all_counts.append(counts)

    # Aggregate results.
    counts_df = aggregate(all_counts, CLASSES)
    # Display statistics.
    display_stats(counts_df, CLASSES)
    # Export to TSV.
    export_to_tsv(counts_df, RESULTS_DIR)
