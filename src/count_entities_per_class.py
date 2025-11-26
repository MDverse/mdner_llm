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


import csv
import json
from pathlib import Path
from typing import Dict, List

from loguru import logger

ANNOTATION_DIR = Path("annotations/v2")
RESULTS_DIR = Path("results")
CLASSES = ["TEMP", "SOFTNAME", "SOFTVERS", "STIME", "MOL", "FFM"]


def list_json_files(directory: Path) -> List[Path]:
    """
    Retrieve all JSON files from a given directory.

    Parameters:
    -----------
        directory (str): The path to the directory containing JSON files.

    Returns:
    --------
    files: List[Path]
        A list of JSON file paths.
    """
    files = list(directory.rglob("*.json"))
    logger.info(f"Found {len(files)} JSON annotation files.")
    return files


def load_json(json_file_path: Path) -> Dict:
    """
    Load a JSON file and return its content as a dictionary.

    Parameters:
    -----------
        filepath (Path): The full path to the JSON file.

    Returns:
    --------
        Dict: Parsed JSON data.
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        logger.error(f"Failed to load {json_file_path}: {e}")
        return {}


def count_entities_per_class(data: Dict) -> Dict[str, int]:
    """
    Count the number of entities per class in a JSON annotation.

    Parameters:
    -----------
        data (Dict): The JSON data loaded from an annotation file.

    Returns:
    --------
        Dict[str, int]: A dictionary containing the count of entities per class.
    """
    # Create empty dictionnary.
    entity_counts = {cls: 0 for cls in CLASSES}

    for entity in data["entities"]:
        entity_counts[entity["label"]] += 1

    return entity_counts


def write_tsv(
    count_entities: List[Dict[str, int]],
    filenames: List[str],
    len_files: List[int],
    output_file: Path,
) -> None:
    """
    Write the aggregated entity counts into a TSV file.

    Parameters:
    -----------
    count_entities: List[Dict[str, int]]
        A list of dictionaries with entity counts.
    filenames: List[str]
        A list of corresponding filenames.
    len_files: List[int]
        A list of annotated text lenghts.
    output_file: str
        Path to the TSV output file.
    """
    header = ["filename", "length"] + [f"NB_{cls}" for cls in CLASSES]

    try:
        with open(output_file, "w", newline="", encoding="utf-8") as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=header, delimiter="\t")
            writer.writeheader()

            for filename, counts, length in zip(filenames, count_entities, len_files):
                row = {"filename": filename, "length": length}
                for cls in CLASSES:
                    row[f"NB_{cls}"] = counts.get(cls, 0)
                writer.writerow(row)

    except Exception as e:
        logger.error(f"Failed to write TSV file: {e}")


if __name__ == "__main__":
    logger.info("Searching for JSON files...")
    json_files = list_json_files(ANNOTATION_DIR)
    count_entities = []
    filenames = []
    len_files = []
    count_all_entities = 0
    count_all_entities_per_class = {cls: 0 for cls in CLASSES}
    logger.info("Counting entities...")
    for filepath in json_files:
        data = load_json(filepath)
        counts = count_entities_per_class(data)
        count_all_entities += sum(counts.values())
        for cls, value in counts.items():
            count_all_entities_per_class[cls] += value

        count_entities.append(counts)
        filenames.append(filepath.name)
        len_files.append(len(data["raw_text"]))

    logger.debug(f"Total number of entities: {count_all_entities}")
    for cls, value in count_all_entities_per_class.items():
        logger.info(f"Number of entities for class '{cls}': {value}")

    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tsv_file_path = RESULTS_DIR / Path("all_annotations_entities_count.tsv")
    write_tsv(count_entities, filenames, len_files, tsv_file_path)

    logger.success(f"Count results saved in {tsv_file_path}")
