"""Count entities par class for each annotation.

This script processes all JSON annotation files in a specified directory ("annotations")
and counts the number of entities for each class defined in the file.
The script then outputs a TSV file containing the filename and the count of entities per class.

Usage :
=======
    uv run src/count_entities_per_class.py

"""


# METADATAS
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "AGPL-3.0 license"
__date__ = "2025"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import os
import json
import csv
from typing import Dict, List

from loguru import logger
from tqdm import tqdm


# CONSTANTS
ANNOTATION_DIR = "annotations"
OUT_TSV_PATH = "annotations/all_annotations_entities_count.tsv"
CLASSES = ["TEMP", "SOFT", "STIME", "MOL", "FFM"]


# FUNCTIONS
def get_json_files(directory: str) -> List[str]:
    """
    Retrieve all JSON files from a given directory.

    Parameters:
    -----------
        directory (str): The path to the directory containing JSON files.

    Returns:
    --------
        List[str]: A list of JSON file paths.
    """
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".json")
    ]
    logger.debug(f"Found {len(files)} JSON annotation files.")
    return files


def load_json(filepath: str) -> Dict:
    """
    Load a JSON file and return its content as a dictionary.

    Parameters:
    -----------
        filepath (str): The full path to the JSON file.

    Returns:
    --------
        Dict: Parsed JSON data.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return {}


def count_entities(data: Dict) -> Dict[str, int]:
    """
    Count the number of entities per class in a JSON annotation.

    Parameters:
    -----------
        data (Dict): The JSON data loaded from an annotation file.

    Returns:
    --------
        Dict[str, int]: A dictionary containing the count of entities per class.
    """
    entity_counts = {cls: 0 for cls in CLASSES}

    if "annotations" not in data:
        logger.warning("Missing 'annotations' key in JSON data.")
        return entity_counts

    # Each annotation entry is typically a [text, {"entities": [...]}] pair
    for annotation in data["annotations"]:
        if isinstance(annotation, list) and len(annotation) > 1:
            entities = annotation[1].get("entities", [])
            for _, _, label in entities:
                if label in entity_counts:
                    entity_counts[label] += 1

    return entity_counts


def write_tsv(results: List[Dict[str, int]], filenames: List[str], output_file: str) -> None:
    """
    Write the aggregated entity counts into a TSV file.

    Parameters:
    -----------
        results (List[Dict[str, int]]): A list of dictionaries with entity counts.
        filenames (List[str]): A list of corresponding filenames.
        output_file (str): Path to the TSV output file.
    """
    header = ["filename"] + [ f"NB_{c}" for c in CLASSES]

    try:
        with open(output_file, "w", newline="", encoding="utf-8") as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=header, delimiter="\t")
            writer.writeheader()

            for filename, counts in zip(filenames, results):
                row = {"filename": filename}
                for cls in CLASSES:
                    row[f"NB_{cls}"] = counts.get(cls, 0)
                writer.writerow(row)
    except Exception as e:
        logger.error(f"Failed to write TSV file: {e}")


# MAIN PROGRAM
if __name__ == "__main__":
    logger.info("Starting JSON annotation counting...")

    json_files = get_json_files(ANNOTATION_DIR)
    results = []
    filenames = []
    count_all_entities = 0
    count_all_entities_per_class = {cls: 0 for cls in CLASSES}

    for filepath in json_files:
        data = load_json(filepath)
        counts = count_entities(data)

        count_all_entities += sum(counts.values())
        for cls, value in counts.items():
            count_all_entities_per_class[cls] += value

        results.append(counts)
        filenames.append(os.path.basename(filepath))
    
    logger.debug(f"Total number of entities : {count_all_entities}")
    logger.debug(f"Entity count per class: {count_all_entities_per_class}")

    write_tsv(results, filenames, OUT_TSV_PATH)
    
    logger.success(f"Successfully counted entities for each class and saved in {OUT_TSV_PATH}!")


