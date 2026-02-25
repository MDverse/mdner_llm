"""
Format JSON annotations.

This script processes all JSON annotation files in a specified directory ("annotations/
v1") and formats them into a more readable structure. The output JSON contains the raw
text and a list of entities with their labels and exact positions. Formatted JSON files
are saved in the "annotations/v2" directory.

Example:
--------
Input JSON file (before formatting):

{
    "classes": ["TEMP", "SOFT", "STIME", "MOL", "FFM"],
    "annotations": [
        [
            "Modeling of Arylamide Helix Mimetics in the p53 Peptide Binding Site...",
            {
                "entities": [
                    [12, 21, "MOL"],
                    [44, 47, "MOL"],
                    [557, 561, "FFM"]
                ]
            }
        ]
    ]
}

Output JSON file (after formatting):

{
    "classes": [
        "SOFTNAME",
        "SOFTVERS",
        "STIME",
        "MOL",
        "FFM",
        "TEMP"
    ],
    "raw_text":
        "Modeling of Arylamide Helix Mimetics in the p53 Peptide Binding Site...",
    "entities": [
        {
            "label": "MOL",
            "text": "Arylamide",
            "start": 12,
            "end": 21
        },
        {
            "label": "MOL",
            "text": "p53",
            "start": 44,
            "end": 47
        },
        {
            "label": "FFM",
            "text": "GAFF",
            "start": 557,
            "end": 561
        }
    ]
}

Usage:
------
    uv run src/format_json_annotations.py
"""

# METADATAS
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "AGPL-3.0 license"
__date__ = "2025"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import json
from pathlib import Path

from loguru import logger

# CONSTANTS
ANNOTATION_DIR = Path("annotations/v1")
OUT_DIR = Path("annotations/v2")
CLASSES = ["SOFTNAME", "SOFTVERS", "STIME", "MOL", "FFM", "TEMP"]


# FUNCTIONS
def split_soft_entities(entities: list[dict]) -> list[dict]:
    """
    Split 'SOFT' entities into 'SOFTNAME' and 'SOFTVERS' entities.

    Parameters
    ----------
    entities : List[Dict]
        List of entities, each with keys: label, text, start, end.

    Returns
    -------
    List[Dict]
        New list of entities with 'SOFT' entities split into two:
        - 'SOFTNAME' for the software name
        - 'SOFTVERS' for the software version
    """
    new_entities: list[dict] = []

    for ent in entities:
        if ent["label"] == "SOFT":
            text = ent["text"]
            start = ent["start"]

            # Split by the first space (assumes format "name version")
            if " " in text:
                soft_name, soft_version = text.split(" ", 1)
                new_entities.append(
                    {
                        "label": "SOFTNAME",
                        "text": soft_name,
                        "start": start,
                        "end": start + len(soft_name),
                    }
                )
                new_entities.append(
                    {
                        "label": "SOFTVERS",
                        "text": soft_version,
                        "start": start + len(soft_name) + 1,  # for the space
                        "end": ent["end"],
                    }
                )
            else:
                # If no version detected, keep as SOFTNAME only
                new_entities.append(
                    {
                        "label": "SOFTNAME",
                        "text": text,
                        "start": start,
                        "end": ent["end"],
                    }
                )
        else:
            # Keep all other entities unchanged
            new_entities.append(ent)

    return new_entities


def format_entities(entities: list[list], annotated_text: str) -> list[dict]:
    """
    Convert raw entity tuples into a list of dictionaries with labels and positions.

    Parameters
    ----------
    entities : List[list]
        List of entities where each entity is [start, end, label].
    annotated_text : str
        The full text in which entities are annotated.

    Returns
    -------
    List[Dict]
        A list of dictionaries with keys: label, text, start, end.
    """
    all_entities = []
    for ent in entities:
        start, end, label = ent
        text = annotated_text[start:end]
        all_entities.append({"label": label, "text": text, "start": start, "end": end})
    all_entities_no_soft = split_soft_entities(all_entities)

    return all_entities_no_soft


def format_json_annotations(annotations_path: Path, out_dir: Path) -> None:
    """
    Read JSON annotation files from a directory, format their entities, and save them.

    Parameters
    ----------
    annotations_path : Path
        Path to the directory containing input JSON annotation files.
    out_dir : Path
        Path to the output directory to save formated JSON annotation files.
    """
    logger.info("Starting to format older annotations...")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in annotations_path.iterdir() if f.suffix == ".json"]
    logger.debug(f"There are {len(files)} JSON annotation files!")

    for filepath in files:
        with filepath.open("r", encoding="utf-8") as file:
            data = json.load(file)
            annotated_text = data["annotations"][0][0]
            entities = data["annotations"][0][1]["entities"]

            formatted_data = {
                "classes": CLASSES,
                "raw_text": annotated_text,
                "entities": format_entities(entities, annotated_text),
            }

        out_path = out_dir / filepath.name
        with out_path.open("w", encoding="utf-8") as out_file:
            json.dump(formatted_data, out_file, ensure_ascii=False, indent=4)

    logger.success(f"Saved new formatted annotations in {out_dir} successfully!")


# MAIN PROGRAM
if __name__ == "__main__":
    format_json_annotations(ANNOTATION_DIR, OUT_DIR)
