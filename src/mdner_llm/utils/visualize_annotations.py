"""Visualize annotated entities using spaCy displaCy."""

import json
from pathlib import Path
from typing import Any

from spacy import displacy


def convert_annotations(file_path: str) -> list[dict[str, Any]]:
    """
    Convert a custom JSON annotation file to spaCy displaCy format.

    The input JSON file must contain:
        - "raw_text": the original text string
        - "entities": a list of dictionaries with "start", "end", and "label"

    Parameters
    ----------
    file_path : str
        Path to the JSON file.

    Returns
    -------
    List[Dict[str, Any]]
        A list containing a single dictionary formatted for displaCy
        with keys "text" and "ents".
    """
    # Load annotation data from JSON file
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    # Normalize raw text by removing line breaks and extra whitespace
    formatted_text = data["raw_text"].replace("\n", " ")

    # Convert entity spans to displaCy-compatible format
    ents = [
        {
            "start": item["start"],
            "end": item["end"],
            "label": item["label"],
        }
        for item in data["entities"]
    ]

    # Return displaCy input structure
    return [{"text": formatted_text, "ents": ents}]


def visualize_annotations(file_path: Path) -> None:
    """
    Render annotated entities in the browser using spaCy displaCy.

    Parameters
    ----------
    file_path : Path
        Path to the JSON annotation file.
    """
    # Define entity label colors
    colors = {
        "TEMP": "#ffb3ba",
        "SOFTNAME": "#ffffba",
        "SOFTVERS": "#ffffe4",
        "STIME": "#baffc9",
        "MOL": "#bae1ff",
        "FFM": "#cdb4db",
    }

    options = {"colors": colors}

    # Print header in console
    print("=" * 80)
    print(f"VISUALIZATION OF ENTITIES ({file_path.name})")
    print("=" * 80)

    # Convert annotations and render with displaCy
    converted_data = convert_annotations(str(file_path))
    displacy.render(converted_data, style="ent", manual=True, options=options)

    print()
