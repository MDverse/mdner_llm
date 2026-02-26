"""Correct annotation files by adding missing entities and removing incorrect ones."""

import json
from pathlib import Path

from mdner_llm.utils.visualize_annotations import visualize_annotations


def remove_entity_annotation_file(
    file_path: Path,
    entities_to_remove: list[tuple[str, str]],
) -> None:
    """
    Remove specific (label, text) entities from an annotation file.

    Parameters
    ----------
    file_path : Path
        Path to the formatted annotation JSON file.
    entities_to_remove : List[Tuple[str, str]]
        List of (label, text) pairs to remove.
    """
    # Load annotation data
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)
        # Filter out matching entities
        data["entities"] = [
            ent
            for ent in data["entities"]
            if (ent["label"], ent["text"]) not in entities_to_remove
        ]
    # Save updated file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def find_entity_positions(raw_text: str, entity_text: str) -> list[tuple[int, int]]:
    """
    Return all (start, end) positions of entity_text in raw_text.

    Parameters
    ----------
    raw_text : str
        Full text to search in.
    entity_text : str
        Substring to locate.

    Returns
    -------
    List[Tuple[int, int]]
        Character index pairs for each occurrence.
    """
    positions: list[tuple[int, int]] = []
    start_idx = 0

    # Iteratively search for non-overlapping occurrences
    while True:
        start = raw_text.find(entity_text, start_idx)
        if start == -1:
            break

        end = start + len(entity_text)
        positions.append((start, end))
        start_idx = end

    return positions


def add_entity_annotation_file(
    file_path: Path,
    new_entities: list[tuple[str, str]],
) -> None:
    """
    Add new (label, text) entities to an annotation file.

    Parameters
    ----------
    file_path : Path
        Path to the formatted annotation JSON file.
    new_entities : List[Tuple[str, str]]
        List of (label, text) pairs to insert.
    """
    # Load annotation data
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    raw_text = data["raw_text"]

    # Insert entities if not already present
    for label, text in new_entities:
        positions = find_entity_positions(raw_text, text)

        for start, end in positions:
            entity_dict = {
                "label": label,
                "text": text,
                "start": start,
                "end": end,
            }

            if entity_dict not in data["entities"]:
                data["entities"].append(entity_dict)

    # Save updated file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def correct_and_visualize(
    file_path: Path, add_ent: list[tuple[str, str]], remove_ent: list[tuple[str, str]]
) -> None:
    """Correct an annotation file then visualize the results."""
    if add_ent:
        add_entity_annotation_file(file_path, add_ent)
    if remove_ent:
        remove_entity_annotation_file(file_path, remove_ent)

    visualize_annotations(file_path)
