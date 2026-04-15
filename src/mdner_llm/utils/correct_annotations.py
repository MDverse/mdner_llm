"""Correct annotation files by adding missing entities and removing incorrect ones."""

import json
import operator
import re
from pathlib import Path

from loguru import logger

from mdner_llm.utils.visualize_annotations import visualize_annotations_from_json_file


def remove_entity_annotation_file(
    file_path: Path,
    entities_to_remove: list[tuple[str, str, int | None]],
) -> None:
    """
    Remove specific entities from an annotation file.

    Parameters
    ----------
    file_path : Path
        Path to the formatted annotation JSON file.
    entities_to_remove : list[tuple[str, str, int | None]]
        List of (label, text, index) tuples to remove.
        - If index is None, remove all occurrences.
        - If index is an int, remove only that occurrence (0-based).
    """
    # Load annotation data
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    new_entities = []
    for ent in data["entities"]:
        keep = True
        for label, text, idx in entities_to_remove:
            if ent["category"] == label and ent["text"] == text:
                if idx is None:
                    keep = False  # Remove all occurrences
                    break
                else:
                    # Remove only the specific indexed occurrence
                    # Count occurrences
                    matches = [
                        e
                        for e in data["entities"]
                        if e["category"] == label and e["text"] == text
                    ]
                    if matches.index(ent) == idx:
                        keep = False
                        break
        if keep:
            new_entities.append(ent)

    data["entities"] = sorted(new_entities, key=operator.itemgetter("start"))

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
    list[tuple[int, int]]
        Character index pairs for each occurrence.
    """
    positions = []
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


def has_overlap(start: int, end: int, entities: list[dict]) -> bool:
    """Check if a span overlaps with existing entities.

    Parameters
    ----------
    start : int
        Start index of the new entity.
    end : int
        End index of the new entity.
    entities : list[dict]
        List of existing entities with "start" and "end" keys.

    Returns
    -------
    bool
        True if there is an overlap, False otherwise.
    """
    return any(start < ent["end"] and ent["start"] < end for ent in entities)


def add_entity_annotation_file(
    file_path: Path,
    new_entities: list[tuple[str, str]],
) -> None:
    """
    Add new (label, text) entities to an annotation file.

    Entities are only added if:
    - They are not already present
    - They do not overlap with existing entities

    Parameters
    ----------
    file_path : Path
        Path to the formatted annotation JSON file.
    new_entities : list[tuple[str, str]]
        List of (label, text) pairs to insert.
    """
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    raw_text = data["raw_text"]

    for category, text in new_entities:
        positions = find_entity_positions(raw_text, text)

        for start, end in positions:
            entity_dict = {
                "category": category,
                "text": text,
                "start": start,
                "end": end,
            }

            # Skip if exact duplicate
            if entity_dict in data["entities"]:
                continue

            # Skip if overlap
            if has_overlap(start, end, data["entities"]):
                continue

            data["entities"].append(entity_dict)

    # Sort entities
    data["entities"] = sorted(data["entities"], key=operator.itemgetter("start"))

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def correct_and_visualize(
    file_path: Path | str,
    add_ent: list[tuple[str, str]],
    remove_ent: list[tuple[str, str]],
) -> None:
    """Correct an annotation file then visualize the results."""
    # Ensure file_path is a Path object
    file_path = Path(file_path)
    if add_ent:
        add_entity_annotation_file(file_path, add_ent)
    if remove_ent:
        remove_entity_annotation_file(file_path, remove_ent)

    visualize_annotations_from_json_file(file_path)


def clean_trailing_dot(text: str) -> str:
    """
    Remove a dot if it is at the end of a word and not followed by a digit.

    Parameters
    ----------
    text : str
        Raw text to clean.

    Returns
    -------
    str
        Cleaned text.
    """
    # Pattern explanation:
    # \. matches a literal dot
    # (?!\d) is a negative lookahead that asserts
    # the dot is not followed by a digit
    return re.sub(r"\.(?!\d)", "", text)


def clean_annotation_file_temperatures(file_path: Path) -> None:
    """
    Correct temperature annotations in a JSON annotation file.

    Parameters
    ----------
    file_path : Path
        Path to the formatted annotation JSON file.
    """
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)
    # Clean trailing dots for temperature annotations
    count = 0
    for ent in data.get("entities", []):
        if ent.get("label") == "TEMP":
            raw_entity = ent["text"]
            ent["text"] = clean_trailing_dot(ent["text"])
            if ent["text"] != raw_entity:
                logger.info(f"Changed '{raw_entity}' to '{ent['text']}'")
                count += 1
            # Ajust end index if text length changed
            if ent["text"] != data["raw_text"][ent["start"] : ent["end"]]:
                ent["end"] = ent["start"] + len(ent["text"])

    if count > 0:
        logger.success(f"Cleaned {count} temperature annotations in {file_path.name}.")

    # Save cleaned file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_ffm_list(md_path: Path) -> list[str]:
    """
    Load controlled force field names from a markdown file.

    Parameters
    ----------
    md_path : Path
        Path to the markdown file containing the controlled list.

    Returns
    -------
    list[str]
        List of force field names.
    """
    ffm_names = []
    with open(md_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Ignore empty lines and comments (lines starting with # or *)
            if not line or line.startswith(("#", "*")):
                continue
            ffm_names.append(line.split()[0])

    return ffm_names


def split_ffm_entities(
    annotation_file: Path,
    ffm_md_path: Path,
) -> None:
    """
    Split FFMNAME entities into FFMNAME + FFMVERS when applicable.

    Parameters
    ----------
    annotation_file : Path
        Path to annotation JSON file.
    ffm_md_path : Path
        Path to markdown file containing controlled FFM names.
    """
    # Load controlled FFM names
    ffm_names = load_ffm_list(ffm_md_path)
    # Sort by length to match longest names first (e.g., "CHARMM36" before "CHARMM")
    # This prevents partial matches that could occur if shorter names are matched first.
    ffm_names = sorted(ffm_names, key=len, reverse=True)
    # Load annotation data
    with open(annotation_file, encoding="utf-8") as f:
        data = json.load(f)

    raw_text = data["raw_text"]
    new_entities = []
    for ent in data["entities"]:
        # Only process FFMNAME entities
        if ent["label"] != "FFMNAME":
            new_entities.append(ent)
            continue
        # Extract the text of the entity
        text = ent["text"]
        # Find the first matching controlled FFM name in the entity text
        matched_name = None
        for name in ffm_names:
            if name.lower() in text.lower():
                matched_name = name
                break
        # If no controlled FFM name is found in the entity text
        # keep it as is
        if not matched_name:
            new_entities.append(ent)
            continue
        # Find the exact position of the matched FFM name
        pattern = re.compile(re.escape(matched_name), re.IGNORECASE)
        match = pattern.search(text)
        # If the matched name is not found in the entity text (should not happen),
        # keep the original entity
        if not match:
            new_entities.append(ent)
            continue
        ffm_start = ent["start"] + match.start()
        ffm_end = ffm_start + len(match.group(0))
        # Create new FFMNAME entity for the matched name
        new_entities.append(
            {
                "label": "FFMNAME",
                "text": text[match.start() : match.end()],
                "start": ffm_start,
                "end": ffm_end,
            }
        )

        # Check if there is a version of the FFM
        remaining_start_in_ent = match.end()
        remaining_text = text[remaining_start_in_ent:]
        # If there is remaining text after the matched FFM name
        if remaining_text:
            vers_start = ent["start"] + remaining_start_in_ent
            vers_end = vers_start + len(remaining_text)
            # Create new FFMVERS entity for the remaining text
            new_entities.append(
                {
                    "label": "FFMVERS",
                    "text": raw_text[vers_start:vers_end],
                    "start": vers_start,
                    "end": vers_end,
                }
            )

    data["entities"] = sorted(new_entities, key=operator.itemgetter("start"))

    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
