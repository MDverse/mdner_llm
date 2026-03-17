"""Visualize annotated entities using spaCy displaCy."""

import json
from pathlib import Path
from typing import Any

from loguru import logger
from spacy import displacy

from mdner_llm.core.logger import create_logger
from mdner_llm.models.entities import ListOfEntities
from mdner_llm.models.entities_with_positions import ListOfEntitiesPositions

COLORS = {
    "TEMP": "#ffb3ba",
    "SOFTNAME": "#ffffba",
    "SOFTVERS": "#ffffe4",
    "STIME": "#baffc9",
    "MOL": "#bae1ff",
    "FFM": "#cdb4db",
}


def _convert_annotations_to_displacy(
    json_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert a custom JSON annotation file to spaCy displaCy format.

    The input JSON file must contain:
        - "raw_text": the original text string
        - "entities": a list of dictionaries with "start", "end", and "label"

    Parameters
    ----------
    json_data : dict[str, Any]
        The JSON data containing the raw text and entities.

    Returns
    -------
    list[dict[str, Any]]
        A list containing a single dictionary formatted for displaCy
        with keys "text" and "ents".
    """
    # Convert entity spans to displaCy-compatible format
    ents = [
        {
            "start": item["start"],
            "end": item["end"],
            "label": item["label"],
        }
        for item in json_data["entities"]
    ]

    # Return displaCy input structure
    return [{"text": json_data["raw_text"], "ents": ents}]


def visualize_annotations_from_json_file(file_path: Path) -> None:
    """
    Render annotated entities in the browser using spaCy displaCy.

    Parameters
    ----------
    file_path : Path
        Path to the JSON annotation file.
    """
    # Load annotation data from JSON file
    path = Path(file_path)
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    # Print header in console
    print("=" * 80)
    print(f"VISUALIZATION OF ENTITIES ({file_path.name}: {data.get('url', 'no URL')})")
    print("=" * 80)

    # Convert annotations and render with displaCy
    converted_data = _convert_annotations_to_displacy(data)
    displacy.render(
        converted_data, style="ent", manual=True, options={"colors": COLORS}
    )

    print()


def _load_selected_annotation_names(
    selected_annotations_path: Path,
) -> set[str]:
    """
    Load selected annotation file names from a text file.

    Each line of the file must contain a path to a JSON annotation file.
    Only the file name (not the full path) is retained for matching.

    Parameters
    ----------
    selected_annotations_path : Path
        Path to the text file containing annotation file paths.

    Returns
    -------
    set[str]
        Set of JSON file names to keep.
    """
    with selected_annotations_path.open("r", encoding="utf-8") as file:
        return {Path(line.strip()).name for line in file if line.strip()}


def visualize_all_annotations_from_dir(
    annotation_dir: Path | str, selected_annotations: Path | str | None = None
) -> None:
    """
    Visualize all JSON annotation files in a directory.

    Parameters
    ----------
    annotation_dir : Path | str
        Path to the directory containing JSON annotation files.
    selected_annotations : Path | str | None, optional
        Path to a text file containing paths to selected JSON annotation
        files. Matching is performed on file names only.
    """
    # Create logger
    logger = create_logger()
    # Validate annotation directories
    # Load all JSON annotation files in the specified directory
    annotation_files = list(Path(annotation_dir).glob("*.json"))
    logger.info(
        f"Found {len(annotation_files)} JSON annotation files in {annotation_dir}."
    )
    if not annotation_files:
        logger.error(f"No JSON annotation files found in {annotation_dir}")
        return

    if selected_annotations is not None:
        selected_annotations_path = Path(selected_annotations)
        # Validate that the selected annotations file exists
        if not selected_annotations_path.exists():
            logger.error(
                f"Selected annotations file not found: {selected_annotations_path}"
            )
            return
        # Load selected annotation file names
        selected_names = _load_selected_annotation_names(
            selected_annotations_path,
        )
        # Filter the list of annotation files
        annotation_files = [
            path for path in annotation_files if path.name in selected_names
        ]
        logger.info(f"After filtering, {len(annotation_files)} JSON files remain.")
        if not annotation_files:
            logger.warning("No matching JSON files found after filtering.")
            return

    for annotation_file_path in annotation_files:
        # Visualize each annotation file
        visualize_annotations_from_json_file(annotation_file_path)


def convert_ner_response_to_entities(
    response: dict[str, list[dict]],
) -> ListOfEntitiesPositions:
    """Convert raw NER response into ListOfEntitiesPositions.

    Parameters
    ----------
    response : dict[str, list[dict]]
        The raw response from the GLINER containing entity annotations.

    Returns
    -------
        Structured ListOfEntitiesPositions instance.
    """
    entities = []
    raw_entities = response.get("entities", {})

    for label, items in raw_entities.items():
        for item in items:
            entities.append(  # noqa: PERF401
                {
                    "label": label,
                    "text": item["text"],
                    "start": item["start"],
                    "end": item["end"],
                }
            )
    return ListOfEntitiesPositions(entities=entities)


def convert_annotations_from_llm(response, text_to_annotate):
    """
    Convert custom entity list to spaCy displaCy format.

    If response is ListOfEntities → compute spans by locating text occurrences.
    If response is ListOfEntitiesPositions → use provided start/end positions.

    Parameters
    ----------
    response : ListOfEntities | ListOfEntitiesPositions

    Returns
    -------
    list[dict]  (spaCy displaCy manual format)
    """
    ents = []

    # ---------- CASE 1 : response is ListOfEntitiesPositions ----------
    # Already has (start, end, label)
    if isinstance(response, ListOfEntitiesPositions):
        ents.extend(
            {"start": ent.start, "end": ent.end, "label": ent.label}
            for ent in response.entities
        )
        return [{"text": text_to_annotate, "ents": ents}]

    # ---------- CASE 2 : response is ListOfEntities ----------
    # We must find spans in TEXT_TO_ANNOTATE
    if isinstance(response, ListOfEntities):
        text_lower = text_to_annotate.lower()
        consumed = [False] * len(text_to_annotate)

        for entity in response.entities:
            span_text = entity.text
            span_lower = span_text.lower()

            start = -1
            search_pos = 0

            while True:
                start = text_lower.find(span_lower, search_pos)
                if start == -1:
                    break

                end = start + len(span_text)

                # avoid overlap
                if not any(consumed[start:end]):
                    for i in range(start, end):
                        consumed[i] = True

                    ents.append({"start": start, "end": end, "label": entity.label})
                    break
                else:
                    search_pos = start + 1

            if start == -1:
                logger.warning(f"Warning: entity '{span_text}' not found in text.")

        return [{"text": text_to_annotate.replace("\n", " "), "ents": ents}]


def visualize_llm_annotation(
    response: ListOfEntities | ListOfEntitiesPositions, text_to_annotate: str
):
    """
    Visualize named entities from LLM annotations using spaCy's displaCy.

    Parameters
    ----------
    response (ListOfEntities | ListOfEntitiesPositions):
        The annotated entities returned by the LLM.
    text_to_annotate (str):
        The original text on which entities were predicted.
    """
    print("=" * 80)
    print("🧐 VISUALIZATION OF ENTITIES ")
    print("=" * 80)
    converted_data = convert_annotations_from_llm(response, text_to_annotate)
    displacy.render(
        converted_data, style="ent", manual=True, options={"colors": COLORS}
    )
    print()
