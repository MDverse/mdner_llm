"""Module for characterizing and visualizing annotations."""

import json
from pathlib import Path

import loguru
from spacy import displacy

from mdner_llm.logger import create_logger
from mdner_llm.models.entities import ListOfEntities
from mdner_llm.visualization.colors import COLORS


def convert_annotations_from_model(
    response: ListOfEntities, text_to_annotate: str
) -> list[dict]:
    """
    Convert custom entity list to spaCy displaCy format.

    If response is ListOfEntities → compute spans by locating text occurrences.

    Returns
    -------
    list[dict]  (spaCy displaCy manual format)
    """
    ents = []
    # If positions are provided, use them directly
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
                    ents.append({"start": start, "end": end, "label": entity.category})
                    break
                else:
                    search_pos = start + 1

            if start == -1:
                loguru.logger.warning(
                    f"Warning: entity '{span_text}' not found in text."
                )

        return [{"text": text_to_annotate.replace("\n", " "), "ents": ents}]


def visualize_llm_annotation(response: ListOfEntities, text_to_annotate: str) -> None:
    """Visualize named entities from LLM annotations using spaCy's displaCy."""
    print("=" * 80)
    print("🧐 VISUALIZATION OF ENTITIES ")
    print("=" * 80)
    converted_data = convert_annotations_from_model(response, text_to_annotate)
    displacy.render(
        converted_data, style="ent", manual=True, options={"colors": COLORS}
    )
    print()


def convert_annotations_to_displacy(
    json_data: dict[str, str | list],
) -> list[dict[str, str | list]]:
    """Convert custom JSON annotation file to spaCy displaCy format.

    Uses explicit entity spans if provided in 'entities'; otherwise falls back
    to 'formatted_response' and resolves character spans.

    Returns
    -------
    list[dict]  (spaCy displaCy manual format)
    """
    source_text = json_data.get("raw_text") or json_data.get("text", "")

    # 1. First priority: direct 'entities' with start/end positions
    raw_entities = json_data.get("entities")
    if raw_entities:
        spans = [
            {
                "start": item["start"],
                "end": item["end"],
                "label": item.get("category") or item.get("label"),
            }
            for item in raw_entities
            if isinstance(item, dict) and item.get("start") is not None
        ]
        if spans:
            return [{"text": source_text, "ents": spans}]

    # 2. Fallback: extract entities from formatted_response and compute spans
    formatted_data = json_data.get("formatted_response")
    if formatted_data:
        # Convert dictionary to ListOfEntities if necessary
        if isinstance(formatted_data, dict):
            response = ListOfEntities.model_validate(formatted_data)
        elif isinstance(formatted_data, ListOfEntities):
            response = formatted_data
        else:
            response = None

        if response:
            return convert_annotations_from_model(response, source_text)

    return [{"text": source_text, "ents": []}]


def visualize_annotations_from_json_file(file_path: Path) -> None:
    """Render annotated entities in the browser using spaCy displaCy."""
    # Load annotation data from JSON file
    path = Path(file_path)
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    # Print header in console
    print("=" * 80)
    print(f"VISUALIZATION OF ENTITIES ({file_path.name}: {data.get('url', 'no URL')})")
    print("=" * 80)
    # Convert annotations and render with displaCy
    converted_data = convert_annotations_to_displacy(data)
    displacy.render(
        converted_data, style="ent", manual=True, options={"colors": COLORS}
    )
    print()


def visualize_all_annotations_from_dir(
    annotation_dir: Path | str,
) -> None:
    """Visualize all JSON annotation files in a directory."""
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

    for annotation_file_path in annotation_files:
        # Visualize each annotation file
        visualize_annotations_from_json_file(annotation_file_path)


def export_annotations_to_html(
    annotation_dir: Path | str, output_html_path: Path | str
) -> None:
    """Export entity annotations with titles and URLs to a standalone HTML file."""
    annotation_dir, output_path = Path(annotation_dir), Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_files = sorted(annotation_dir.glob("*.json"))
    docs = []
    # Build displaCy docs prefixed with formatted metadata headers
    for file_path in json_files:
        with file_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        url = data.get("url", "no URL")
        header = f"**FILE**: {file_path.name}\n"
        header += f"**URL**: {url}\n"
        converted = convert_annotations_to_displacy(data)
        for doc in converted:
            offset = len(header)
            docs.append(
                {
                    "text": header + doc["text"],
                    "ents": [
                        {
                            **span,
                            "start": span["start"] + offset,
                            "end": span["end"] + offset,
                        }
                        for span in doc["ents"]
                    ],
                }
            )
    # Render all documents at once inside a single standalone page
    html = displacy.render(
        docs,
        style="ent",
        manual=True,
        options={"colors": COLORS},
        page=True,
    )
    output_path.write_text(html, encoding="utf-8")
