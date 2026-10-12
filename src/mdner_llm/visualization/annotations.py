"""Module for characterizing and visualizing annotations."""

import json
import math
import operator
from collections import defaultdict
from pathlib import Path

import loguru
from IPython.display import HTML, display
from spacy import displacy

from mdner_llm.logger import create_logger
from mdner_llm.models.entities import ListOfEntities
from mdner_llm.visualization.colors import COLORS

MODEL_SHAPES = ["●", "▲", "■", "◆", "★", "⬟"]


def find_free_spans(
    text_lower: str,
    span_text: str,
    consumed: list[bool],
    *,
    find_all: bool = False,
) -> list[tuple[int, int]]:
    """Find non-overlapping occurrences of a span and mark them as consumed.

    Returns
    -------
    list[tuple[int, int]]
        Start and end positions of the occurrences found.
    """
    spans = []
    span_lower = span_text.lower()
    search_pos = 0
    while search_pos < len(text_lower):
        start = text_lower.find(span_lower, search_pos)
        if start == -1:
            break
        end = start + len(span_text)
        # Skip occurrences overlapping an already placed span
        if any(consumed[start:end]):
            search_pos = start + 1
            continue
        consumed[start:end] = [True] * (end - start)
        spans.append((start, end))
        if not find_all:
            break
        search_pos = end
    return spans


def convert_annotations_from_model(
    response: ListOfEntities, text_to_annotate: str
) -> list[dict]:
    """Convert a list of entities to displaCy manual format.

    Returns
    -------
    list[dict]
        Documents with text and entity spans.
    """
    entities = []
    text_lower = text_to_annotate.lower()
    consumed = [False] * len(text_to_annotate)
    for entity in response.entities:
        # Locate the first free occurrence of the entity text
        spans = find_free_spans(text_lower, entity.text, consumed)
        if not spans:
            loguru.logger.warning(f"Entity '{entity.text}' not found in text.")
        entities.extend(
            {"start": start, "end": end, "label": entity.category}
            for start, end in spans
        )
    return [{"text": text_to_annotate.replace("\n", " "), "ents": entities}]


def visualize_llm_annotation(response: ListOfEntities, text_to_annotate: str) -> None:
    """Visualize named entities from LLM annotations with displaCy."""
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
    """Convert a JSON annotation file to displaCy manual format.

    Returns
    -------
    list[dict]
        Documents with text and entity spans.
    """
    source_text = json_data.get("raw_text") or json_data.get("text", "")
    # First priority: entities with explicit start and end positions
    spans = [
        {
            "start": entity["start"],
            "end": entity["end"],
            "label": entity.get("category") or entity.get("label"),
        }
        for entity in json_data.get("entities") or []
        if isinstance(entity, dict) and entity.get("start") is not None
    ]
    if spans:
        return [{"text": source_text, "ents": spans}]
    # Fallback: compute spans from the formatted response
    formatted_data = json_data.get("formatted_response")
    if formatted_data:
        response = ListOfEntities.model_validate(formatted_data)
        return convert_annotations_from_model(response, source_text)
    return [{"text": source_text, "ents": []}]


def visualize_annotations_from_json_file(file_path: Path | str) -> None:
    """Render the annotated entities of a JSON file with displaCy."""
    path = Path(file_path)
    # Load annotation data
    data = json.loads(path.read_text(encoding="utf-8"))
    # Print header
    print("=" * 80)
    print(f"VISUALIZATION OF ENTITIES ({path.name}: {data.get('url', 'no URL')})")
    print("=" * 80)
    # Convert annotations and render them
    converted_data = convert_annotations_to_displacy(data)
    displacy.render(
        converted_data, style="ent", manual=True, options={"colors": COLORS}
    )
    print()


def visualize_all_annotations_from_dir(annotation_dir: Path | str) -> None:
    """Visualize all JSON annotation files of a directory."""
    logger = create_logger()
    # Find annotation files
    annotation_files = sorted(Path(annotation_dir).glob("*.json"))
    logger.info(
        f"Found {len(annotation_files)} JSON annotation files in {annotation_dir}."
    )
    if not annotation_files:
        logger.error(f"No JSON annotation files found in {annotation_dir}")
        return
    # Render each file
    for annotation_file_path in annotation_files:
        visualize_annotations_from_json_file(annotation_file_path)


def add_header_to_document(document: dict, header: str) -> dict:
    """Prefix a displaCy document with a header and shift its spans.

    Returns
    -------
    dict
        Document with the header and shifted entity spans.
    """
    offset = len(header)
    shifted_entities = [
        {**span, "start": span["start"] + offset, "end": span["end"] + offset}
        for span in document["ents"]
    ]
    return {"text": header + document["text"], "ents": shifted_entities}


def export_annotations_to_html(
    annotation_dir: Path | str, output_html_path: Path | str
) -> None:
    """Export annotations with titles and URLs to a standalone HTML file."""
    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents = []
    # Build one document per file, prefixed with its metadata
    for file_path in sorted(Path(annotation_dir).glob("*.json")):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        header = f"**FILE**: {file_path.name}\n**URL**: {data.get('url', 'no URL')}\n"
        documents.extend(
            add_header_to_document(document, header)
            for document in convert_annotations_to_displacy(data)
        )
    # Render all documents in a single page
    html = displacy.render(
        documents,
        style="ent",
        manual=True,
        options={"colors": COLORS},
        page=True,
    )
    output_path.write_text(html, encoding="utf-8")


def build_model_inference_path(
    model_name: str,
    temperature: float,
    base_llm_dir: Path | str = "results/llm",
) -> Path:
    """Build the inference directory of a model for a given temperature.

    Returns
    -------
    Path
        Directory containing the model inferences.
    """
    inference_dir = Path(base_llm_dir) / "inferences"
    safe_name = model_name.replace("/", "_")
    # Default temperature: raw inferences
    if math.isclose(temperature, 1.0):
        return inference_dir / "raw" / "with_instructor_with_guidelines" / safe_name
    # Other temperatures: consensus folders (0.0 -> temp_0, 0.5 -> temp_0.5)
    return inference_dir / "consensus_raw" / f"temp_{float(temperature):g}" / safe_name


def extract_entity_tuples(payload: dict) -> list[tuple[str, str]]:
    """Extract (text, category) pairs from an annotation payload.

    Returns
    -------
    list[tuple[str, str]]
        Entity texts with their categories.
    """
    entities = payload.get("entities")
    if entities is None:
        entities = payload.get("formatted_response", {}).get("entities", [])
    pairs = []
    for entity in entities:
        # Entities can be dictionaries or objects
        if isinstance(entity, dict):
            text, category = entity.get("text"), entity.get("category")
        else:
            text, category = (
                getattr(entity, "text", None),
                getattr(entity, "category", None),
            )
        if text is not None and category is not None:
            pairs.append((text.strip(), category))
    return pairs


def build_annotators(
    models: list[str],
    temperatures: list[float | int],
    base_llm_dir: Path | str,
) -> list[dict]:
    """Build one annotator per model and temperature.

    Returns
    -------
    list[dict]
        Name, shape and inference directory of each annotator.
    """
    annotators = []
    for model_index, model in enumerate(models):
        annotators.extend(
            {
                "name": model.split("/")[-1],
                "shape": MODEL_SHAPES[model_index % len(MODEL_SHAPES)],
                "dir": build_model_inference_path(model, temperature, base_llm_dir),
            }
            for temperature in temperatures
        )
    return annotators


def collect_span_voters(
    gt_data: dict, gt_stem: str, annotators: list[dict]
) -> defaultdict:
    """Collect which sources (ground truth, models) annotated each span.

    Returns
    -------
    defaultdict
        Mapping (text, category) -> set of voter names.
    """
    span_voters = defaultdict(set)
    # Ground truth votes
    for text, category in extract_entity_tuples(gt_data):
        span_voters[text, category].add("GT")
    # Model votes (true and false positives)
    for annotator in annotators:
        matches = sorted(annotator["dir"].glob(f"{gt_stem}*.json"))
        if not matches:
            continue
        model_data = json.loads(matches[0].read_text(encoding="utf-8"))
        for text, category in extract_entity_tuples(model_data):
            span_voters[text, category].add(annotator["name"])
    return span_voters


def build_displacy_entities(
    source_text: str,
    span_voters: dict,
    annotators: list[dict],
    colors: dict,
) -> tuple[list[dict], dict]:
    """Locate voted spans in the text and assign their labels and colors.

    Returns
    -------
    tuple[list[dict], dict]
        displaCy entities and label-to-color mapping.
    """
    text_lower = source_text.lower()
    consumed = [False] * len(source_text)
    displacy_entities = []
    label_colors = {}
    # Place the longest spans first so they win over shorter overlaps
    sorted_spans = sorted(span_voters, key=lambda span: len(span[0]), reverse=True)
    for entity_text, category in sorted_spans:
        voters = span_voters[entity_text, category]
        shapes = " ".join(
            annotator["shape"]
            for annotator in annotators
            if annotator["name"] in voters
        )
        # Label example: "PROT ● ▲", or "❌ PROT ●" for a false positive
        label = f"{category} {shapes}".strip()
        if "GT" not in voters:
            label = f"❌ {label}"
        label_colors[label] = colors.get(category, "#a5d8ff")
        # Mark every free occurrence of the span
        displacy_entities.extend(
            {"start": start, "end": end, "label": label}
            for start, end in find_free_spans(
                text_lower, entity_text, consumed, find_all=True
            )
        )
    return sorted(displacy_entities, key=operator.itemgetter("start")), label_colors


def build_legend_html(annotators: list[dict]) -> str:
    """Build the HTML legend mapping shapes to model names.

    Returns
    -------
    str
        HTML legend bar.
    """
    shapes_legend = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(
        f"<strong style='font-size: 15px;'>{annotator['shape']}</strong>"
        f" = {annotator['name']}"
        for annotator in annotators
    )
    return f"""
    <div style="font-family: sans-serif; font-size: 13px; background: #f8fafc;
        border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 14px;
        margin-bottom: 14px; color: #1e293b;">
        <strong>Modèles :</strong> {shapes_legend}
    </div>
    """


def build_document_html(
    document_name: str, document_url: str, legend_html: str, rendered_svg: str
) -> str:
    """Build the HTML block of one compared document.

    Returns
    -------
    str
        HTML block with document name, URL, legend and annotations.
    """
    # Link to the source URL when available
    url_link = (
        f'<a href="{document_url}" target="_blank" rel="noopener noreferrer" '
        f'style="color: #1d4ed8; text-decoration: underline;">{document_url}</a>'
        if document_url
        else "<em>N/A</em>"
    )
    return f"""
    <div style="margin-top: 25px; padding: 20px; border-radius: 8px;
        border: 1px solid #cbd5e1; background: #ffffff;">
        <div style="font-family: monospace; font-size: 13px; margin-bottom: 10px;
            color: #0f172a;">
            <strong>DOCUMENT :</strong> {document_name}<br>
            <strong>URL :</strong> {url_link}
        </div>
        {legend_html}
        <div style="font-size: 15px; line-height: 2.3; color: #111827;">
            {rendered_svg}
        </div>
    </div>
    """


def compare_models_with_ground_truth(
    gt_dir_path: Path | str,
    models: list[str],
    temperatures: list[float | int],
    base_llm_dir: Path | str = "results/llm",
    doc_stem: str | None = None,
    colors: dict = COLORS,
) -> None:
    """Display model annotations next to the ground truth for each document."""
    # Select ground truth files
    pattern = f"{doc_stem}*.json" if doc_stem else "*.json"
    gt_files = sorted(Path(gt_dir_path).glob(pattern))
    annotators = build_annotators(models, temperatures, base_llm_dir)
    legend_html = build_legend_html(annotators)
    for gt_path in gt_files:
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        source_text = gt_data.get("raw_text") or gt_data.get("text", "")
        # Gather votes and compute spans
        span_voters = collect_span_voters(gt_data, gt_path.stem, annotators)
        entities, label_colors = build_displacy_entities(
            source_text, span_voters, annotators, colors
        )
        # Render with displaCy
        rendered_svg = displacy.render(
            [{"text": source_text.replace("\n", " "), "ents": entities}],
            style="ent",
            manual=True,
            options={"colors": label_colors},
            jupyter=False,
        )
        # Display the final block
        document_html = build_document_html(
            gt_path.name, gt_data.get("url", ""), legend_html, rendered_svg
        )
        display(HTML(document_html))
