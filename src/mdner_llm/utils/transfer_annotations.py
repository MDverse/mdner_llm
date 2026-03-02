"""
Transfer annotations to cleaned descriptions.

This script updates annotation JSON files by replacing the `raw_text`
field with cleaned descriptions from another directory and
recomputing entity character offsets.


Usage:
    python src/transfer_annotations.py \
        --old-annotations-path annotations/v2 \
        --clean-descriptions-path annotations/v3 \

Arguments:
    --old-annotations-path     Directory containing original JSON annotation files.
    --clean-descriptions-path  Directory containing cleaned text files.

Example:
    uv run src/transfer_annotations.py \
        --old-annotations-path annotations/v2 \
        --clean-descriptions-path annotations/v3
"""

import json
from pathlib import Path

import click
from loguru import logger

NEW_CLASSES = [
    "SOFTNAME",
    "SOFTVERS",
    "STIME",
    "MOL",
    "FFM",
    "TEMP",
]


def find_all_occurrences(text: str, entity: str) -> list[int]:
    """Return all start indices of a entity in a text.

    Parameters
    ----------
    text : str
        The text to search within.
    entity : str
        The entity to find in the text.

    Returns
    -------
    list[int]
        A list of start indices where the entity occurs in the text.
    """
    starts = []
    start = 0
    # Loop until no more occurrences are found
    while True:
        # Find the next occurrence of the entity
        # starting from the last found index
        idx = text.find(entity, start)
        # If no more occurrences are found, break the loop
        if idx == -1:
            break
        starts.append(idx)
        start = idx + 1
    return starts


def recompute_entities(
    raw_text: str,
    entities: list[dict[str, str | list[str] | list[dict] | None]],
) -> list[dict[str, str | list[str] | list[dict] | None]]:
    """Recompute start and end positions of entities in new text.

    Parameters
    ----------
    raw_text : str
        The cleaned text to search within.
    entities : list[dict[str, str | list[str] | list[dict] | None]]
        The list of entities with 'label' and 'text' fields.

    Returns
    -------
    list[dict[str, str | list[str] | list[dict] | None]]
        A list of entities with updated 'start' and 'end' positions.
    """
    updated_entities = []
    used_positions = set()

    # For each entity
    for entity in entities:
        # Extract the label and text value
        label = entity["label"]
        text_value = entity["text"]
        # Find all occurrences of the entity text in the raw text
        occurrences = find_all_occurrences(raw_text, text_value)
        # Try to find an occurrence that hasn't been used yet
        position = None
        for occ in occurrences:
            if occ not in used_positions:
                position = occ
                used_positions.add(occ)
                break
        # If no unused occurrence is found, skip this entity
        if position is None:
            continue
        # Append the updated entity with new start and end positions
        updated_entities.append(
            {
                "label": label,
                "text": text_value,
                "start": position,
                "end": position + len(text_value),
            }
        )

    return updated_entities


def process_file(
    json_path: Path,
    clean_text_path: Path,
    output_path: Path,
) -> None:
    """Process a single annotation file."""
    # Read the original JSON annotation
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # Read the cleaned text
    clean_text = clean_text_path.read_text(encoding="utf-8")

    # Update the raw_text field with the cleaned text
    data["raw_text"] = clean_text
    # Update the classes to the new schema
    data["classes"] = NEW_CLASSES
    # Recompute entity positions based on the cleaned text
    data["entities"] = recompute_entities(
        clean_text,
        data.get("entities", []),
    )
    # Write the updated annotation back to a new JSON file
    output_path = output_path / json_path.name
    output_path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


@click.command()
@click.option(
    "--old-annotations-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("annotations/v2"),
    help="Directory containing original annotation JSON files.",
)
@click.option(
    "--clean-descriptions-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("annotations/v3"),
    help="Directory containing cleaned text descriptions.",
)
def main(
    old_annotations_path: Path,
    clean_descriptions_path: Path,
) -> None:
    """CLI entry point."""
    logger.info("Starting annotation transfer process...")
    clean_descriptions_path.mkdir(parents=True, exist_ok=True)
    # Collect all JSON files from the old annotations path
    old_json_files = list(old_annotations_path.glob("*.json"))
    logger.info(
        f"Found {len(old_json_files)} annotation JSON files in `{old_annotations_path}`"
    )
    # Process each JSON file and transfer annotations
    for json_file in sorted(old_json_files):
        txt_file = clean_descriptions_path / f"{json_file.stem}.txt"
        if not txt_file.exists():
            logger.warning(
                f"Missing cleaned description {txt_file} for {json_file.name}."
            )
            logger.warning(f"Skipping {json_file.name}.")
            continue

        process_file(json_file, txt_file, clean_descriptions_path)
        logger.info(f"Processed {json_file.name}.")

    logger.success(
        f"Saved {len(old_json_files)} updated annotations successfully "
        f"to `{clean_descriptions_path}`!"
    )


if __name__ == "__main__":
    main()
