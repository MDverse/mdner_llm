"""Normalize extracted entities from JSON inferences."""

import json
import re
import unicodedata
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import click
import loguru
from loguru import logger
from pydantic import ValidationError

from mdner_llm.logger import create_logger
from mdner_llm.models.entities import ListOfEntities, ListOfEntitiesNormalized

STIME_RE = re.compile(r"([0-9]+)(\.?[0-9]+)? *(ps|ns|μs|ms|s)", re.IGNORECASE)
STEMP_RE = re.compile(r"([0-9]+)(\.?[0-9]+)?( *˚? *[a-z]*)?", re.IGNORECASE)
LATEX_SUBSCRIPT_RE = re.compile(r"_([0-9]+[+-]?)")


def load_json_data(file_path: Path) -> dict[str, Any] | None:
    """
    Safely load a JSON file.

    Returns
    -------
        dict[str, Any] | None: The parsed data or None if a IO/JSON error occurs.
    """
    try:
        with open(file_path, encoding="utf-8") as json_file:
            return json.load(json_file)
    except (FileNotFoundError, JSONDecodeError) as error:
        logger.error(f"Failed to read or parse JSON file {file_path.name}: {error}")
        return None


def clean_text_fields(value: Any) -> Any:
    """Recursively clean superscripts, apostrophes, dashes, and invisible characters.

    Returns
    -------
        Any: The cleaned value, preserving the original type.
    """
    if isinstance(value, dict):
        return {k: clean_text_fields(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_text_fields(item) for item in value]
    if isinstance(value, str):
        value = value.strip().lower()
        # Exponentiation and index characters are normalized to their base forms
        # e.g., "cacl₂" -> "cacl2", "NO3⁻" -> "NO3-"
        value = unicodedata.normalize("NFKC", value)
        table = str.maketrans(
            {
                # Non-breaking hyphen
                "‑": "-",  # e.g., "sars‑cov‑2" -> "sars-cov-2"  # noqa: RUF001, RUF003
                # Mathematical minus sign
                "−": "-",  # e.g., "U(VI)−complex" -> "U(VI)-complex"  # noqa: RUF001, RUF003
                # En dash (commonly used for charges in literature)
                "–": "-",  # e.g., "(-)-METH" -> "(-)-METH"  # noqa: RUF001
                # Em dash
                "—": "-",  # e.g., "protein—ligand" -> "protein-ligand"
                # Typographic/curly apostrophe
                "’": "'",  # e.g., "glycol’s" -> "glycol's"  # noqa: RUF001, RUF003
                # Soft hyphen / Invisible discretionary hyphen
                "\xad": None,  # e.g., "poly\xad(ethylene)" -> "poly(ethylene)"
            }
        )
        return value.translate(table)
    return value


def check_numeric_pattern(
    predicted_entity: str, source_text: str, pattern: re.Pattern
) -> bool:
    """Check if the numeric part of an entity exists in the source text.

    Returns
    -------
        bool: True if the numeric part is found in the source text, False otherwise.
    """
    match = pattern.match(predicted_entity)
    if match:
        numeric_part = match.group(1) + (match.group(2) or "")
        return numeric_part in source_text
    return False


def check_hallucination(
    predicted_entity: str, category: str, source_text: str, source_file: str, url: str
) -> bool:
    """Determine if an entity is hallucinated.

    Returns
    -------
        bool: True if the entity is hallucinated, False otherwise.
    """
    # 1. Direct exact match check
    if predicted_entity in source_text:
        return False

    # 2. Category-specific regex decomposition
    if category in ("STIME", "STEMP"):
        pattern = STIME_RE if category == "STIME" else STEMP_RE
        if check_numeric_pattern(predicted_entity, source_text, pattern):
            return False
    elif category == "MOL":
        # Handle LaTeX-style subscripts from predictions (e.g., "cacl_2" -> "cacl2")
        # This aligns LaTeX notations with cleaned raw text formats
        clean_mol = LATEX_SUBSCRIPT_RE.sub(r"\1", predicted_entity)
        if clean_mol in LATEX_SUBSCRIPT_RE.sub(r"\1", source_text):
            return False

    # 3. Multi-word proximity check
    words = predicted_entity.split()
    if len(words) > 1:
        tokens = source_text.split()
        # Allows 5 words before and 5 words after
        window_size = len(words) + 10
        for i in range(len(tokens) - len(words) + 1):
            sub_window = " ".join(tokens[i : i + window_size])
            if all(word in sub_window for word in words):
                return False

    logger.warning(f"Entity '{predicted_entity}' ({category}) is hallucinated.")
    logger.warning(f"File: {source_file}.")
    logger.warning(f"URL: {url}.")
    return True


def normalize_json_content(
    data: dict[str, Any], logger: "loguru.Logger" = loguru.logger
) -> dict[str, Any] | None:
    """Normalize entities in the JSON content and check for hallucinations.

    Returns
    -------
        dict[str, Any] | None: The updated data with normalized entities
        or None if validation fails.
    """
    source_text = clean_text_fields(data.get("text", ""))
    # Parse the original entities using the Pydantic model
    try:
        original_list = ListOfEntities.model_validate(
            data.get("formatted_response", {})
        )
    except ValidationError as e:
        logger.error(f"Pydantic validation failed for input data: {e}")
        return None
    # Normalize entities and compute hallucination flags
    normalized_entities = []
    for entity in original_list.entities:
        ent_dict = entity.model_dump()
        predicted_entity_cleaned = clean_text_fields(entity.text)
        ent_dict["text_normalized"] = predicted_entity_cleaned
        ent_dict["is_hallucinated"] = check_hallucination(
            predicted_entity_cleaned,
            entity.category,
            source_text,
            data.get("input_json_path", "NA"),
            data.get("url", "NA"),
        )
        normalized_entities.append(ent_dict)
    # Create a new ListOfEntitiesNormalized instance and validate it
    try:
        normalized_list = ListOfEntitiesNormalized.model_validate(
            {"entities": normalized_entities}
        )
        data["normalized_entities"] = normalized_list.model_dump()
        return data
    except ValidationError as e:
        logger.error(f"Pydantic validation failed for normalized output: {e}")
        return None


def save_json_data(
    data: dict[str, Any], output_path: Path, logger: "loguru.Logger" = loguru.logger
) -> bool:
    """
    Safely save the updated dictionary to a destination path.

    Returns
    -------
        bool: True if saving succeeded, False otherwise.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except OSError as e:
        logger.error(f"Failed to write file {output_path.name}: {e}")
        return False


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing the input inference JSON files.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where normalized JSON files will be saved.",
)
def main(input_dir: Path, output_dir: Path) -> None:
    """Load JSON files, normalize their entities, and save the updated data."""
    logger = create_logger(f"logs/normalize_{input_dir.name}.log")
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load all JSON files from the input directory
    json_files = list(input_dir.glob("*.json"))
    total_files = len(json_files)
    if total_files == 0:
        logger.warning(f"No JSON files found in {input_dir}")
        return
    logger.info(f"Found {total_files} JSON file(s) to process from {input_dir}.")

    processed_count = 0
    for file_path in json_files:
        # Load the json file
        data = load_json_data(file_path)
        if data is None:
            continue
        # Normalize the entities in the JSON content
        updated_data = normalize_json_content(data, logger)
        if updated_data is None:
            continue
        # Save the updated data to the output directory
        output_path = output_dir / f"normalized_{file_path.name}"
        if save_json_data(updated_data, output_path, logger):
            processed_count += 1
            # Log progress
            percent_done = (processed_count / total_files) * 100
            logger.info(
                f"Processed {processed_count}/{total_files} files ({percent_done:.1f}%)"
            )

    logger.success("Normalization processing complete.")
    logger.success(f"Total successfully processed: {processed_count}/{total_files}")


if __name__ == "__main__":
    main()
