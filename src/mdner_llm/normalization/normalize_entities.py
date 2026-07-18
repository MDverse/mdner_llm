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
from mdner_llm.models.entities import ListOfEntities
from mdner_llm.models.entities_normalized import ListOfEntitiesNormalized
from mdner_llm.normalization.normalize_stemp import norm_temp
from mdner_llm.normalization.normalize_stime_with_regex import norm_stime_regex

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


def norm_ffm(ffm_db: dict[str, Any], predicted_entity: str) -> dict[str, Any]:
    """Normalize a force field model entity using the provided database entry.

    Returns
    -------
        dict[str, Any]: A dictionary containing normalized fields
        for the force field model.
    """
    entry = {}
    for ffm in ffm_db:
        # Create a list of names and aliases for comparison
        names = [ffm.get("name", ""), *ffm.get("aliases", [])]
        if predicted_entity in [name.lower().strip() for name in names]:
            entry = ffm
            break
    if not entry:
        logger.warning(f"FFM: No match found for '{predicted_entity}'.")
    return {
        "text_normalized": entry.get("name") or predicted_entity,
        "tag": entry.get("category"),
        "family": entry.get("family"),
        "aliases": entry.get("aliases"),
        "resolution": entry.get("resolution"),
        "molecular_type": entry.get("molecular_type"),
        "ontology_link": entry.get("ontology_link"),
        "publication_link": entry.get("publication"),
    }


def norm_softname(predicted_entity: str, base_dir: Path) -> dict[str, Any]:
    """Normalize a software name entity using its local codemeta.json file.

    Returns
    -------
        dict[str, Any]: A dictionary containing normalized fields
        for the software name, or defaults if not found.
    """
    # Path to the expected codemeta.json file
    meta_path = base_dir / predicted_entity / "codemeta.json"
    # Load data if file exists, otherwise use an empty dictionary
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as json_file:
                meta = json.load(json_file)
        except json.JSONDecodeError:
            logger.warning(
                f"SOFTNAME: Failed to parse codemeta.json for '{predicted_entity}'."
            )
    else:
        logger.warning(f"SOFTNAME: codemeta.json not found for '{predicted_entity}'.")
    # Map CodeMeta keys to Pydantic model attributes
    raw_authors = meta.get("author", [])
    formatted_authors = [
        {
            "id": a.get("id"),
            "type": a.get("type", "Person"),
            "first_name": a.get("givenName", "").strip(),
            "last_name": a.get("familyName", "").strip(),
            "affiliation": a.get("affiliation"),
        }
        for a in raw_authors
    ]
    return {
        "name": meta.get("name", predicted_entity).strip(),
        "authors": formatted_authors,
        "description": meta.get("description"),
        "version": meta.get("version"),
        "date_last_modification": meta.get("dateModified"),
        "code_repository_link": meta.get("codeRepository"),
        "download_url": meta.get("downloadUrl"),
        "related_link": meta.get("relatedLink"),
        "publication_link": meta.get("referencePublication"),
        "license": meta.get("license"),
        "keywords": meta.get("keywords"),
        "programming_language": meta.get("programmingLanguage"),
    }


def normalize_json_content(
    data: dict[str, Any],
    ffm_db: dict[str, Any],
    softname_codemeta_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
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
        if entity.category == "STEMP":
            ent_dict["value"], ent_dict["unit"] = norm_temp(entity.text)
        elif entity.category == "FFM":
            ent_dict.update(norm_ffm(ffm_db, predicted_entity_cleaned))
        elif entity.category == "SOFTNAME":
            ent_dict.update(
                norm_softname(predicted_entity_cleaned, softname_codemeta_dir)
            )
        elif entity.category == "STIME":
            ent_dict.update(norm_stime_regex(predicted_entity_cleaned))
        elif entity.category == "MOL":
            pass
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


def main(
    input_dir: Path, ffm_db_path: Path, softname_codemeta_dir: Path, output_dir: Path
) -> None:
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

    # Load the force field database for normalization
    try:
        with open(ffm_db_path, encoding="utf-8") as f:
            json_data = json.load(f)
            ffm_db = json_data.get("force_fields", [])
        logger.info(f"Loaded force field database from {ffm_db_path}.")
    except (FileNotFoundError, JSONDecodeError) as e:
        logger.error(f"Failed to load force field database: {e}")
        return

    processed_count = 0
    for file_path in json_files:
        # Load the json inference file
        data = load_json_data(file_path)
        if data is None:
            continue
        # Normalize the entities in the JSON content
        updated_data = normalize_json_content(
            data, ffm_db, softname_codemeta_dir, logger
        )
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


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing the input inference JSON files.",
)
@click.option(
    "--ffm-db-path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="Path to the force field database JSON file.",
)
@click.option(
    "--softname-codemeta-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing the software name codemeta files.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where normalized JSON files will be saved.",
)
def run_main_from_cli(
    input_dir: Path, ffm_db_path: Path, softname_codemeta_dir: Path, output_dir: Path
) -> None:
    """Run the normalization process from the command line."""
    main(input_dir, ffm_db_path, softname_codemeta_dir, output_dir)


if __name__ == "__main__":
    run_main_from_cli()
