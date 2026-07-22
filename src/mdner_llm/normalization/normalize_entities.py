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
from mdner_llm.normalization.ground_molecule_from_all_database import (
    call_pdb,
    call_uniprot,
    get_type,
    query_chebi_by_name,
)
from mdner_llm.normalization.normalize_stemp import norm_temp
from mdner_llm.normalization.normalize_stime_wth_llm import norm_stime

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


def norm_from_db(db_path: Path, category: str, predicted_entity: str) -> dict[str, Any]:
    """Normalize an entity using a JSON inventory file (matching names or aliases).

    Returns
    -------
    dict[str, Any]
        Dictionary containing the normalized text name.
    """
    if not db_path.exists():
        logger.warning(f"{category}: Database file not found at {db_path}.")
        return {"text_normalized": predicted_entity}

    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
        records = next(iter(data.values()))
    except (JSONDecodeError, StopIteration, OSError) as e:
        logger.error(f"{category}: Failed to read database {db_path.name}: {e}")
        return {"text_normalized": predicted_entity}

    for record in records:
        names = [record.get("name", ""), *record.get("aliases", [])]
        if predicted_entity in [n.lower().strip() for n in names if n]:
            return {"text_normalized": record.get("name") or predicted_entity}

    logger.warning(f"{category}: No match found for '{predicted_entity}'.")
    return {"text_normalized": predicted_entity}


def clean_molecule_name(entity_name: str) -> str:
    """Remove generic biological/chemical stop-words from entity names.

    Returns
    -------
    str
        Cleaned name with generic terms stripped and whitespace normalized.
    """
    blacklist = (
        r"compound?|protein?|enzyme?|ligand?|receptor?|peptide?|"
        r"antibodies?|antigen?|hormone?|substrate?|cofactor?|inhibitor|"
        r"activator?|agonist?|antagonist?|modulator?|complexe?|oligomer"
    )
    pattern = re.compile(rf"^\b({blacklist})s?\b|\b({blacklist})s?\b$", re.IGNORECASE)
    return pattern.sub("", entity_name)


def norm_mol(predicted_entity: str) -> dict[str, Any]:
    """Normalize a molecular entity using PDB, UniProt, ChEBI, or pattern fallbacks.

    Returns
    -------
    dict[str, Any]
        Dictionary containing normalized fields matching MoleculeNormalized model.
    """
    detected_type = get_type(predicted_entity)
    # 1. PDB Accessions
    if detected_type == "PDB":
        pdb_data = call_pdb(predicted_entity)
        pdb_id = pdb_data.get("id")
        return {
            "text_normalized": pdb_data.get("name") or predicted_entity,
            "molecular_type": "PDB",
            "url_from_normalization": f"https://www.rcsb.org/structure/{pdb_id}"
            if pdb_id
            else None,
        }
    # 2. UniProt Accessions
    if detected_type == "UNIPROT":
        uniprot_data = call_uniprot(predicted_entity)
        uniprot_id = uniprot_data.get("id")
        return {
            "text_normalized": uniprot_data.get("name") or predicted_entity,
            "molecular_type": "UNIPROT",
            "url_from_normalization": f"https://www.uniprot.org/uniprotkb/{uniprot_id}/entry"
            if uniprot_id
            else None,
        }
    # 3. Small Molecules
    if detected_type == "SMALL_MOLECULE":
        cleaned_predicted_entity = clean_molecule_name(predicted_entity)
        chebi_id, chebi_name = query_chebi_by_name(cleaned_predicted_entity)
        return {
            "text_normalized": chebi_name or cleaned_predicted_entity,
            "molecular_type": "SMALL_MOLECULE",
            "url_from_normalization": f"https://www.ebi.ac.uk/chebi/CHEBI:{chebi_id}"
            if chebi_id
            else None,
        }
    # 4. Other sequence types (DNA, RNA, LIPID, etc.)
    return {
        "text_normalized": cleaned_predicted_entity,
        "molecular_type": detected_type,
    }


def normalize_json_content(
    data: dict[str, Any],
    ffm_db_path: Path,
    softname_db_path: Path,
    model_name: str,
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
            ent_dict.update(norm_from_db(ffm_db_path, "FFM", predicted_entity_cleaned))
        elif entity.category == "SOFTNAME":
            ent_dict.update(
                norm_from_db(softname_db_path, "SOFTNAME", predicted_entity_cleaned)
            )
        elif entity.category == "STIME":
            items = norm_stime(predicted_entity_cleaned, model_name)
            normalized_entities.extend({**ent_dict, **item} for item in items)
            continue
        elif entity.category == "MOL":
            ent_dict.update(norm_mol(predicted_entity_cleaned))
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
    inferences_dir: Path,
    ffm_db_path: Path,
    softname_db_path: Path,
    model_name: str,
    output_dir: Path,
) -> None:
    """Load JSON files, normalize their entities, and save the updated data."""
    logger = create_logger(f"logs/normalize_{inferences_dir.name}.log")
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load all JSON files from the input directory
    json_files = list(inferences_dir.glob("*.json"))
    total_files = len(json_files)
    if total_files == 0:
        logger.warning(f"No JSON files found in {inferences_dir}")
        return
    logger.info(f"Found {total_files} JSON file(s) to process from {inferences_dir}.")

    processed_count = 0
    for file_path in json_files:
        # Load the json inference file
        data = load_json_data(file_path)
        if data is None:
            continue
        # Normalize the entities in the JSON content
        updated_data = normalize_json_content(
            data, ffm_db_path, softname_db_path, model_name, logger
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

    logger.success(f"Normalization processing complete and saved to {output_dir}.")


@click.command()
@click.option(
    "--inferences-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing the input inference JSON files.",
)
@click.option(
    "--ffm-db-path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="Path to the force field database JSON file.",
)
@click.option(
    "--softname-db-path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="Path to the software name database JSON file.",
)
@click.option(
    "--model-name",
    type=str,
    help="Name of the LLM model to use for simulation time normalization.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where normalized JSON files will be saved.",
)
def run_main_from_cli(
    inferences_dir: Path,
    ffm_db_path: Path,
    softname_db_path: Path,
    model_name: str,
    output_dir: Path,
) -> None:
    """Run the normalization process from the command line."""
    main(
        inferences_dir,
        ffm_db_path,
        softname_db_path,
        model_name,
        output_dir,
    )


if __name__ == "__main__":
    run_main_from_cli()
