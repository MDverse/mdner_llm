# run_qc_analysis.py

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from loguru import logger

# ======================================================================================
# Configuration
# ======================================================================================

DATE_TIME_STR = input(
    "Enter the date and time string to analyse (YYYY-MM-DD_HH-MM-SS): "
)

LLM_ANNOTATIONS = f"llm_outputs/{DATE_TIME_STR}/annotations/"
QC_RESULTS_FOLDER = f"llm_outputs/{DATE_TIME_STR}/stats/"
QC_RESULTS_PATH = os.path.join(QC_RESULTS_FOLDER, "quality_control_results.csv")

TAGS = ["MOL", "SOFTN", "SOFTV", "STIME", "TEMP", "FFM"]

CONSENSUS_SCORING = input("Is this for consensus scoring? (yes/no): ").strip().lower()
if CONSENSUS_SCORING not in ["yes", "no"]:
    logger.error("Invalid input. Choose 'yes' or 'no'.")
    sys.exit(1)
# If consensus scoring is enabled, we will use the temperatures
if CONSENSUS_SCORING == "yes":
    CONSENSUS_SCORING = True
else:
    CONSENSUS_SCORING = False

# ======================================================================================
# Helper Functions
# ======================================================================================


def strip_tags(text: str, tags: List[str]) -> str:
    """
    Remove specified tags from the text (MOL, SOFTN, SOFTV, STIME, TEMP, FFM).
    """
    for tag in tags:
        text = re.sub(f"</?{re.escape(tag)}>", "", text)
    return text.strip()


def compare_annotated_to_original(original: str, annotated: str) -> bool:
    """
    Compare the annotated text with the original text after stripping tags.
    """
    return strip_tags(annotated, TAGS).strip().lower() == original.strip().lower()


def process_llm_json_file(json_file: Union[str, Path]) -> tuple:
    """
    Process the LLM JSON file and return:
    - the text to annotate (the raw text without labels)
    - the response (the LLM response with labels)
    - the model name (the name of the model used)
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["text_to_annotate"], data["response"], data["model"]


def extract_entities_from_llm_text(text: str) -> Dict[str, List[str]]:
    """
    Extract entities from the LLM response text using regex.
    The entities are expected to be in the format <TAG>content</TAG>.
    """
    result = {tag: [] for tag in TAGS}
    pattern = re.compile(r"<([A-Z]+)>(.*?)</\1>")
    for tag, content in pattern.findall(text):
        if tag in result:
            result[tag].append(content.strip())
    return result


def find_one_valid_llm_entity(
    llm_entities: Dict[str, List[str]], input_text: str
) -> bool:
    """
    Check if at least one entity from the LLM response is present in the input text.
    """
    for values in llm_entities.values():
        for value in values:
            if value in input_text:
                return True
    return False


def define_quality_entities(
    llm_entities: Dict[str, List[str]], input_text: str
) -> bool:
    """Looks at each entity, and checks if it belong to one
    of the 3 following groups:
    - fully_valid: the entity is present in the input text
    - partially_valid: the entity is present in the input text but not
    fully (part of the entity was hallucinated)
    - invalid: the entity is not present in the input text

    Args:
        llm_entities (Dict[str, List[str]]): entities extracted from the LLM response
        input_text (str): original text to compare with

    Returns:
        fully_valid: (int) count of all the valid entities
        partially_valid: (int) count of all the partially valid entities
        invalid: (int) count of all the invalid entities
    """
    fully_valid = 0
    invalid = 0
    invalid_entities = {}

    text_lc = input_text.lower()

    for tag in TAGS:
        if tag not in invalid_entities:
            invalid_entities[tag] = []

        # Check if the tag has any entities
        if not llm_entities[tag]:
            continue

        # Check each entity for the tag
        for entity in llm_entities[tag]:
            ent_lc = entity.lower().strip()

            # 1) FULL match
            if ent_lc and ent_lc in text_lc:
                fully_valid += 1
                continue

            # 2) NO match
            else:
                invalid += 1
                invalid_entities[tag].append(entity)
    return fully_valid, invalid, invalid_entities


def save_qc_results_to_csv(
    rows: List[Dict[str, Any]], output_dir: Union[str, Path]
) -> None:
    """
    Save the quality control results to a CSV file.
    If the file already exists, append the new results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "quality_control_results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, mode="a", header=not csv_path.exists())


def extract_annotations_from_structured_llm_output(
    llm_output: Dict[str, Any],
) -> Dict[str, List[str]]:
    """_summary_

    Args:
        llm_output (Dict[str, Any]): _description_

    Returns:
        Dict[str, List[str]]: _description_
    """
    result = {tag: [] for tag in TAGS}
    # Extract annotations from a structured LLM output.
    entity_list_response = llm_output["response"]
    for annotation in entity_list_response:
        label = annotation["label"]
        text = annotation["text"]
        if label in result:
            result[label].append(text.strip())
    return result


# ======================================================================================
# Main Logic
# ======================================================================================


def quality_control(path_to_test: Union[str, Path]) -> None:
    path_to_test = Path(path_to_test)
    results_file = Path(QC_RESULTS_PATH)
    error_counter = 0
    total_files = 0
    if results_file.exists():
        logger.info(f"Overwriting existing file: {results_file}")
        os.remove(results_file)

    rows: List[Dict[str, Any]] = []

    for prompt in os.listdir(path_to_test):
        prompt_folder = path_to_test / prompt

        for model in os.listdir(prompt_folder):
            model_path = prompt_folder / model

            if model.startswith("meta-llama"):
                subdirs = [dir.name for dir in model_path.iterdir() if dir.is_dir()]
                if len(subdirs) > 1:
                    logger.warning(
                        f"Multiple submodels found in {model_path}, using the first."
                    )
                only_model = subdirs[0]
                model_folder = model_path / only_model
                model = f"{model}/{only_model}"
            else:
                model_folder = model_path

            if CONSENSUS_SCORING:
                for temp in os.listdir(model_folder):
                    temp_folder = model_folder / temp
                    if not temp_folder.is_dir():
                        continue

                    for filename in os.listdir(temp_folder):
                        total_files += 1
                        file_path = temp_folder / filename

                        input_text, response, _ = process_llm_json_file(file_path)
                        # llm_entities = extract_entities_from_llm_text(response)

                        try:
                            json_response = json.loads(response)

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON from response: {e}")
                            error_counter += 1
                            continue

                        try:
                            llm_entities = (
                                extract_annotations_from_structured_llm_output(
                                    json_response
                                )
                            )
                        except KeyError as e:
                            logger.error(f"KeyError in response structure: {e}")
                            error_counter += 1
                            continue

                        #  exact_text_result = compare_annotated_to_original(input_text, response)
                        entities_result = find_one_valid_llm_entity(
                            llm_entities, input_text
                        )

                        # input_text, response, _ = process_llm_json_file(file_path)
                        # print(f"Response: {response}")
                        # llm_entities = json.loads(response)
                        entities_result = find_one_valid_llm_entity(
                            llm_entities, input_text
                        )

                        fully_valid, invalid, invalid_entities = (
                            define_quality_entities(llm_entities, input_text)
                        )

                        mol_fp = "; ".join(invalid_entities.get("MOL", []))
                        softn_fp = "; ".join(invalid_entities.get("SOFTN", []))
                        softv_fp = "; ".join(invalid_entities.get("SOFTV", []))
                        stime_fp = "; ".join(invalid_entities.get("STIME", []))
                        temp_fp = "; ".join(invalid_entities.get("TEMP", []))
                        ffm_fp = "; ".join(invalid_entities.get("FFM", []))

                        rows.append(
                            {
                                "prompt": prompt,
                                "model": model,
                                "temperature": temp,
                                "filename": filename,
                                "text_unchanged": None,
                                "one_entity_verified": entities_result,
                                "fully_valid": fully_valid,
                                "invalid": invalid,
                                "total_entities": fully_valid + invalid,
                                "full_path": str(file_path),
                                "MOL_FP": mol_fp,
                                "SOFTN_FP": softn_fp,
                                "SOFTV_FP": softv_fp,
                                "STIME_FP": stime_fp,
                                "TEMP_FP": temp_fp,
                                "FFM_FP": ffm_fp,
                            }
                        )
            else:
                for filename in os.listdir(model_folder):
                    total_files += 1
                    file_path = model_folder / filename

                    input_text, response, _ = process_llm_json_file(file_path)
                    # llm_entities = extract_entities_from_llm_text(response)

                    try:
                        json_response = json.loads(response)

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON from response: {e}")
                        error_counter += 1
                        continue

                    try:
                        llm_entities = extract_annotations_from_structured_llm_output(
                            json_response
                        )
                    except KeyError as e:
                        logger.error(f"KeyError in response structure: {e}")
                        error_counter += 1
                        continue

                    #  exact_text_result = compare_annotated_to_original(input_text, response)
                    entities_result = find_one_valid_llm_entity(
                        llm_entities, input_text
                    )

                    # input_text, response, _ = process_llm_json_file(file_path)
                    # print(f"Response: {response}")
                    # llm_entities = json.loads(response)
                    entities_result = find_one_valid_llm_entity(
                        llm_entities, input_text
                    )

                    fully_valid, invalid, invalid_entities = define_quality_entities(
                        llm_entities, input_text
                    )

                    mol_fp = "; ".join(invalid_entities.get("MOL", []))
                    softn_fp = "; ".join(invalid_entities.get("SOFTN", []))
                    softv_fp = "; ".join(invalid_entities.get("SOFTV", []))
                    stime_fp = "; ".join(invalid_entities.get("STIME", []))
                    temp_fp = "; ".join(invalid_entities.get("TEMP", []))
                    ffm_fp = "; ".join(invalid_entities.get("FFM", []))

                    rows.append(
                        {
                            "prompt": prompt,
                            "model": model,
                            "temperature": 1.0,
                            "filename": filename,
                            "text_unchanged": None,
                            "one_entity_verified": entities_result,
                            "fully_valid": fully_valid,
                            "invalid": invalid,
                            "total_entities": fully_valid + invalid,
                            "full_path": str(file_path),
                            "MOL_FP": mol_fp,
                            "SOFTN_FP": softn_fp,
                            "SOFTV_FP": softv_fp,
                            "STIME_FP": stime_fp,
                            "TEMP_FP": temp_fp,
                            "FFM_FP": ffm_fp,
                        }
                    )

    save_qc_results_to_csv(rows, QC_RESULTS_FOLDER)
    if error_counter > 0:
        logger.error(
            f"Encountered {error_counter} errors during processing of {total_files} files."
        )
    else:
        logger.info("All files processed successfully.")
    logger.success(f"Quality control completed. Results saved to {QC_RESULTS_PATH}")


# === Run script ===

if __name__ == "__main__":
    quality_control(LLM_ANNOTATIONS)
