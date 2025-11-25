# run_scoring_analysis.py

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from loguru import logger


# ======================================================================================
# Configuration
# ======================================================================================

DATE_TIME_STR = input(
    "Enter the date and time string to analyse (YYYY-MM-DD_HH-MM-SS): "
)
ANNOTATIONS_FOLDER = "annotations/"
TAGS = ["MOL", "SOFTN", "SOFTV", "STIME", "TEMP", "FFM"]

QC_RESULTS_PATH = f"llm_outputs/{DATE_TIME_STR}/stats/quality_control_results.csv"
SCORE_RESULTS_FOLDER = f"llm_outputs/{DATE_TIME_STR}/stats/"
SCORE_RESULTS_PATH = os.path.join(SCORE_RESULTS_FOLDER, "scoring_results.csv")


# ======================================================================================
# Entity extraction/formatting functions
# ======================================================================================


def extract_entities_from_annotation(text: str, entities: list) -> Dict[str, List[str]]:
    """
    Extract entities from the ground-truth annotated text.
    Will extract entites in the format:
    {
        "MOL": ["entity1", "entity2"],
        "SOFTN": ["entity1", "entity2"],
        "SOFTV": ["entity1"],
        "STIME": ["entity1", "entity2", "entity3"],
        "TEMP": [],
        "FFM": ["entity1"]
    }
    """
    result = {key: [] for key in TAGS}
    for start, end, entity_type in entities:
        # Ground truth currently uses SOFT
        # instead of SOFTN and  SOFTV
        # so we convert SOFT to SOFTN
        if entity_type == "SOFT":
            entity_type = "SOFTN"
        if entity_type in result:
            result[entity_type].append(text[start:end])
    return result


def extract_annotations_from_structured_llm_output(
    llm_output: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Extract entities from the LLM structured response.
    Will extract entites in the format:
    {
        "MOL": ["entity1", "entity2"],
        "SOFTN": ["entity1", "entity2"],
        "SOFTV": ["entity1"],
        "STIME": ["entity1", "entity2", "entity3"],
        "TEMP": [],
        "FFM": ["entity1"]
    }

    Args:
        llm_output (Dict[str, Any]): The structured output from the LLM.

    Returns:
        Dict[str, List[str]]: A dictionary with tags as keys and lists of
        entities as values.
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


# This function is to be used only for XML-style annotations.
# def extract_entities_from_llm_text(text: str) -> Dict[str, List[str]]:
#     """
#     Extract entities from the LLM annotated text (XML-style annotation).
#     Will extract entites in the format:
#     {
#         "MOL": ["entity1", "entity2"],
#         "SOFTN": ["entity1", "entity2"],
#         "SOFTV": ["entity1"],
#         "STIME": ["entity1", "entity2", "entity3"],
#         "TEMP": [],
#         "FFM": ["entity1"]
#     }
#     """
#     result = {tag: [] for tag in TAGS}
#     pattern = re.compile(r"<([A-Z]+)>(.*?)</\1>")
#     for tag, content in pattern.findall(text):
#         if tag in result:
#             result[tag].append(content.strip())
#     return result


# ======================================================================================
# Scoring
# ======================================================================================


def exact_match_score(
    ground_truth: Dict[str, List[str]],
    predictions: Dict[str, List[str]]
    ) -> Tuple[int, int, float]:
    """_summary_

    Args:
        ground_truth (Dict[str, List[str]]): ground-truth entities
        predictions (Dict[str, List[str]]): predicted LLM entities

    Returns:
        Tuple[int, int, float]: Returns three values:
            - matched: Number of entities that match exactly between
            ground truth and predictions.
            - total: Total number of ground truth entities.
            - score_ratio: The ratio of matched entities to
            total ground truth entities.
    """
    match_count = 0
    total_items = 0

    for key in ground_truth:
        gt_items = set(ground_truth[key])             # Remove duplicates
        pred_items = set(predictions.get(key, []))    # Remove duplicates

        match_count += len(gt_items & pred_items)     # Count set intersection
        total_items += len(gt_items)

    score = match_count / total_items if total_items > 0 else 0.0

    return match_count, total_items, score



def false_positives(
    ground_truth: Dict[str, List[str]],
    predictions: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
    """
    For each tag, return the list of predicted entities that aren’t in the ground truth
    (false positives). This takes into account the entities that exist in the ground truth
    (textually) but are not marked as actual entities + hallucinated entities.

    Args:
        ground_truth (Dict[str, List[str]]): The ground truth entities.
        predictions (Dict[str, List[str]]): The predicted entities.
    
    Returns:
        Dict[str, List[str]]: A dictionary where each key is a tag and the value
        is a list of entities that were predicted but not found in the ground truth.
    """
    return {
        tag: list(
            set(predictions.get(tag, [])) - set(ground_truth.get(tag, []))
        )
        for tag in TAGS
    }


def false_negatives(
    ground_truth: Dict[str, List[str]],
    predictions: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
    """
    For each tag, return the list of ground truth entities that aren’t in the predictions
    (missed entities).

    Args:
        ground_truth (Dict[str, List[str]]): The ground truth entities.
        predictions (Dict[str, List[str]]): The predicted entities.
    
    Returns:
        Dict[str, List[str]]: A dictionary where each key is a tag and the value
        is a list of entities that were in the ground truth but not found
        in the LLM predictions.
    """
    return {
        tag: list(
            set(ground_truth.get(tag, [])) - set(predictions.get(tag, []))
        )
        for tag in TAGS
    }


def per_type_breakdown(
    gt: Dict[str, List[str]],
    pred: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
    """Compute per-tag breakdown of correct predictions.

    For each tag in the global TAGS list, this function compares the predicted entities
    to the ground truth entities and calculates:
        - The number of exact matches (entities correctly predicted).
        - The total number of ground truth entities for that tag.

    Args:
        gt (Dict[str, List[str]]): Ground truth entities, grouped by tag.
        pred (Dict[str, List[str]]): Predicted entities, grouped by tag.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary where each key is a tag and the value is
        a dictionary with:
            - 'exact_matches': (int) number of correctly predicted entities.
            - 'total_gt': (int) total number of ground truth entities for that tag.
    """
    breakdown = {}
    for tag in TAGS:
        gt_set = set(gt.get(tag, []))
        pred_set = set(pred.get(tag, []))
        correct = len(gt_set & pred_set)
        total = len(gt_set)
        breakdown[tag] = {
            "exact_matches": correct,
            "total_gt": total,
        }
    return breakdown


# ======================================================================================
# File processing functions
# ======================================================================================


def extract_annotations_to_score(
    csv_file: Union[str, Path],
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Extract the annotations to score from the Quality Control results CSV file.
    The selction is based on the "one_entity_verified" == True column.
    """
    df = pd.read_csv(csv_file)
    filtered = df[df["one_entity_verified"]]
    return (
        filtered[["prompt", "model", "filename", "full_path"]],
        filtered["filename"].tolist(),
        filtered["full_path"].tolist(),
    )


def process_json_file(json_file: Union[str, Path]) -> Tuple[str, List]:
    """
    Process the JSON ground-truth file and return the input text and the entities.
    ann[0] is the input text and ann[1]["entities"] is the list of entities.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    ann = data["annotations"][0]
    return ann[0], ann[1]["entities"]


def process_llm_json_file(json_file: Union[str, Path]) -> Tuple[str, str, str]:
    """
    Process the LLM JSON file and return the input text, the response, and the model.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["text_to_annotate"], data["response"], data["model"]


def save_scoring_results_to_csv(
    rows: List[Dict[str, Any]],
    output_dir: Union[str, Path]
    ) -> None:
    """
    Save the scoring results to a CSV file.
    If the file already exists, append the new rows to it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scoring_results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, mode="a", header=not csv_path.exists())


# ======================================================================================
# Main Logic
# ======================================================================================


def score_annotations():
    logger.info("Starting scoring analysis...")

    # If the scoring file already exists, remove it and create a new one
    if Path(SCORE_RESULTS_PATH).exists():
        logger.warning(f"Overwriting previous scoring file: {SCORE_RESULTS_PATH}")
        os.remove(SCORE_RESULTS_PATH)

    # Extract the list of annotation files and their metadata from the QC results
    # df contains columns: ["prompt", "model", "filename", "full_path"]
    df, filenames, llm_paths = extract_annotations_to_score(QC_RESULTS_PATH)

    # Prepend annotation folder path to each ground truth filename
    filenames = [os.path.join(ANNOTATIONS_FOLDER, name) for name in filenames]

    # Iterate over each pair of ground truth and LLM-generated annotation files
    for i, (gt_file, llm_file) in enumerate(zip(filenames, llm_paths, strict=False)):

        # Load and process ground truth annotations
        input_text, gt_entities = process_json_file(gt_file)
        gt_extracted = extract_entities_from_annotation(input_text, gt_entities)

        # Load and process LLM annotations
        _, llm_response, _ = process_llm_json_file(llm_file)
        # llm_extracted = extract_entities_from_llm_text(llm_response)
        # llm_extracted = json.loads(llm_response)
        llm_extracted = extract_annotations_from_structured_llm_output(
            json.loads(llm_response)
        )

        # Compute exact match metrics between GT and prediction
        matched, total, score_ratio = exact_match_score(gt_extracted, llm_extracted)

        # Compute false positives, false negatives, and per-type stats
        fps = false_positives(gt_extracted, llm_extracted)
        fns = false_negatives(gt_extracted, llm_extracted)
        breakdown = per_type_breakdown(gt_extracted, llm_extracted)

        prompt_name, model, filename, file_path = df.iloc[i]

        # Build a scoring result row with general metrics
        row = {
            "prompt": prompt_name,
            "model": model,
            "filename": filename,
            "percentage_correct": round(score_ratio * 100, 2),
            "total_correct": matched,
            "total": total,
            "total_fp": sum(len(v) for v in fps.values()),
            "full path": str(file_path),
        }

        # Add per-tag metrics: correct, total, false positives, and false negatives
        for tag in TAGS:
            row.update(
                {
                    f"{tag}_correct": breakdown[tag]["exact_matches"],
                    f"{tag}_total": breakdown[tag]["total_gt"],
                    f"{tag}_FP": "; ".join(fps[tag]),
                    f"{tag}_FN": "; ".join(fns[tag]),
                }
            )

        # Log progress and accuracy for current file
        logger.info(
            f"[{i + 1}/{len(df)}] Scored {filename:^25} | {prompt_name:^13} | {model:>20}"
        )

        # Save current row to the scoring results CSV
        save_scoring_results_to_csv([row], SCORE_RESULTS_FOLDER)

    logger.success(f"Scoring analysis complete. Results saved to {SCORE_RESULTS_PATH}")


if __name__ == "__main__":
    score_annotations()
