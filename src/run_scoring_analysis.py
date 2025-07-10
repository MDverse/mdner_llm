# run_scoring_analysis.py

import json
import os
import re
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


def extract_entities_from_llm_text(text: str) -> Dict[str, List[str]]:
    """
    Extract entities from the LLM annotated text.
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
    result = {tag: [] for tag in TAGS}
    pattern = re.compile(r"<([A-Z]+)>(.*?)</\1>")
    for tag, content in pattern.findall(text):
        if tag in result:
            result[tag].append(content.strip())
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


# ======================================================================================
# Scoring
# ======================================================================================


def exact_match_score(
    gt: Dict[str, List[str]], pred: Dict[str, List[str]]
) -> Tuple[int, int, float]:
    """ """
    matched = sum(1 for k in gt for e in gt[k] if e in set(pred.get(k, [])))
    total = sum(len(v) for v in gt.values())
    return matched, total, matched / total if total > 0 else 0


def detection_ratio(
    gt: Dict[str, List[str]], pred: Dict[str, List[str]]
) -> Dict[str, float]:
    return {
        k: sum(1 for e in gt[k] if e in set(pred.get(k, []))) / len(gt[k])
        if gt[k]
        else None
        for k in TAGS
    }


def false_positives(
    ground_truth: Dict[str, List[str]], predictions: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    For each tag, return the list of predicted entities that aren’t in the ground truth
    (incorrectly predicted entities)
    """
    return {
        tag: [
            entity
            for entity in predictions.get(tag, [])
            if entity not in set(ground_truth.get(tag, []))
        ]
        for tag in TAGS
    }


def false_negatives(
    ground_truth: Dict[str, List[str]], predictions: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    For each tag, return the list of ground truth entities that aren’t in the predictions
    (missed entities)
    """
    return {
        tag: [
            item
            for item in ground_truth.get(tag, [])
            if item not in set(predictions.get(tag, []))
        ]
        for tag in TAGS
    }


def per_type_breakdown(
    gt: Dict[str, List[str]], pred: Dict[str, List[str]]
) -> Dict[str, Dict[str, Any]]:
    breakdown = {}
    for tag in TAGS:
        gt_set = set(gt.get(tag, []))
        pred_set = set(pred.get(tag, []))
        correct = len(gt_set & pred_set)
        total = len(gt_set)
        breakdown[tag] = {
            "exact_matches": correct,
            "total_gt": total,
            "detection_ratio": correct / total if total > 0 else None,
            "false_positives": len(pred_set - gt_set),
            "false_negatives": len(gt_set - pred_set),
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
    with open(json_file, "r") as f:
        data = json.load(f)
    ann = data["annotations"][0]
    return ann[0], ann[1]["entities"]


def process_llm_json_file(json_file: Union[str, Path]) -> Tuple[str, str, str]:
    """
    Process the LLM JSON file and return the input text, the response, and the model.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    return data["text_to_annotate"], data["response"], data["model"]


def save_scoring_results_to_csv(
    rows: List[Dict[str, Any]], output_dir: Union[str, Path]
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

    # We retrieve the annotations to score from the QC results
    # df is a dataframe with the columns: ["prompt", "model", "filename", "full_path"]
    df, filenames, llm_paths = extract_annotations_to_score(QC_RESULTS_PATH)
    filenames = [os.path.join(ANNOTATIONS_FOLDER, name) for name in filenames]

    for i, (gt_file, llm_file) in enumerate(zip(filenames, llm_paths, strict=False)):
        input_text, gt_entities = process_json_file(gt_file)
        gt_extracted = extract_entities_from_annotation(input_text, gt_entities)

        _, llm_response, _ = process_llm_json_file(llm_file)
        # llm_extracted = extract_entities_from_llm_text(llm_response)
        # llm_extracted = json.loads(llm_response)
        llm_extracted = extract_annotations_from_structured_llm_output(
            json.loads(llm_response)
        )

        matched, total, score_ratio = exact_match_score(gt_extracted, llm_extracted)
        fps = false_positives(gt_extracted, llm_extracted)
        fns = false_negatives(gt_extracted, llm_extracted)
        breakdown = per_type_breakdown(gt_extracted, llm_extracted)

        prompt_name, model, filename, file_path = df.iloc[i]

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

        for tag in TAGS:
            row.update(
                {
                    f"{tag}_correct": breakdown[tag]["exact_matches"],
                    f"{tag}_total": breakdown[tag]["total_gt"],
                    f"{tag}_FP": "; ".join(fps[tag]),
                    f"{tag}_FN": "; ".join(fns[tag]),
                }
            )

        logger.info(
            f"[{i + 1}/{len(df)}] Scored {filename:^25} | {prompt_name:^13} | {model:>20} — {row['percentage_correct']}% accuracy"
        )
        save_scoring_results_to_csv([row], SCORE_RESULTS_FOLDER)

    logger.success(f"Scoring analysis complete. Results saved to {SCORE_RESULTS_PATH}")


if __name__ == "__main__":
    score_annotations()
