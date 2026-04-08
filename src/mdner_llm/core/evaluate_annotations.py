"""
Evaluate and benchmark annotations produced by several LLMs on molecular-dynamics texts.

Evaluation includes:

1. **JSON format validity**
   The response must match the expected schema:
   `{"entities": [ {"label": <str>, "text": <str>, ...}, ... ]}`
   (and optionally character positions when using `json_with_positions`).

2. **Hallucination detection**
  Each extracted entity must correspond to text actually present in the source document.

3. **Annotation correctness**
   Compares LLM-predicted entities with ground-truth expert annotations,
   computing metris like True Positive, False Positive, False Negative,
   Precision, Recall, F1, F_beta_0.5.


Outputs:
--------
- Per-annotation **Parquet files** with detailed metrics for each model and framework.
- Aggregated **Excel (.xlsx) summary** combining results across models and frameworks,
reporting format adherence, hallucination rate, and annotation accuracy.


Requirements:
-------------
- Annotation files must exist in `--annotations-dir`. These are typically generated
by running `extract_entities_all_texts.py` for several models/frameworks.
"""

import json
import re
import time
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from loguru import logger
from openai.types.chat import ChatCompletion
from pydantic import ValidationError as PydanticValidationError

from mdner_llm.core.extract_entities import ListOfEntities, ListOfEntitiesPositions
from mdner_llm.core.logger import create_logger
from mdner_llm.utils.common import ensure_dir


def load_json_annotations_as_dataframe(annotations_dir: Path) -> pd.DataFrame:
    """
    Load JSON annotation files into a DataFrame.

    Each row corresponds to one JSON file, and each column corresponds
    to a top-level key found in the JSON objects.

    Parameters
    ----------
    annotations_dir : Path
        Directory containing JSON annotation files.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per file and JSON keys as columns.
    """
    logger.info(f"Loading annotations from {annotations_dir}...")
    records = []
    # Iterate over all JSON files in the directory in sorted order
    for json_file in sorted(annotations_dir.glob("*.json")):
        try:
            # Open and parse the JSON file
            with json_file.open(encoding="utf-8") as handle:
                data: dict[str, object] = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.warning(f"Skipping invalid JSON file {json_file.name}: {exc}")
            continue

        # Add the source filename as metadata
        data["response_metadata"] = json_file.name
        # Store the parsed JSON object
        records.append(data)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame.from_records(records)
    logger.success(
        f"Loaded {df.shape[0]} annotation files into DataFrame successfully!"
    )
    return df


def parse_model(
    obj: ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str | dict,
    model_class: type[ListOfEntities | ListOfEntitiesPositions],
) -> ListOfEntities | ListOfEntitiesPositions | None:
    """
    Parse an object into a Pydantic model instance containing `.entities`.

    Supports:
    - Direct Pydantic model instance
    - ChatCompletion object
    - JSON string
    - Python dict representing JSON

    Parameters
    ----------
    obj : ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str | dict
        The object to parse into a Pydantic model.
    model_class : type[ListOfEntities | ListOfEntitiesPositions]
        The Pydantic model class to use for parsing.

    Returns
    -------
    ListOfEntities | ListOfEntitiesPositions | None
        The parsed model instance with an `.entities` attribute, or `None`
        if parsing fails or if the result has no `.entities`.
    """
    try:
        # If the object is already an instance of the target model, reuse it
        if isinstance(obj, model_class):
            entities_model = obj

        # If the object is a ChatCompletion
        elif isinstance(obj, ChatCompletion):
            # Extract the message content
            content = obj.choices[0].message.content
            # And validate it as JSON against the Pydantic model
            entities_model = model_class.model_validate_json(content)

        # Otherwise, treat the input as JSON data (string or dict)
        else:
            json_str = obj if isinstance(obj, str) else json.dumps(obj)
            entities_model = model_class.model_validate_json(json_str)

    # Catch validation, type, or JSON-related errors
    except (PydanticValidationError, ValueError, TypeError):
        return None

    # Ensure the parsed model exposes an `entities` attribute
    if not hasattr(entities_model, "entities"):
        return None

    return entities_model


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _group_by_label(entities):
    grouped = defaultdict(set)
    for e in entities:
        grouped[e.label].add(normalize_text(e.text))
    return grouped


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, df.columns.map(lambda x: isinstance(x, str))]
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def compute_confusion_metrics(
    df: pd.DataFrame,
    results_dir: Path,
    pred_col: str = "raw_llm_response",
    gt_col: str = "groundtruth",
    text_col: str = "text",
    prompt_tag_col: str = "prompt_tag",
    beta: float = 0.5,
) -> pd.DataFrame:
    """
    Compute confusion matrix metrics per annotation file.

    Metrics are computed at entity level using exact matching
    on tuples (label, normalized_text).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing predictions, groundtruth, original text,
        and prompt tag.
    results_dir : Path
        Directory where the Excel file will be saved.
    pred_col : str
        Column name containing LLM responses.
    gt_col : str
        Column name containing ground-truth annotations.
    text_col : str
        Column name containing the original text that was annotated.
    prompt_tag_col : str
        Column name defining expected JSON format.
    beta : float
        Beta value for F-beta score.

    Returns
    -------
    pd.DataFrame
        DataFrame with confusion metrics and quality flags added.
    """
    logger.info("Computing evaluation metrics per annotation...")

    def _compute_row(row: pd.Series) -> pd.Series:
        # Extract row info
        response = row[pred_col]
        groundtruth = row[gt_col]
        original_text = row[text_col]
        prompt_tag = row[prompt_tag_col]
        file = row["response_metadata"]

        # Check format and hallucinations
        is_format_valid = is_valid_output_format(response, prompt_tag)
        no_hallucination = has_no_hallucination(
            response,
            original_text,
            prompt_tag,
        )

        # Determine model type based on prompt
        model_class = (
            ListOfEntities if prompt_tag == "json" else ListOfEntitiesPositions
        )
        # Parse prediction and groundtruth
        pred_model = parse_model(response, model_class)
        gt_model = parse_model(groundtruth, model_class)

        if pred_model is None or gt_model is None:
            # Skip metrics if parsing fails
            reason = (
                "prediction parsing failed."
                if pred_model is None
                else "groundtruth parsing failed."
            )
            logger.warning(f"[{file}] Metrics skipped: {reason}")

            return pd.Series(
                {
                    "is_format_valid": is_format_valid,
                    "has_no_hallucination": no_hallucination,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision_of_annotation": 0.0,
                    "recall_of_annotation": 0.0,
                    "f1_of_annotation": 0.0,
                    f"fbeta_{beta}_of_annotation": 0.0,
                }
            )
        preds_by_label = _group_by_label(pred_model.entities)
        gt_by_label = _group_by_label(gt_model.entities)
        all_labels = set(preds_by_label) | set(gt_by_label)

        rows = []
        for label in all_labels:
            pred_set = preds_by_label.get(label, set())
            gt_set = gt_by_label.get(label, set())

            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)

            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            f1 = _safe_div(2 * precision * recall, precision + recall)
            fbeta = _safe_div(
                (1 + beta**2) * precision * recall,
                (beta**2 * precision) + recall,
            )

            logger.info(f"[{file}]:")
            logger.info(f"Is correct output format: {is_format_valid}")
            logger.info(f"Is without hallucination: {no_hallucination}")
            logger.info(f"Evaluating label: {label}")
            logger.debug(f"Predictions ({len(pred_set)}): {sorted(pred_set)}")
            logger.debug(f"Groundtruth ({len(gt_set)}): {sorted(gt_set)}")
            logger.info(f"Confusion counts: TP={tp} | FP={fp} | FN={fn}")
            logger.info(
                f"Evaluation metrics: precision={precision:.3f} | "
                f"recall={recall:.3f} | f1={f1:.3f} | "
                f"fbeta_{beta}={fbeta:.3f}\n"
            )

            rows.append(
                {
                    "response_metadata": file,
                    "label": label,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    f"fbeta_{beta}": fbeta,
                    "is_format_valid": is_format_valid,
                    "has_no_hallucination": no_hallucination,
                }
            )

        return pd.DataFrame(rows)

    # Apply row-wise computation
    metrics = pd.concat(df.apply(_compute_row, axis=1).to_list(), ignore_index=True)
    res_df = pd.concat([df, metrics], axis=1)
    res_df = clean_dataframe(res_df)

    # Saving metrics by text into a parquet file
    parquet_path = (
        results_dir
        / f"per_text_metrics_{datetime.now(UTC).strftime('%Y-%m-%dT%H-%M-%S')}.parquet"
    )
    res_df.to_parquet(parquet_path, index=False)

    logger.success(
        f"Saved metrics computation for each files in {parquet_path} successfully!\n"
    )
    return res_df


def _extract_json_string(
    response: ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str,
) -> str | None:
    """
    Extract the JSON string from a model response.

    Parameters
    ----------
    response : ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str
        The raw response from the model. It can be:
        - a Pydantic model (`ListOfEntities` or `ListOfEntitiesPositions`),
        - a `ChatCompletion` object containing JSON text,
        - or a raw JSON string.

    Returns
    -------
    str | None
        The extracted JSON string if available, otherwise `None`.
    """
    # If response is already a string, return it
    if isinstance(response, str):
        return response

    # If response is a ChatCompletion object
    if isinstance(response, ChatCompletion):
        choices = getattr(response, "choices", None)
        if not choices:  # no choices available
            return None

        message = getattr(choices[0], "message", None)
        if message is None:  # message missing
            return None

        # Return the content of the first message
        return getattr(message, "content", None)

    # For other types (e.g., Pydantic models), return None
    return None


def _validate_json_string(response_str: str, prompt_tag: str) -> bool:
    """
    Validate a JSON string against the expected Pydantic model.

    Parameters
    ----------
    response_str : str
        JSON string to validate.
    prompt_tag : str
        Tag specifying the expected JSON format:
        - "json" for `ListOfEntities`
        - "json_with_positions" for `ListOfEntitiesPositions`.

    Returns
    -------
    bool
        True if the JSON string is valid according to the expected format,
        False otherwise.
    """
    try:
        # Validate JSON string against the appropriate Pydantic model
        if prompt_tag == "json":
            ListOfEntities.model_validate_json(response_str)
        else:
            ListOfEntitiesPositions.model_validate_json(response_str)
    except PydanticValidationError:
        return False  # Validation failed
    else:
        return True  # Validation succeeded


def is_valid_output_format(
    response: ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str,
    prompt_tag: str,
) -> bool:
    """
    Check whether the model response is valid according to the expected output format.

    Parameters
    ----------
    response : Any
        The raw model response:
        - a Pydantic model (ListOfEntities or ListOfEntitiesPositions),
        - a ChatCompletion object containing JSON text,
        - or a raw JSON string.
    prompt_tag : str
        Tag defining expected JSON format ('json' or 'json_with_positions').

    Returns
    -------
    bool
        True if the response is valid, False otherwise.
    """
    # Case 1: Already a Pydantic instance
    if (isinstance(response, ListOfEntities) and prompt_tag == "json") or (
        isinstance(response, ListOfEntitiesPositions)
        and prompt_tag == "json_with_positions"
    ):
        return True

    # Case 2: Extract JSON string
    response_str = _extract_json_string(response)
    if response_str is None:
        return False

    return _validate_json_string(response_str, prompt_tag)


def has_no_hallucination(
    response: ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str,
    original_text: str,
    prompt_tag: str = "json",
) -> bool:
    """
    Check that all predicted entities appear in the original text.

    Parameters
    ----------
    response : ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str
        The validated model response or raw JSON string.
    original_text : str
        The text that was annotated.
    prompt_tag : str
        Tag defining expected JSON format ("json" or "json_with_positions").

    Returns
    -------
    bool
        True if no predicted entity is missing from the original text.
    """
    if not is_valid_output_format(response, prompt_tag):
        return False
    # Select model class
    model_class = ListOfEntities if prompt_tag == "json" else ListOfEntitiesPositions

    # Parse response into entities_model
    try:
        if isinstance(response, model_class):
            entities_model = response
        elif isinstance(response, ChatCompletion):
            content = response.choices[0].message.content
            entities_model = model_class.model_validate_json(content)
        else:
            entities_model = model_class.model_validate_json(response)
    except PydanticValidationError:
        return False

    # Ensure entity list exists
    if not hasattr(entities_model, "entities"):
        return False

    # Normalize text once
    norm_text = normalize_text(original_text)

    # Check each entity actually appears in original text
    for entity in entities_model.entities:
        if not entity.text:
            return False
        if normalize_text(entity.text) not in norm_text:
            return False

    return True


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """
    Perform element-wise division with explicit zero-denominator handling.

    Parameters
    ----------
    num : pd.Series
        Numerator values.
    den : pd.Series
        Denominator values.

    Returns
    -------
    pd.Series
        Result of num / den where den > 0, otherwise 0.0,
        rounded to 3 decimal places.
    """
    # Divide where denominator > 0, else set result to 0
    result = np.where(den > 0, num / den, 0.0)
    # Round the results to 3 decimal places
    return np.round(result, 3)


def compute_label_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute global metrics per label (micro-averaged)."""
    grouped = (
        df.groupby("label")
        .agg(
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
        )
        .reset_index()
    )

    tp = grouped["true_positives"]
    fp = grouped["false_positives"]
    fn = grouped["false_negatives"]

    grouped["precision"] = safe_divide(tp, tp + fp)
    grouped["recall"] = safe_divide(tp, tp + fn)

    grouped["f1"] = safe_divide(
        2 * grouped["precision"] * grouped["recall"],
        grouped["precision"] + grouped["recall"],
    )

    beta = 0.5
    grouped[f"fbeta_{beta}"] = safe_divide(
        (1 + beta**2) * grouped["precision"] * grouped["recall"],
        beta**2 * grouped["precision"] + grouped["recall"],
    )

    return grouped


def compute_overall_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute global micro metrics across all labels."""
    tp = df["true_positives"].sum()
    fp = df["false_positives"].sum()
    fn = df["false_negatives"].sum()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    beta = 0.5
    fbeta = _safe_div(
        (1 + beta**2) * precision * recall,
        (beta**2 * precision) + recall,
    )

    return pd.DataFrame(
        [
            {
                "label": "overall",
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                f"fbeta_{beta}": fbeta,
            }
        ]
    )


def _add_metrics(grouped: pd.DataFrame) -> pd.DataFrame:
    beta = 0.5

    tp = grouped["true_positives"]
    fp = grouped["false_positives"]
    fn = grouped["false_negatives"]

    grouped["precision"] = safe_divide(tp, tp + fp)
    grouped["recall"] = safe_divide(tp, tp + fn)

    grouped["f1"] = safe_divide(
        2 * grouped["precision"] * grouped["recall"],
        grouped["precision"] + grouped["recall"],
    )

    grouped[f"fbeta_{beta}"] = safe_divide(
        (1 + beta**2) * grouped["precision"] * grouped["recall"],
        beta**2 * grouped["precision"] + grouped["recall"],
    )

    return grouped


def compute_grouped_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute evaluation statistics grouped by framework and model.

    For each combination of `framework_name` and `model_name`, this function:
    - Computes the number of annotations
    - Aggregates true positives, false positives, and false negatives
    - Computes precision, recall, F1 score, and F-beta score (β=0.5 by default)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the following columns:
        - "framework_name": name of the annotation framework
        - "model_name": LLM model name
        - "raw_llm_response": raw LLM output
        - "text_to_annotate": original text that was annotated
        - "prompt_tag": tag specifying expected JSON format
            ("json" or "json_with_positions")
        - "true_positives", "false_positives", "false_negatives": counts per row

    Returns
    -------
    pd.DataFrame
        Aggregated metrics per framework and model with the following columns:
        - "framework_name", "model_name"
        - "nb_annotations": number of annotations
        - "pct_is_format_valid": fraction of valid format responses
        - "pct_without_hallucination": fraction of responses without hallucinations
        - "true_positives", "false_positives", "false_negatives": summed per group
        - "precision", "recall", "f1", "fbeta_0.5": aggregated metrics
    """
    logger.info(
        "Computing evaluation metrics grouped by framework, model, and label..."
    )
    df = df.copy()

    group_cols = ["framework_name", "model_name", "prompt_tag"]

    grouped_label = (
        df.groupby([*group_cols, "label"])
        .agg(
            nb_of_texts_with_label=("text", "nunique"),
            pct_is_format_valid=("is_format_valid", lambda s: 100 * s.mean()),
            pct_has_no_hallucination=("has_no_hallucination", lambda s: 100 * s.mean()),
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
        )
        .reset_index()
    )

    overall = (
        df.groupby(group_cols)
        .agg(
            nb_of_texts_with_label=("text", "nunique"),
            pct_is_format_valid=("is_format_valid", lambda s: 100 * s.mean()),
            pct_has_no_hallucination=("has_no_hallucination", lambda s: 100 * s.mean()),
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
        )
        .reset_index()
    )

    overall["label"] = "OVERALL"
    grouped_label = _add_metrics(grouped_label)
    overall = _add_metrics(overall)

    return pd.concat([grouped_label, overall], ignore_index=True)


def normalize_text(text: str) -> str:
    """Normalize text by removing special characters and converting to lowercase.

    Parameters
    ----------
    text : str
        The text to normalize.

    Returns
    -------
    str
        The normalized text.
    """
    # Normalize unicode characters
    text_normalized = unicodedata.normalize("NFKD", text)
    # Convert to lowercase
    text_normalized = text_normalized.lower()
    # Remove extra whitespace
    text_normalized = re.sub(r"\s+", " ", text_normalized)
    # Strip leading and trailing whitespace
    return text_normalized.strip()


def serialize_response(resp: Any) -> str:
    """
    Serialize various response objects into a JSON-safe string representation.

    Parameters
    ----------
    resp : Any
        The object to serialize. This may be a string, a custom class instance,
        or a model response object such as ChatCompletion.

    Returns
    -------
    str
        A JSON-compatible string representation of the input object.
    """
    # If it's already a string, nothing to do.
    if isinstance(resp, str):
        return resp

    # If it's a ListOfEntities or ListOfEntitiesPositions object
    if isinstance(resp, (ListOfEntities, ListOfEntitiesPositions)):
        return resp.model_dump_json(indent=2)

    # Specific handling for ChatCompletion-like objects
    if isinstance(resp, ChatCompletion):
        return json.dumps(resp.__dict__, default=str)

    return str(resp)


def save_grouped_stats_to_excel(
    df: pd.DataFrame,
    results_dir: Path,
    filename_prefix: str = "evaluation_summary",
) -> Path:
    """
    Save grouped evaluation statistics to an Excel file with MultiIndex columns.

    The output Excel file is structured with one row per model and
    MultiIndex columns of the form:
        (framework_name, metric)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by `compute_grouped_stats`.
    results_dir : Path
        Directory where the Excel file will be saved.
    filename_prefix : str
        Prefix used for the output Excel filename.
        Default: evaluation_summary.

    Returns
    -------
    Path
        Path to the generated Excel file.
    """
    metrics_map = {
        "nb_of_texts_with_label": "Number of Texts with Label",
        "pct_is_format_valid": "Is correct Output Format",
        "pct_has_no_hallucination": "Has no Hallucination",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "fbeta_0.5": "F-beta 0.5",
    }
    metrics = list(metrics_map.keys())

    # Pivot to include framework in columns
    df_pivot = df.pivot_table(
        index=["model_name", "label"],
        columns="framework_name",
        values=metrics,
        aggfunc="first",  # or "mean"
    )

    # Rename columns: metric -> label lisible, framework -> title case
    df_pivot.columns = pd.MultiIndex.from_tuples(
        [(fw.replace("_", " ").title(), metrics_map[m]) for m, fw in df_pivot.columns]
    )

    # Order metrics inside each framework
    ordered_tuples = [
        (fw, m)
        for fw in df_pivot.columns.get_level_values(0).unique()
        for m in metrics_map.values()
        if (fw, m) in df_pivot.columns
    ]
    df_pivot = df_pivot.loc[:, ordered_tuples]

    # Save Excel
    output_path = (
        results_dir
        / f"{filename_prefix}_{datetime.now(UTC).strftime('%Y-%m-%dT%H-%M-%S')}.xlsx"
    )
    df_pivot.to_excel(output_path, index=True)

    logger.success(
        f"Grouped evaluation statistics saved successfully to: {output_path}\n"
    )

    return output_path


@click.command()
@click.option(
    "--annotations-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("results/llm/annotations"),
    show_default=True,
    help="Directory containing the JSON annotation files to evaluate.",
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("results/llm/evaluation_stats"),
    show_default=True,
    help="Target directory where evaluation results will be saved.",
    callback=ensure_dir,
)
def main(
    annotations_dir: Path,
    results_dir: Path,
) -> None:
    """
    Evaluate the quality of JSON entity annotations.

    Parameters
    ----------
    annotations_dir : Path
        Directory containing the JSON annotation files to evaluate.
    results_dir : Path
        Directory where evaluation results, logs, and reports will be written.
    """
    # Configure logging
    logger = create_logger(
        f"logs/evaluate_annotations_{datetime.now(UTC).strftime('%Y-%m-%dT%H-%M-%S')}.log"
    )
    logger.info("Starting LLM annotation evaluation...")
    start_time = time.perf_counter()
    # Loading annotations with metadatas
    df = load_json_annotations_as_dataframe(annotations_dir)
    # Compute confusion metrics (TP, FP, TN) by text
    df_with_conf_metrics = compute_confusion_metrics(df, results_dir)
    # Compute confusion metrics (TP, FP, TN) grouped by framework and model
    df_grouped_stats = compute_grouped_stats(df_with_conf_metrics)
    # Saving into an excel
    save_grouped_stats_to_excel(df=df_grouped_stats, results_dir=results_dir)
    elapsed_time = int(time.perf_counter() - start_time)
    logger.success(f"Evaluation duration: {timedelta(seconds=elapsed_time)} 🎉")


# MAIN PROGRAM
if __name__ == "__main__":
    # Evaluate json annotations through all models
    main()
