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

import ast
import json
import re
import time
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import click
import pandas as pd
from loguru import logger
from openai.types.chat import ChatCompletion
from pydantic import ValidationError as PydanticValidationError

from mdner_llm.core.extract_entities import ListOfEntities, ListOfEntitiesPositions
from mdner_llm.core.logger import create_logger
from mdner_llm.utils.common import ensure_dir


def safe_json_load(value: object) -> object:
    """Safely parse JSON strings.

    If the input is not a string, it is returned as-is.
    If it is a string, the function attempts to parse it as JSON.
    If parsing fails, the original string is returned.

    Returns
    -------
    object
        The parsed JSON object if parsing is successful, otherwise the original value.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


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
    logger.info(f"Loading annotations from {annotations_dir}.")
    records = []
    # Iterate over all JSON files in the directory in sorted order
    for json_file in sorted(annotations_dir.glob("*.json")):
        try:
            # Open and parse the JSON file
            with json_file.open(encoding="utf-8") as handle:
                data = json.load(handle)
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


def extract_content(data: dict) -> str | None:
    """Extract message.content from serialized choices.

    Parameters
    ----------
    data : dict
        The dictionary containing the "choices" key with serialized content.

    Returns
    -------
    str | None
        The extracted content string if found, otherwise None.
    """
    # Check if "choices" key exists in the data
    if "choices" not in data:
        return None
    choices = data["choices"]
    if not choices or not isinstance(choices, list):
        return None
    choice = choices[0]
    # Use regex to extract the content string from the serialized choice
    match = re.search(r"content='(.*?)', ", choice, re.DOTALL)
    if not match:
        return None
    # Return the extracted content
    return ast.literal_eval(f"'{match.group(1)}'")


def normalize_input(
    obj: ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str | dict,
) -> ListOfEntities | ListOfEntitiesPositions | ChatCompletion | dict:
    """
    Normalize input into a consistent format for parsing.

    This function handles various input types that may be encountered when
    processing LLM responses, including:
    - Direct Pydantic model instances (ListOfEntities or ListOfEntitiesPositions)
    - ChatCompletion objects containing JSON text
    - Raw JSON strings
    - Python dictionaries representing JSON

    Parameters
    ----------
    obj : ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str | dict
        The object to normalize.

    Returns
    -------
    ListOfEntities | ListOfEntitiesPositions | ChatCompletion | dict
        The normalized object, ready for parsing.
    """
    # If the object is already a Pydantic model instance
    if isinstance(obj, (ListOfEntities, ListOfEntitiesPositions)):
        # return it as-is
        return obj

    # If the object is a dict
    if isinstance(obj, dict):
        # If it has an "entities" key
        if "entities" in obj:
            # we assume it's already in the expected format and return it
            return obj
        else:
            # otherwise, we try to extract content from it and parse as JSON
            content = extract_content(obj)
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content
            else:
                return None

    # If the object is a ChatCompletion
    if isinstance(obj, ChatCompletion):
        # extract the content and parse it as JSON
        content = obj.choices[0].message.content
        return json.loads(content) if content else {}
    # If the object is a string
    if isinstance(obj, str):
        # Attempt to parse the string as JSON
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return obj  # Return original string if it's not valid JSON
    # For any other type, return it as-is
    return obj


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
        obj = normalize_input(obj)
        if isinstance(obj, model_class):
            return obj

        if isinstance(obj, dict):
            model = model_class.model_validate(obj)
        else:
            return None

        if not getattr(model, "entities", None):
            return None
        else:
            return model

    except PydanticValidationError:
        # logger.warning(f"Validation error for {obj}: {exc}")
        return None


def parse_llm_and_groundtruth(df: pd.DataFrame) -> pd.DataFrame:
    """Parse LLM responses and ground truth columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing:
        - "raw_llm_response"
        - "groundtruth"

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed columns:
        - "llm_response"
        - "groundtruth"
    """
    logger.info("Parsing LLM responses and ground truth annotations.")
    df = df.copy()
    llm_responses = []
    groundtruths = []
    for row in df.itertuples(index=False):
        # Parse LLM response into a consistent format
        # This handles cases where the response is already a dict,
        # a ChatCompletion object, or a raw JSON string.
        # It normalizes all of these into a dict that can be parsed
        parsed_response = safe_json_load(row.raw_llm_response)
        llm_response = normalize_input(parsed_response)
        llm_responses.append(llm_response)
        # Parse ground truth into Pydantic model
        parsed_gt = safe_json_load(row.groundtruth)
        normalized_gt = parse_model(parsed_gt, model_class=ListOfEntities)
        groundtruths.append(normalized_gt)
    # Add parsed columns to the DataFrame
    df["llm_response"] = llm_responses
    df["groundtruth"] = groundtruths
    # Display first example for verification
    first_response = df["raw_llm_response"].iloc[0]
    logger.debug("First example:")
    logger.info(
        f"LLM response (Type: `{type(first_response)}` to `{type(llm_responses[0])}`):"
    )
    logger.debug(f"LLM response (raw: {type(first_response)}): {first_response}")
    logger.debug(f"LLM response (parsed: {type(llm_responses[0])}): {llm_responses[0]}")
    logger.info(f"Ground truth (Type: {type(groundtruths[0])})")
    logger.debug(f"Ground truth (parsed: {type(groundtruths[0])}): {groundtruths[0]}")
    logger.success("Parsed LLM responses and ground truth annotations successfully!")
    return df


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

    if response is None:
        return False

    try:
        # Validate JSON string against the appropriate Pydantic model
        if prompt_tag == "json":
            ListOfEntities.model_validate(response)
        else:
            ListOfEntitiesPositions.model_validate(response)
    except PydanticValidationError:
        return False  # Validation failed
    else:
        return True  # Validation succeeded


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
            entities_model = model_class.model_validate(content)
        else:
            entities_model = model_class.model_validate(response)
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


def add_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns of quality checks to the DataFrame.

    Adds two boolean columns:
    - `is_valid_output_format`:
        True if the LLM response matches the expected JSON format.
    - `has_no_hallucination`:
        True if all predicted entities appear in the original text.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the following columns:
        - "raw_llm_response": the raw output from the LLM
        - "text": the original text that was annotated
        - "prompt_tag": tag defining expected JSON format

    Returns
    -------
    pd.DataFrame
        DataFrame with additional quality check columns.
    """
    df = df.copy()
    is_valid_list = []
    has_no_hallu_list = []
    # Iterate row-wise to apply checks
    for row in df.itertuples(index=False):
        # Check output format validity
        is_valid_list.append(
            is_valid_output_format(
                row.llm_response,
                row.prompt_tag,
            )
        )
        # Check hallucination (only if format is valid)
        has_no_hallu_list.append(
            has_no_hallucination(
                row.llm_response,
                row.text,
                row.prompt_tag,
            )
        )
    # Add results as new columns
    df["is_valid_output_format"] = is_valid_list
    df["has_no_hallucination"] = has_no_hallu_list

    return df


def group_texts_by_label(entities: list) -> dict[str, list[str]]:
    """Group entity texts by their labels.

    Parameters
    ----------
    entities : list
        List of entities with "label" and "text" fields.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping labels to lists of texts.
    """
    grouped = defaultdict(list)
    for ent in entities:
        label = getattr(ent, "label", None)
        text = getattr(ent, "text", None)
        # Only group if both label and text are present and non-empty
        if label and text:
            grouped[label].append(text)
    return dict(grouped)


def build_label_level_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a label-level DataFrame from the original DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame with one row per annotation file.

    Returns
    -------
    pd.DataFrame
        Label-level DataFrame with one row per entity label
        for each line in the original DataFrame.
    """
    rows = []
    for _, row in df.iterrows():
        # Parse groundtruth
        gt_model = row["groundtruth"]
        gt_entities = gt_model.entities if gt_model is not None else []
        # Parse prediction
        pred_model = parse_model(
            row["llm_response"],
            model_class=ListOfEntities,
        )
        pred_entities = pred_model.entities if pred_model is not None else []
        # Group by label
        gt_by_label = group_texts_by_label(gt_entities)
        pred_by_label = group_texts_by_label(pred_entities)
        all_labels = set(gt_by_label) | set(pred_by_label)
        # Create one row per label with all relevant info and metrics
        for label in all_labels:
            # Start with the original row data
            new_row = row.to_dict()
            # Add label-specific info
            new_row.update(
                {
                    "label": label,
                    "groundtruth_by_label": gt_by_label.get(label, []),
                    "prediction_by_label": pred_by_label.get(label, []),
                }
            )
            rows.append(new_row)

    return pd.DataFrame(rows)


def compute_confusion_metrics_by_row(row):
    """Compute confusion metrics (TP, FP, FN) for a single row at entity level.

    Parameters
    ----------
    row : pd.Series
        A row of the DataFrame containing at least the following columns:
        - "groundtruth_by_label": list of ground-truth entity texts for the label
        - "prediction_by_label": list of predicted entity texts for the label

    Returns
    -------
    pd.Series
        A Series containing the following metrics:
        - "true_positives": count of correctly predicted entities
        - "false_positives": count of incorrectly predicted entities
        - "false_negatives": count of missed entities
        - "tp_entities": list of correctly predicted entity texts
        - "fp_entities": list of incorrectly predicted entity texts
        - "fn_entities": list of missed entity texts
    """
    # Convert lists of entities to sets
    gt = set(row.get("groundtruth_by_label", []))
    pred = set(row.get("prediction_by_label", []))
    # Compute true positives, false positives, and false negatives
    tp = gt & pred
    fp = pred - gt
    fn = gt - pred
    # Return metrics as a Series
    # to be added as new columns in the DataFrame
    return pd.Series(
        {
            "true_positives": len(tp),
            "false_positives": len(fp),
            "false_negatives": len(fn),
            "tp_entities": list(tp),
            "fp_entities": list(fp),
            "fn_entities": list(fn),
        }
    )


def compute_confusion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-row confusion metrics at entity level.

    Adds:
    - true_positives, false_positives, false_negatives
    - tp_entities, fp_entities, fn_entities

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the following columns:
        - "groundtruth_by_label": list of ground-truth entity texts for the label
        - "prediction_by_label": list of predicted entity texts for the label

    Returns
    -------
    pd.DataFrame
        DataFrame with confusion metrics added.
    """
    df = df.copy()
    metrics = df.apply(compute_confusion_metrics_by_row, axis=1)
    return pd.concat([df, metrics], axis=1)


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    """Safely divide two pandas Series.

    Parameters
    ----------
    a : pd.Series
        The numerator series.
    b : pd.Series
        The denominator series.

    Returns
    -------
    pd.Series
        The result of a / b,
        where division by zero is handled gracefully by returning NA.
    """
    return a / b.replace(0, pd.NA)


def _add_metrics(grouped: pd.DataFrame) -> pd.DataFrame:
    beta = 0.5
    # Extract confusion metrics
    tp = grouped["true_positives"]
    fp = grouped["false_positives"]
    fn = grouped["false_negatives"]
    # Compute precision, recall, F1, and F-beta scores
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
        - "is_valid_output_format": fraction of valid format responses
        - "pct_without_hallucination": fraction of responses without hallucinations
        - "true_positives", "false_positives", "false_negatives": summed per group
        - "precision", "recall", "f1", "fbeta_0.5": aggregated metrics
    """
    logger.info(
        "Computing evaluation metrics grouped by framework, model, and label..."
    )
    df = df.copy()
    group_cols = ["framework_name", "model_name", "prompt_tag"]
    # First, compute metrics per label by grouping by framework,
    # model, prompt_tag, and label
    grouped_label = (
        df.groupby([*group_cols, "label"])
        .agg(
            nb_of_texts_with_label=("text", "nunique"),
            nb_of_gt_entities_by_label=(
                "groundtruth_by_label",
                lambda s: s.apply(len).sum(),
            ),
            nb_of_pred_entities_by_label=(
                "prediction_by_label",
                lambda s: s.apply(len).sum(),
            ),
            is_valid_output_format=("is_valid_output_format", lambda s: 100 * s.mean()),
            has_no_hallucination=("has_no_hallucination", lambda s: 100 * s.mean()),
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
        )
        .reset_index()
    )
    # Compute overall metrics by grouping only by framework and model
    # Merging all labels together to get overall metrics per model
    overall = (
        df.groupby(group_cols)
        .agg(
            nb_of_texts_with_label=("text", "nunique"),
            is_valid_output_format=("is_valid_output_format", lambda s: 100 * s.mean()),
            has_no_hallucination=("has_no_hallucination", lambda s: 100 * s.mean()),
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
        )
        .reset_index()
    )
    overall["label"] = "OVERALL"
    # Compute metrics for both grouped DataFrames
    grouped_label = _add_metrics(grouped_label)
    overall = _add_metrics(overall)
    # Combine per-label and overall metrics into a single DataFrame
    grouped = pd.concat([grouped_label, overall], ignore_index=True)
    logger.success(
        f"Computed metrics per model and per label for {len(grouped)} "
        "samples successfully!"
    )
    return grouped


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


def save_df_to_parquet(
    df,
    path: Path,
) -> None:
    """Serialize all columns before saving to parquet."""
    df_serialized = df.copy()

    for col in df_serialized.columns:
        df_serialized[col] = df_serialized[col].map(serialize_response)

    df_serialized.to_parquet(
        path,
        index=False,
    )


def save_grouped_stats_to_excel(
    df: pd.DataFrame,
    output_path: Path,
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
    output_path : Path
        Path where the Excel file will be saved.

    Returns
    -------
    Path
        Path to the generated Excel file.
    """
    metrics_map = {
        "nb_of_texts_with_label": "Number of Texts with Label",
        "nb_of_gt_entities_by_label": "Number of Ground Truth Entities",
        "nb_of_pred_entities_by_label": "Number of Predicted Entities",
        "is_valid_output_format": "Is correct Output Format",
        "has_no_hallucination": "Has no Hallucination",
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
    df_pivot.to_excel(output_path, index=True)

    logger.success(
        f"Grouped evaluation statistics saved successfully to: {output_path}"
    )
    return output_path


def main(annotations_dir: Path, results_dir: Path) -> None:
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
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    logger = create_logger(f"logs/eval_llm_and_framework_{timestamp}.log")
    logger.info("Starting LLM annotation evaluation.")
    start_time = time.perf_counter()
    # Loading annotations with metadatas
    df = load_json_annotations_as_dataframe(annotations_dir)
    # Parsing LLM responses and ground truth into structured formats
    df = parse_llm_and_groundtruth(df)
    # Checking that the output format is correct
    # and the absence of hallucination
    df = add_quality_columns(df)
    # Build label-level dataset
    df_label = build_label_level_dataframe(df)
    # Compute confusion metrics (TP, FP, TN) by annotation file and label
    df_with_conf_metrics = compute_confusion_metrics(df_label)
    # Save the detailed evaluation results DataFrame to a Parquet file
    save_df_to_parquet(
        df_with_conf_metrics,
        results_dir / f"per_text_metrics_{timestamp}.parquet",
    )
    # Compute grouped stats by model and framework
    df_grouped_stats = compute_grouped_stats(df_with_conf_metrics)
    # Saving into an excel
    save_grouped_stats_to_excel(
        df=df_grouped_stats,
        output_path=results_dir / f"overall_metrics_{timestamp}.xlsx",
    )
    elapsed_time = int(time.perf_counter() - start_time)
    logger.success(f"Evaluation duration: {timedelta(seconds=elapsed_time)} 🎉")


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
def run_main_from_cli(
    annotations_dir: Path,
    results_dir: Path,
) -> None:
    """
    Evaluate the quality of JSON entity annotations from CLI.

    Parameters
    ----------
    annotations_dir : Path
        Directory containing the JSON annotation files to evaluate.
    results_dir : Path
        Directory where evaluation results, logs, and reports will be written.
    """
    main(
        annotations_dir=annotations_dir,
        results_dir=results_dir,
    )


# MAIN PROGRAM
if __name__ == "__main__":
    # Evaluate json annotations through all models
    run_main_from_cli()
