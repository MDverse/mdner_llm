"""
Evaluate and benchmark annotations produced by several LLMs on molecular-dynamics texts.

This script loads the *N most recent* annotation files from `annotations/v2` and
evaluates how well different language models (e.g., GPT-4, Gemini, MoonshotAI Kimik2,
Qwen, Meta Llama 3.1/3.3) extract structured entities from MD descriptions.

Each model is tested under four configurations:
- `no_validation`
- `validation_instructor`
- `validation_llamaindex`
- `validation_pydanticai`

For every model output, the script checks:
1. **JSON format validity**
   The response must match the expected schema:
   `{"entities": [ {"label": <str>, "text": <str>, ...}, ... ]}`
   (and optionally character positions when using `json_with_positions`).

2. **Hallucination detection**
  Each extracted entity must correspond to text actually present in the source document.

3. **Annotation correctness**
   The model’s entities are compared against expert-validated annotations.
   Each prediction is tagged as correct or incorrect.

All detailed per-response results for each model are saved as **Parquet files**.A final
**Excel (.xlsx)** summary aggregates statistics across models, enabling comparisons on:
- format adherence,
- hallucination rate,
- annotation accuracy.

This tool is designed to benchmark LLM reliability when producing structured,
domain-specific annotations.

Usage:
=======
    uv run src/evaluate_json_annotations.py [--annotations-dir PATH]
                                            [--nb-annotations INT]
                                            [--tag-prompt 'json|json_with_positions']
                                            [--results-dir PATH]

Arguments:
==========
    --annotations-dir: PATH
        Directory containing the annotation JSON files to evaluate.
        The script automatically selects the most recent samples.
        Default: "annotations/v2"

    --nb-annotations: INT
        Number of recent annotation files to process.
        Default: 10

    --tag-prompt: STR
        Output format expected from the model:
        - "json": only label and text
        - "json_with_positions": label, text, start, end
        Default: "json"

    --results-dir: PATH
        Directory where all evaluation outputs (Parquet files+summary Excel) are saved.
        Default: "results/json_evaluation_stats/<timestamp>"

Example:
========
    uv run src/evaluate_json_annotations.py \
        --annotations-dir annotations/v2 \
        --results-dir results/json_evaluation_stats/test

This command will evaluate the 50 most recent annotation files found in
`annotations/v2`, run all models on them using the JSON prompt format, and save
all per-model parquet files plus a global XLSX summary inside
`results/json_evaluation_stats/test`.
"""  # noqa: RUF002

# METADATAS
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "AGPL-3.0 license"
__date__ = "2025"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import unicodedata

sys.path.append(str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
from loguru import logger
from openai.types.chat import ChatCompletion
from pydantic import ValidationError as PydanticValidationError
from tqdm import tqdm

# UTILITY IMPORTS
from models.pydantic_output_models import ListOfEntities, ListOfEntitiesPositions


# FUNCTIONS
def setup_logger(loguru_logger: Any, log_dir: str | Path = "logs") -> None:
    """Configure a Loguru logger to write logs into a rotating daily log file.

    Parameters
    ----------
    loguru_logger : Any
        A Loguru logger instance (typically `loguru.logger`).
    log_dir : str or Path, optional
        Directory where log files will be stored. Default is "logs".
    """
    # Ensure log directory exists
    log_folder = Path(log_dir)
    log_folder.mkdir(parents=True, exist_ok=True)
    # Reset any previous configuration
    loguru_logger.remove()
    # Define log format
    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss}"
        "| <level>{level:<8}</level> "
        "| <level>{message}</level>"
    )
    loguru_logger.add(
        log_folder / "evaluate_json_annotations_{time:YYYY-MM-DD}.log",
        format=fmt,
        level="DEBUG",
    )
    loguru_logger.add(
        sys.stdout,
        format=fmt,
        level="DEBUG",
    )


def ensure_dir(ctx, param, value: Path) -> Path:
    """
    Create the directory if it does not already exist.

    Callback for Click options to ensure the provided path
    is a valid directory. Behaves like `mkdir -p`.

    Parameters
    ----------
    ctx : click.Context
        The Click context for the current command invocation.
        (Required by Click callbacks but unused in this function.)
    param : click.Parameter
        The Click parameter associated with this callback.
        (Required by Click callbacks but unused in this function.)
    value : Path
        The directory path provided by the user, already converted
        into a `pathlib.Path` object by Click.

    Returns
    -------
    Path
        The same path, after ensuring the directory exists.
    """
    value.mkdir(parents=True, exist_ok=True)
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
    logger.info(f"Loading annotations from {annotations_dir}...")
    records: list[dict[str, object]] = []

    for json_file in sorted(annotations_dir.glob("*.json")):
        try:
            with json_file.open(encoding="utf-8") as handle:
                data: dict[str, object] = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.warning(f"Skipping invalid JSON file {json_file.name}: {exc}")
            continue

        data["__file__"] = json_file.name
        records.append(data)

    df = pd.DataFrame.from_records(records)
    logger.success(
        f"Loaded {df.shape[0]} annotation files into DataFrame successfully! \n")

    return pd.DataFrame.from_records(records)


def add_raw_text_column(df: pd.DataFrame,
    text_file_col: str = "text_file"
) -> pd.DataFrame:
    """
    Extract 'raw_text' from JSON into a new column'text_to_annotate' in the Df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column with paths to JSON files.
    text_file_col : str
        Name of the column containing paths to JSON files.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column 'text_to_annotate' containing
        the raw text from each JSON file.
    """
    logger.info("Adding the text to annotate in the dataframe...")
    raw_texts: list[str] = []

    for _idx, json_path in enumerate(df[text_file_col]):
        path = Path(json_path)
        if not path.exists():
            logger.warning(f"JSON file not found: {json_path}")
            raw_texts.append("")
            continue

        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            raw_text = data.get("raw_text", "")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to read {json_path}: {exc}")
            raw_text = ""

        raw_texts.append(raw_text)

    df = df.copy()
    df["text_to_annotate"] = raw_texts
    logger.success(f"Added 'text_to_annotate' column for {len(df)} files successfully! \n")
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
        if isinstance(obj, model_class):
            entities_model = obj
        elif isinstance(obj, ChatCompletion):
            content = obj.choices[0].message.content
            entities_model = model_class.model_validate_json(content)
        else:  # str or dict
            json_str = obj if isinstance(obj, str) else json.dumps(obj)
            entities_model = model_class.model_validate_json(json_str)
    except (PydanticValidationError, ValueError, TypeError):
        return None

    if not hasattr(entities_model, "entities"):
        return None

    return entities_model


def compute_confusion_metrics(
    df: pd.DataFrame,
    pred_col: str = "raw_llm_response",
    gt_col: str = "groundtruth",
    text_col: str = "text_to_annotate",
    prompt_tag_col: str = "tag_prompt",
    beta: float = 0.5,
) -> pd.DataFrame:
    """
    Compute confusion matrix metrics per annotation file.

    Metrics are computed at entity level using exact matching
    on tuples (label, normalized_text).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing predictions, groundtruth, original text, and prompt tag.
    pred_col : str
        Column name containing LLM responses (various types supported).
    gt_col : str
        Column name containing ground-truth annotations.
    text_col : str
        Column name containing the original text that was annotated.
    prompt_tag_col : str
        Column name defining expected JSON format ("json" or "json_with_positions").
    beta : float
        Beta value for F-beta score (default: 0.5).

    Returns
    -------
    pd.DataFrame
        DataFrame with confusion metrics and derived scores added.
    """
    logger.info("Computing evaluation metrics per annotation...")

    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    def _compute_row(row: pd.Series) -> pd.Series:
        response = row[pred_col]
        groundtruth = row[gt_col]
        prompt_tag = row[prompt_tag_col]

        # Select correct model class
        model_class = ListOfEntities if prompt_tag == "json" else ListOfEntitiesPositions

        # Parse both response and groundtruth in one line using a list comprehension
        pred_model, gt_model = [parse_model(o, model_class) for o in (response, groundtruth)]
        # Handle missing models
        if pred_model is None or gt_model is None:
            return pd.Series(
                {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    f"fbeta_{beta}": 0.0,
                }
            )

        # Create sets of (label, normalized_text)
        preds_set = {(e.label, normalize_text(e.text)) for e in pred_model.entities}
        gt_set = {(e.label, normalize_text(e.text)) for e in gt_model.entities}

        tp = len(preds_set & gt_set)
        fp = len(preds_set - gt_set)
        fn = len(gt_set - preds_set)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        fbeta = _safe_div((1 + beta**2) * precision * recall, (beta**2 * precision) + recall)

        return pd.Series(
            {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                f"fbeta_{beta}": fbeta,
            }
        )

    metrics = df.apply(_compute_row, axis=1)
    res_df = pd.concat([df, metrics], axis=1)
    mean_metrics = {
        "Precision": res_df["precision"].mean(),
        "Recall": res_df["recall"].mean(),
        "F1": res_df["f1"].mean(),
        f"F{beta}": res_df[f"fbeta_{beta}"].mean(),
    }
    logger.debug(
        f"\n{'Metric':>10} | {'Mean':>6}\n"
        f"{'-' * 20}\n" + "\n".join(f"{name:>10} | {value:6.3f}"
        for name, value in mean_metrics.items()))
    logger.success(f"Completed metrics computation for {len(res_df)} files!\n")
    return res_df


def is_valid_output_format(
    response: ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str,
    prompt_tag: str
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
    # Case 1: Already a Pydantic instance matching the prompt_tag
    if isinstance(response, ListOfEntities) and prompt_tag == "json":
        return True
    if (isinstance(response, ListOfEntitiesPositions)
        and prompt_tag == "json_with_positions"):
        return True

    # Case 2: Extract JSON string if response is ChatCompletion or str
    response_str = None
    if isinstance(response, ChatCompletion):
        try:
            choices = getattr(response, "choices", None)
            if not choices or len(choices) == 0:
                return False

            message = getattr(choices[0], "message", None)
            if message is None:
                return False

            response_str = getattr(message, "content", None)
            if response_str is None:
                return False

        except Exception:
            return False

    elif isinstance(response, str):
        response_str = response
    # If we have a JSON string
    if response_str is not None:
        try:
            if prompt_tag == "json":
                ListOfEntities.model_validate_json(response_str)
            else:
                ListOfEntitiesPositions.model_validate_json(response_str)
            return True
        except PydanticValidationError:
            return False

    # Case 3: Not a recognized response type
    return False


def has_no_hallucination(
    response: ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str,
    original_text: str,
    prompt_tag: str = "json"
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


def compute_grouped_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute evaluation statistics grouped by framework and model.

    For each combination of `framework_name` and `model_name`, this function:
    - Computes the number of annotations
    - Calculates the fraction of valid LLM responses
    - Calculates the fraction of responses without hallucinations
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
        - "pct_valid": fraction of valid responses
        - "pct_no_hallucination": fraction of responses without hallucinations
        - "true_positives", "false_positives", "false_negatives": summed per group
        - "precision", "recall", "f1", "fbeta_0.5": aggregated metrics
    """
    df = df.copy()

    # Apply row-level checks for validity and hallucinations
    def _row_flags(row: pd.Series) -> pd.Series:
        response = row["raw_llm_response"]
        original_text = row["text_to_annotate"]
        prompt_tag = row.get("prompt_tag", "json")

        valid = is_valid_output_format(response, prompt_tag)
        no_halluc = has_no_hallucination(response, original_text, prompt_tag)

        return pd.Series({
            "is_valid": valid,
            "no_hallucination": no_halluc,
        })

    flags = df.apply(_row_flags, axis=1)
    df = pd.concat([df, flags], axis=1)

    # Aggregate per framework and per model
    grouped = (
        df.groupby(["framework_name", "model_name"])
        .agg(
            nb_annotations=("raw_llm_response", "count"),
            pct_valid=("is_valid", "mean"),
            pct_no_hallucination=("no_hallucination", "mean"),
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
        )
        .reset_index()
    )

    # Compute precision, recall, F1, F-beta per group
    beta = 0.5
    grouped["precision"] = grouped["true_positives"] / (
        grouped["true_positives"] + grouped["false_positives"] + 1e-8
    )
    grouped["recall"] = grouped["true_positives"] / (
        grouped["true_positives"] + grouped["false_negatives"] + 1e-8
    )
    grouped["f1"] = 2 * (grouped["precision"] * grouped["recall"]) / (
        grouped["precision"] + grouped["recall"] + 1e-8
    )
    grouped[f"fbeta_{beta}"] = (
        (1 + beta**2) * grouped["precision"] * grouped["recall"] /
        (beta**2 * grouped["precision"] + grouped["recall"] + 1e-8)
    )

    return grouped


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
    text_normalized = text_normalized.strip()
    return text_normalized


def compute_entity_match_percent(
    response: ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str,
    groundtruth: ListOfEntities | ListOfEntitiesPositions,
    original_text: str,
    prompt_tag: str,
) -> tuple[int, int, float]:
    """
    Compute correct entities, total groundtruth entities, and their % in the response.

    Parameters
    ----------
    response : ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str
        The validated model response or raw JSON string.
    groundtruth : ListOfEntities | ListOfEntitiesPositions
        The reference annotation.
    original_text : str
        The text that was annotated.
    prompt_tag : str
        Tag defining expected JSON format ("json" or "json_with_positions").

    Returns
    -------
    tuple[int, int, float]
        A tuple containing:
        - nb_correct_entities (int): number of correctly predicted entities
        - nb_groundtruth_entities (int): number of entities in the groundtruth
        - percent (float): percentage of groundtruth entities present in the
        response (0 to 100)
    """
    # Select correct model class
    model_class = ListOfEntities if prompt_tag == "json" else ListOfEntitiesPositions

    # Parse response into a model instance
    try:
        if isinstance(response, model_class):
            entities_model = response
        elif isinstance(response, ChatCompletion):
            content = response.choices[0].message.content
            entities_model = model_class.model_validate_json(content)
        else:  # str JSON
            entities_model = model_class.model_validate_json(response)
    except (PydanticValidationError, ValueError, TypeError):
        nb_groundtruth_entities = len({normalize_text(e.text) for e in groundtruth.entities})
        return 0, nb_groundtruth_entities, 0.0

    if not hasattr(entities_model, "entities"):
        nb_groundtruth_entities = len({normalize_text(e.text) for e in groundtruth.entities})
        return 0, nb_groundtruth_entities, 0.0

    # Case 1 : text-only entities
    if prompt_tag == "json":
        gt_texts = {normalize_text(e.text) for e in groundtruth.entities}
        response_texts = {normalize_text(e.text) for e in entities_model.entities}
        matched = gt_texts & response_texts

    else:
        # Case 2 : entities with positions
        gt_texts = {
            (normalize_text(e.text), getattr(e, "start", None), getattr(e, "end", None))
            for e in groundtruth.entities
        }
        response_texts = {
            (normalize_text(e.text), getattr(e, "start", None), getattr(e, "end", None))
            for e in entities_model.entities
        }
        matched = gt_texts & response_texts

    nb_correct_entities = len(matched)
    nb_groundtruth_entities = len(gt_texts)
    if (is_valid_output_format(response, prompt_tag) 
        or has_no_hallucination(response, original_text, prompt_tag)):
        percent = round(nb_correct_entities / max(nb_groundtruth_entities, 1) * 100, 3)
    else:
        percent = 0.0

    logger.debug(f"Response = {response_texts}")
    logger.debug(f"Groundtruth = {gt_texts}")
    logger.debug(f"Correct entities = {nb_correct_entities} / \
                 {nb_groundtruth_entities}")
    logger.debug(f"Correctness percent = {percent} %")
    return nb_correct_entities, nb_groundtruth_entities, percent


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


def append_annotation_result(
    df: pd.DataFrame,
    model_name: str,
    provider: str,
    validator: str,
    prompt_tag: str,
    text_to_annotate: str,
    json_path: Path,
    model_response: dict[str, ListOfEntities | ListOfEntitiesPositions],
    groundtruth: ListOfEntities | ListOfEntitiesPositions,
) -> pd.DataFrame:
    """
    Evaluate a model response and append a result row to the evaluation DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Existing dataframe where the new row will be added.
    model_name : str
        Name of the LLM used.
    provider : str
        Backend provider (OpenAI or OpenRouter).
    validator : str
        Name of the validation method ('no_validation', 'instructor',
        'llamaindex', 'pydanticai').
    prompt_tag : str
        Tag defining expected JSON format ('json' or 'json_with_positions').
    text_to_annotate : str
        Original text provided to the model.
    json_path : Path
        Path to the groundtruth annotation file.
    model_response : dict
        The model prediction processed by `annotate()`.
    groundtruth : ListOfEntities | ListOfEntitiesPositions
        Expert annotations used to check correctness.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with the appended row.
    """
    # Evaluation of the model's response
    is_correct_output_format = is_valid_output_format(model_response, prompt_tag)
    is_without_hallucination = has_no_hallucination(model_response, text_to_annotate, prompt_tag)
    nb_correct_entities, nb_groundtruth_entities, correctness_percent = (
        compute_entity_match_percent(
        model_response, groundtruth, text_to_annotate, prompt_tag)
    )

    # Append the row
    new_row = {
        "model_name": model_name,
        "provider": provider,
        "validator": validator,
        "prompt": prompt_tag,
        "text_to_annotate": text_to_annotate,
        "json_path": str(json_path),
        "model_response": serialize_response(model_response),
        "nb_correct_entities": nb_correct_entities,
        "groundtruth": serialize_response(groundtruth),
        "nb_groundtruth_entities": nb_groundtruth_entities,
        "is_correct_output_format": is_correct_output_format,
        "is_without_hallucination": is_without_hallucination,
        "correctness_percent": correctness_percent,
    }

    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


def summarize_model_stats(parquet_path: Path) -> dict[str, dict[str, float]]:
    """
    Compute aggregated evaluation statistics per validator from a Parquet results file.

    This function loads a Parquet file containing model evaluation annotations and
    computes, for each validator, the percentage of:
    - correctly formatted outputs,
    - outputs without hallucination,
    - correct answers.

    Percentages are returned as floats in the range 0-100, rounded to one decimal place.

    Parameters
    ----------
    parquet_path : Path
        Path to the Parquet file containing evaluation results. The file must include
        the following columns:
        - ``validator``
        - ``is_correct_output_format``
        - ``is_without_hallucination``
        - ``correctness_percent``.

    Returns
    -------
    dict[str, dict[str, float]]
        A mapping where each key is a validator name, and the corresponding value is
        a dictionary with the aggregated metrics:
        ``{
            "correct_format": float,
            "no_hallucination": float,
            "correct_answer": float,
        }
    """
    df = pd.read_parquet(parquet_path)

    results = {}

    for validator in df["validator"].unique():
        sub = df[df["validator"] == validator]

        total = len(sub)
        if total == 0:
            results[validator] = {
                "correct_format": 0.0,
                "no_hallucination": 0.0,
                "correct_answer": 0.0,
            }
            continue

        # Sum of correct entities in llm response and in groundtruth
        total_correct = sub["nb_correct_entities"].sum()
        total_groundtruth = sub["nb_groundtruth_entities"].sum()
        # Percent of global correct_answer
        correct_answer_percent = round(
            (total_correct / max(total_groundtruth, 1)) * 100, 1
)
        results[validator] = {
            "correct_format": round(100 * sub["is_correct_output_format"].mean(), 1),
            "no_hallucination": round(100 * sub["is_without_hallucination"].mean(), 1),
            "correct_answer": correct_answer_percent
,
        }

    return results


def save_evaluation_results(
    all_summary_rows: list[list],
    annotations_count: int,
    results_dir: Path,
    tag_prompt: str
) -> None:
    """
    Save evaluation results to an Excel file with multi-index columns.

    Parameters
    ----------
    all_summary_rows : List[list]
        Rows containing summary statistics for each model and validator.
    annotations_count : int
        Number of annotations evaluated (used in the filename).
    results_dir : Path
        Directory where the Excel file will be saved.
    tag_prompt: str
        Descriptor indicating the format of the expected LLM output.
    """
    # Create a simple DataFrame first
    df_simple = pd.DataFrame(
        all_summary_rows,
        columns=[
            "Model (Provider)",

            "nof_val_correct_format",
            "nof_val_no_hallu",
            "nof_val_correct_answer",

            "instr_correct_format",
            "instr_no_hallu",
            "instr_correct_answer",

            "llama_correct_format",
            "llama_no_hallu",
            "llama_correct_answer",

            "pyd_correct_format",
            "pyd_no_hallu",
            "pyd_correct_answer",
        ]
    )

    # Create MultiIndex for nicer Excel formatting
    prefix = "JSON" if tag_prompt == "json" else "JSON_POSITIONS"
    methods = [
        prefix + " without format validation",
        prefix + " + Instructor",
        prefix + " + LlamaIndex",
        prefix + " + PydanticAI"
    ]
    subcols = [
        "Correct output format (%)",
        "Without hallucination (%)",
        "Correct answer (%)"
    ]
    multi_columns = pd.MultiIndex.from_tuples(
        [(method, sub) for method in methods for sub in subcols]
    )
    df_results = pd.DataFrame(
        df_simple.drop(columns=["Model (Provider)"]).values,
        columns=multi_columns,
        index=df_simple["Model (Provider)"]
    )
    # Save to Excel
    path = results_dir / f"evaluation_summary_{annotations_count}" \
                        f"_annotations_{results_dir.name}.xlsx"
    df_results.to_excel(path, index=True)
    logger.success(f"Evaluation stats saved to: {path} successfully! \n")


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

    Returns
    -------
    Path
        Path to the generated Excel file.
    """
    metrics = [
        "nb_annotations",
        "pct_valid",
        "pct_no_hallucination",
        "precision",
        "recall",
        "f1",
        "fbeta_0.5",
    ]

    # Pivot to MultiIndex columns: (framework, metric)
    df_pivot = (
        df.set_index(["model_name", "framework_name"])[metrics]
        .unstack("framework_name")
    )

    # Reorder column levels: (framework, metric)
    df_pivot.columns = df_pivot.columns.swaplevel(0, 1)
    df_pivot = df_pivot.sort_index(axis=1, level=0)

    output_path = results_dir / f"{filename_prefix}_{results_dir.name}.xlsx"
    df_pivot.to_excel(output_path, index=True)

    logger.success(
        f"Grouped evaluation statistics saved successfully to: {output_path}\n"
    )

    return output_path


@click.command()
@click.option(
    "--annotations-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("results/llm_annotations"),
    show_default=True,
    help="Directory containing the JSON annotation files to evaluate."
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path(f"results/json_evaluation_stats/{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    show_default=True,
    help="Target directory where evaluation results will be saved.",
    callback=ensure_dir
)
def evaluate_json_annotations(
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
    setup_logger(logger, results_dir)
    logger.info("Starting evaluation of JSON annotation outputs...")
    logger.debug(f"Annotations directory: {annotations_dir}")
    logger.debug(f"Results directory: {results_dir}")

    # Loading annotations with metadatas
    df = load_json_annotations_as_dataframe(annotations_dir)
    # Adding the text to annotate into the df
    df_with_text = add_raw_text_column(df)
    #df_with_text.to_parquet("annotations_summary.parquet", index=False)

    # Compute confusion metrics (TP, FP, TN, FN)
    df_with_conf_metrics = compute_confusion_metrics(df_with_text)
    #df_with_conf_metrics.to_parquet("annotations_summary_with_metrics.parquet", index=False)

    df_grouped_stats = compute_grouped_stats(df_with_conf_metrics)

    save_grouped_stats_to_excel(
        df=df_grouped_stats,
        results_dir=results_dir,
    )
    """# Retrieve MD annotated texts with their verified annotations
    annotations = load_interesting_annotations(
        annotations_dir,
        nb_annotations,
        tag_prompt,
    )

    # Loop through LLMs
    for model_name in MODELS_OPENROUTER:
        logger.info(
            f"=================== 🤖 Evaluating model: {model_name} ==================="
        )
        provider = "OpenRouter" if model_name in MODELS_OPENROUTER else "OpenAI"

        # Create an empty dataframe for this model
        eval_df = pd.DataFrame(
            columns=[
                "model_name",
                "provider",
                "validator",
                "prompt",
                "text_to_annotate",
                "json_path",
                "model_response",
                "nb_correct_entities",
                "groundtruth",
                "nb_groundtruth_entities",
                "is_correct_output_format",
                "is_without_hallucination",
                "correctness_percent",
            ]
        )

        annotation_times = {
            "no_validation": 0.0,
            "instructor": 0.0,
            "llamaindex": 0.0,
            "pydanticai": 0.0
        }

        # Loop through MD texts to annotate
        for record in tqdm(
                annotations,
                total=len(annotations),
                desc="Processing texts",
                unit="text",
                ncols=100,
                colour="blue",
                leave=True
            ):
            file_path = record["file_path"]
            text_to_annotate = record["text_to_annotate"]
            groundtruth = record["groundtruth"]

            # ------------------------------------------------------
            # 1. Annotation without validation
            # ------------------------------------------------------
            start = time.time()
            response_no_val = annotate(
                                text_to_annotate,
                                model_name,
                                instructor_clients[model_name],
                                tag_prompt,
                                validation=False)
            annotation_times["no_validation"] += time.time() - start
            eval_df = append_annotation_result(
                eval_df,
                model_name=model_name,
                provider=provider,
                validator="no_validation",
                prompt_tag=tag_prompt,
                text_to_annotate=text_to_annotate,
                json_path=file_path,
                model_response=response_no_val,
                groundtruth=groundtruth,
            )

            # ------------------------------------------------------
            # 2. Annotation with INSTRUCTOR validation
            # ------------------------------------------------------
            start = time.time()
            response_instructor_val = annotate(
                                text_to_annotate,
                                model_name,
                                instructor_clients[model_name],
                                tag_prompt,
                                validation=True,
                                validator="instructor")
            annotation_times["instructor"] += time.time() - start

            eval_df = append_annotation_result(
                eval_df,
                model_name=model_name,
                provider=provider,
                validator="instructor",
                prompt_tag=tag_prompt,
                text_to_annotate=text_to_annotate,
                json_path=file_path,
                model_response=response_instructor_val,
                groundtruth=groundtruth,
            )

            # ------------------------------------------------------
            # 3. Annotation with LLAMAINDEX validation
            # ------------------------------------------------------
            start = time.time()
            response_llamaindex_val = annotate(
                                text_to_annotate,
                                model_name,
                                llama_clients[model_name],
                                tag_prompt,
                                validation=True,
                                validator="llamaindex")
            annotation_times["llamaindex"] += time.time() - start
            eval_df = append_annotation_result(
                eval_df,
                model_name=model_name,
                provider=provider,
                validator="llamaindex",
                prompt_tag=tag_prompt,
                text_to_annotate=text_to_annotate,
                json_path=file_path,
                model_response=response_llamaindex_val,
                groundtruth=groundtruth,
            )
            # ------------------------------------------------------
            # 4. Annotation with PyDANTICAI validation
            # ------------------------------------------------------
            start = time.time()
            response_pydanticai_val = annotate(
                                text_to_annotate,
                                model_name,
                                py_clients[model_name],
                                tag_prompt,
                                validation=True,
                                validator="pydanticai")
            annotation_times["pydanticai"] += time.time() - start
            eval_df = append_annotation_result(
                eval_df,
                model_name=model_name,
                provider=provider,
                validator="pydanticai",
                prompt_tag=tag_prompt,
                text_to_annotate=text_to_annotate,
                json_path=file_path,
                model_response=response_pydanticai_val,
                groundtruth=groundtruth,
            )

        # Save model's evaluation
        model_out_path = results_dir / f"{model_name.split("/")[1]}_{len(annotations)}" \
                                        "_annotations_stats.parquet"
        eval_df.to_parquet(model_out_path, index=False)

        model_key = f"{model_name} ({provider})"
        stats = summarize_model_stats(model_out_path)
        # Add row with 12 values
        # (4 validator with 3 things to check : format, hallucination, correctness)
        all_summary_rows.append([
            model_key,

            stats["no_validation"]["correct_format"],
            stats["no_validation"]["no_hallucination"],
            stats["no_validation"]["correct_answer"],

            stats["instructor"]["correct_format"],
            stats["instructor"]["no_hallucination"],
            stats["instructor"]["correct_answer"],

            # None, None, None,   # llamaindex
            # None, None, None,   # pydanticai

            stats["llamaindex"]["correct_format"],
            stats["llamaindex"]["no_hallucination"],
            stats["llamaindex"]["correct_answer"],

            stats["pydanticai"]["correct_format"],
            stats["pydanticai"]["no_hallucination"],
            stats["pydanticai"]["correct_answer"],
        ])

        for validator in ["no_validation", "instructor", "llamaindex", "pydanticai"]:
            logger.debug(
                f"Validator: {validator} | "
                f"Correct format: {stats[validator]["correct_format"]}% | "
                f"No hallucination: {stats[validator]["no_hallucination"]}% | "
                f"Correct answer: {stats[validator]["correct_answer"]}%"
            )
        for validator, total_time in annotation_times.items():
            avg_time = total_time / len(annotations)
            logger.debug(
                f"{validator}: total={total_time:.2f}s | avg={avg_time:.2f}s per text"
            )

    # Save summary stats for each models to xlsx
    save_evaluation_results(all_summary_rows, len(annotations), results_dir, tag_prompt)"""


# MAIN PROGRAM
if __name__ == "__main__":
    # Evaluate json annotations through all models
    evaluate_json_annotations()
