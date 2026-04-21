"""Evaluate annotations produced by several LLMs/Gliner models on MD texts."""

import json
import re
import time
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import ValidationError as PydanticValidationError

from mdner_llm.common import ensure_dir, sanitize_filename
from mdner_llm.logger import create_logger
from mdner_llm.models.entities import ListOfEntities


def load_json_annotations_as_dataframe(annotations_dir: Path) -> pd.DataFrame:
    """
    Load JSON annotation files into a DataFrame.

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

        # Parse specific fields with Pydantic
        for key in ("formatted_response", "groundtruth"):
            if key in data and data[key] is not None:
                try:
                    data[key] = ListOfEntities.model_validate(data[key])
                except PydanticValidationError as exc:
                    logger.warning(
                        f"Failed to parse '{key}' in {json_file.name}: {exc}"
                    )

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
    response: ListOfEntities | None,
    original_text: str,
    *,
    is_valid_output_format: bool,
) -> bool:
    """
    Check that all predicted entities appear in the original text.

    Parameters
    ----------
    response : ListOfEntities | None
        Parsed Pydantic model containing entities.
    original_text : str
        Text that was annotated.

    Returns
    -------
    bool
        True if all entities appear in the original text.
    """
    if not is_valid_output_format:
        return False
    if response is None or not hasattr(response, "entities"):
        return False

    norm_text = normalize_text(original_text)
    for entity in response.entities:  # noqa: SIM110
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

    Returns
    -------
    pd.DataFrame
        DataFrame with additional quality check columns.
    """
    df = df.copy()
    # Check output format validity
    df["is_valid_output_format"] = df["status"].eq("ok")
    has_no_hallu_list = []
    # Iterate row-wise to apply checks
    for row in df.itertuples(index=False):
        # Check hallucination (only if format is valid)
        has_no_hallu_list.append(  # noqa: PERF401
            has_no_hallucination(
                row.formatted_response,
                row.text,
                is_valid_output_format=row.is_valid_output_format,
            )
        )
    # Add results as new columns
    df["has_no_hallucination"] = has_no_hallu_list
    return df


def group_texts_by_label(entities: list) -> dict[str, list[str]]:
    """Group entity texts by their labels.

    Parameters
    ----------
    entities : list
        List of entities with "category" and "text" fields.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping categories to lists of texts.
    """
    grouped = defaultdict(list)
    for ent in entities:
        category = getattr(ent, "category", None)
        text = getattr(ent, "text", None)
        # Only group if both category and text are present and non-empty
        if category and text:
            grouped[category].append(text)
    return dict(grouped)


def build_category_level_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a category-level DataFrame from the original DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame with one row per annotation file.

    Returns
    -------
    pd.DataFrame
        Category-level DataFrame with one row per entity category
        for each line in the original DataFrame.
    """
    rows = []
    for _, row in df.iterrows():
        # Parse groundtruth
        gt_entities = row["groundtruth"].entities
        # Parse prediction
        pred_entities = row["formatted_response"].entities
        # Group by category
        gt_by_label = group_texts_by_label(gt_entities)
        pred_by_label = group_texts_by_label(pred_entities)
        all_labels = set(gt_by_label) | set(pred_by_label)
        # Create one row per category with all relevant info and metrics
        for category in all_labels:
            # Start with the original row data²
            new_row = row.to_dict()
            # Add category-specific info
            new_row.update(
                {
                    "category": category,
                    "groundtruth_by_label": set(gt_by_label.get(category, [])),
                    "prediction_by_label": set(pred_by_label.get(category, [])),
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
        - "groundtruth_by_label": list of ground-truth entity texts for the category
        - "prediction_by_label": list of predicted entity texts for the category

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
        - "groundtruth_by_label": list of ground-truth entity texts for the category
        - "prediction_by_label": list of predicted entity texts for the category

    Returns
    -------
    pd.DataFrame
        DataFrame with confusion metrics added.
    """
    df = df.copy()
    metrics = df.apply(compute_confusion_metrics_by_row, axis=1)
    return pd.concat([df, metrics], axis=1)


def save_df_to_parquet(
    df,
    path: Path,
) -> None:
    """Serialize all columns before saving to parquet."""
    df_serialized = df.copy()

    # Serialize pydantic models for row "formatted_response" and "groundtruth"
    # by converting them to their JSON representation into mode_dump
    for col in df_serialized.columns:
        df_serialized[col] = df_serialized[col].apply(
            lambda x: x.model_dump() if hasattr(x, "model_dump") else x
        )
        df_serialized[col] = df_serialized[col].apply(
            lambda x: list(x) if isinstance(x, set) else x
        )
    df_serialized.to_parquet(
        path,
        index=False,
    )


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    """Divide two Series safely, returning NaN when denominator is 0.

    Returns
    -------
    pd.Series
        Result of division, with NaN where denominator is 0.
    """
    return a / b.replace(0, np.nan)


def compute_grouped_stats(
    df: pd.DataFrame, df_categories: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute evaluation metrics per label + OVERALL per (model, framework).

    Returns
    -------
    pd.DataFrame
        DataFrame with grouped metrics by model, framework, and label,
        including overall metrics.
    """
    # Group by model, framework, and category to compute metrics per label
    grouped_label = (
        df_categories.groupby(["model_name", "framework_name", "category"])
        .agg(
            nb_of_texts_with_label=("text", "nunique"),
            nb_gt_entities=("groundtruth_by_label", lambda s: sum(len(x) for x in s)),
            nb_predicted_entities=(
                "prediction_by_label",
                lambda s: sum(len(x) for x in s),
            ),
            is_correct_output_format=(
                "is_valid_output_format",
                lambda s: 100 * s.mean(),
            ),
            has_no_hallucinations=("has_no_hallucination", lambda s: 100 * s.mean()),
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
            average_input_tokens=("input_tokens", "mean"),
            average_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
    )
    # Group by model and framework to compute overall metrics across all categories
    # First compute text-level metrics (e.g., number of texts, cost, time)
    # by model+framework
    per_text_stats = (
        df.groupby(["model_name", "framework_name"])
        .agg(
            nb_of_texts_with_label=("text", "nunique"),
            is_correct_output_format=(
                "is_valid_output_format",
                lambda s: 100 * s.mean(),
            ),
            has_no_hallucinations=("has_no_hallucination", lambda s: 100 * s.mean()),
            total_cost_usd=("inference_cost_usd", "sum"),
            total_inference_time_sec=("inference_time_sec", "sum"),
            average_input_tokens=("input_tokens", "mean"),
            average_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
    )
    # Then compute entity-level metrics (TP, FP, FN) by model+framework
    per_entity_stats = (
        df_categories.groupby(["model_name", "framework_name"])
        .agg(
            nb_gt_entities=("groundtruth_by_label", lambda s: s.map(len).sum()),
            nb_predicted_entities=("prediction_by_label", lambda s: s.map(len).sum()),
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
        )
        .reset_index()
    )
    # Merge text-level and entity-level stats to compute overall metrics
    overall = per_text_stats.merge(
        per_entity_stats, on=["model_name", "framework_name"]
    )
    overall["category"] = "OVERALL"
    grouped = pd.concat([grouped_label, overall], ignore_index=True)
    tp = grouped["true_positives"]
    fp = grouped["false_positives"]
    fn = grouped["false_negatives"]

    grouped["precision_score"] = safe_divide(tp, tp + fp)
    grouped["recall_score"] = safe_divide(tp, tp + fn)

    grouped["f1_score"] = safe_divide(
        2 * grouped["precision_score"] * grouped["recall_score"],
        grouped["precision_score"] + grouped["recall_score"],
    )

    beta = 0.5
    grouped["fbeta_0.5_score"] = safe_divide(
        (1 + beta**2) * grouped["precision_score"] * grouped["recall_score"],
        beta**2 * grouped["precision_score"] + grouped["recall_score"],
    )
    return grouped


def save_grouped_stats_to_csv(df: pd.DataFrame, output_path: Path) -> Path:
    """Save grouped metrics to a flat CSV file.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    df.to_csv(output_path, index=False)
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
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    logger = create_logger(
        f"logs/evaluate_entities_extraction_from_{sanitize_filename(str(annotations_dir))}_{timestamp}.log"
    )
    logger.info("Starting LLM annotation evaluation.")
    start_time = time.perf_counter()
    # Loading annotations with metadatas
    df = load_json_annotations_as_dataframe(annotations_dir)
    # Checking that the output format is correct
    # and the absence of hallucination
    df = add_quality_columns(df)
    # Build category-level dataset
    df_category = build_category_level_dataframe(df)
    # Compute confusion metrics (TP, FP, TN) by annotation file and category
    df_with_conf_metrics = compute_confusion_metrics(df_category)
    # Save the detailed evaluation results DataFrame to a Parquet file
    save_df_to_parquet(
        df_with_conf_metrics,
        results_dir / "per_text_and_category_confusion_metrics.parquet",
    )
    # Compute grouped stats by model and framework
    df_grouped_stats = compute_grouped_stats(df, df_with_conf_metrics)
    # Saving into an excel
    save_grouped_stats_to_csv(
        df_grouped_stats,
        results_dir / "grouped_evaluation_metrics.csv",
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
    """Evaluate the quality of JSON entity annotations from CLI."""
    main(
        annotations_dir=annotations_dir,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    run_main_from_cli()
