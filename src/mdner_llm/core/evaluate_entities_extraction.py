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


def count_hallucinated_entities(
    data_row,
    *,
    is_valid_output_format: bool,
) -> tuple[int, int]:
    """Count hallucinated entities using pre-computed flags from normalization step.

    Returns
    -------
    tuple[int, int]
        (number of hallucinated entities, number of predicted entities)
    """
    if not is_valid_output_format or not isinstance(
        data_row.get("normalized_entities"), dict
    ):
        return 0, 0

    entities = data_row["normalized_entities"].get("entities", [])
    nb_predicted = len(entities)
    # Count where is_hallucinated is explicitly True
    nb_hallucinated = sum(1 for ent in entities if ent.get("is_hallucinated", False))

    return nb_hallucinated, nb_predicted


def add_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns of quality checks to the DataFrame.

    Adds the following columns:
    - `is_valid_output_format`:
        True if the LLM response matches the expected JSON format.
    - `nb_hallucinated_entities`:
        Number of predicted entities not found in the original text.
    - `nb_predicted_entities_raw`:
        Total number of predicted entities.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional quality check columns.
    """
    df = df.copy()
    df["is_valid_output_format"] = df["status"].eq("ok")
    # Pass the dictionary row to read normalized_entities
    hallucination_counts = [
        count_hallucinated_entities(
            row,
            is_valid_output_format=row["is_valid_output_format"],
        )
        for _, row in df.iterrows()
    ]
    df["nb_hallucinated_entities"], df["nb_predicted_entities_raw"] = zip(
        *hallucination_counts, strict=True
    )
    return df


def group_texts_by_category(entities: list) -> dict[str, list[str]]:
    """Group entity texts by their categories.

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
            grouped[category].append(normalize_text(text))
    return dict(grouped)


def split_predictions_by_category_and_hallucination(
    normalized_entities: list[dict], category: str
) -> tuple[set[str], set[str]]:
    """Extract hallucinated and grounded entities for a specific category.

    Returns
    -------
    tuple[set[str], set[str]]
        (set of hallucinated entity texts, set of grounded entity texts)
    """
    hallucinated = set()
    grounded = set()

    for ent in normalized_entities:
        if ent.get("category") == category:
            text = normalize_text(ent.get("text", ""))
            if text:
                if ent.get("is_hallucinated", False):
                    hallucinated.add(text)
                else:
                    grounded.add(text)

    return hallucinated, grounded


def build_category_level_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a category-level DataFrame from the original DataFrame.

    Returns
    -------
    pd.DataFrame
        Category-level DataFrame with one row per entity category
        for each line in the original DataFrame.
    """
    rows = []
    rows = []
    for _, row in df.iterrows():
        gt_entities = row["groundtruth"].entities
        gt_by_category = group_texts_by_category(gt_entities)

        # Get the list of normalized entities dicts
        norm_data = row.get("normalized_entities", {})
        pred_entities = (
            norm_data.get("entities", []) if isinstance(norm_data, dict) else []
        )

        # Collect all unique categories present in GT or Preds
        pred_categories = {
            ent.get("category") for ent in pred_entities if ent.get("category")
        }
        all_categories = set(gt_by_category) | pred_categories

        for category in all_categories:
            new_row = row.to_dict()

            # Use our new helper to get pre-calculated split sets
            hallucinated, grounded = split_predictions_by_category_and_hallucination(
                pred_entities, category
            )

            # Reconstruction of all predicted texts for this category
            pred_texts = hallucinated | grounded

            new_row.update(
                {
                    "category": category,
                    "groundtruth_by_category": set(gt_by_category.get(category, [])),
                    "prediction_by_category": pred_texts,
                    "hallucinated_by_category": hallucinated,
                    "grounded_prediction_by_category": grounded,
                }
            )
            rows.append(new_row)

    return pd.DataFrame(rows)


def compute_confusion_metrics_by_row(row):
    """Compute confusion metrics (TP, FP, FN) for a single row at entity level.

    Returns
    -------
    pd.Series
        Series with TP, FP, FN, hallucination-free FP, entity lists, and counts.
    """
    gt = set(row.get("groundtruth_by_category", []))
    pred = set(row.get("prediction_by_category", []))
    hallucinated = set(row.get("hallucinated_by_category", []))

    tp = gt & pred
    fp = pred - gt
    fn = gt - pred
    fp_no_hallucination = fp - hallucinated

    return pd.Series(
        {
            "true_positives": len(tp),
            "false_positives": len(fp),
            "false_positives_no_hallucination": len(fp_no_hallucination),
            "false_negatives": len(fn),
            "tp_entities": list(tp),
            "fp_entities": list(fp),
            "fn_entities": list(fn),
        }
    )


def save_df_to_parquet(
    df,
    path: Path,
) -> None:
    """Serialize all columns before saving to parquet."""
    df_serialized = df.copy()
    for col in df_serialized.columns:
        df_serialized[col] = df_serialized[col].apply(
            lambda x: x.model_dump() if hasattr(x, "model_dump") else x
        )
        df_serialized[col] = df_serialized[col].apply(
            lambda x: sorted(x) if isinstance(x, set) else x
        )
    df_serialized.to_parquet(path, index=False)


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
    Compute evaluation metrics per category + OVERALL per (model, framework).

    Returns
    -------
    pd.DataFrame
        DataFrame with grouped metrics by model, framework, and category,
        including overall metrics.
    """
    # Group by model, framework, and category to compute metrics per category
    grouped_category = (
        df_categories.groupby(["model_name", "framework_name", "category"])
        .agg(
            nb_texts_with_category=(
                "groundtruth_by_category",
                lambda s: (s.apply(len) > 0).sum(),
            ),
            nb_groundtruth_entities=(
                "groundtruth_by_category",
                lambda s: sum(len(x) for x in s),
            ),
            nb_predicted_entities=(
                "prediction_by_category",
                lambda s: sum(len(x) for x in s),
            ),
            nb_hallucinated_entities=(
                "hallucinated_by_category",
                lambda s: sum(len(x) for x in s),
            ),
            pct_correct_format=(
                "is_valid_output_format",
                lambda s: 100 * s.mean(),
            ),
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_positives_no_hallucination=(
                "false_positives_no_hallucination",
                "sum",
            ),
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
            nb_texts_with_category=("text", "nunique"),
            pct_correct_format=(
                "is_valid_output_format",
                lambda s: 100 * s.mean(),
            ),
            nb_hallucinated_entities=("nb_hallucinated_entities", "sum"),
            nb_predicted_entities_raw=("nb_predicted_entities_raw", "sum"),
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
            nb_groundtruth_entities=(
                "groundtruth_by_category",
                lambda s: s.map(len).sum(),
            ),
            nb_predicted_entities=(
                "prediction_by_category",
                lambda s: s.map(len).sum(),
            ),
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_positives_no_hallucination=(
                "false_positives_no_hallucination",
                "sum",
            ),
            false_negatives=("false_negatives", "sum"),
        )
        .reset_index()
    )
    # Merge text-level and entity-level stats to compute overall metrics
    overall = per_text_stats.merge(
        per_entity_stats, on=["model_name", "framework_name"]
    )
    overall["category"] = "OVERALL"
    grouped = pd.concat([grouped_category, overall], ignore_index=True)
    grouped["pct_hallucinations"] = 100 * safe_divide(
        grouped["nb_hallucinated_entities"], grouped["nb_predicted_entities"]
    )

    tp = grouped["true_positives"]
    fp = grouped["false_positives"]
    fp_clean = grouped["false_positives_no_hallucination"]
    fn = grouped["false_negatives"]

    grouped["precision_score"] = safe_divide(tp, tp + fp)
    grouped["precision_score_no_hallucination"] = safe_divide(tp, tp + fp_clean)
    grouped["recall_score"] = safe_divide(tp, tp + fn)
    grouped["f1_score"] = safe_divide(
        2 * grouped["precision_score"] * grouped["recall_score"],
        grouped["precision_score"] + grouped["recall_score"],
    )
    grouped["f1_score_no_hallucination"] = safe_divide(
        2 * grouped["precision_score_no_hallucination"] * grouped["recall_score"],
        grouped["precision_score_no_hallucination"] + grouped["recall_score"],
    )
    beta = 0.5
    grouped[f"fbeta_{beta}_score"] = safe_divide(
        (1 + beta**2) * grouped["precision_score"] * grouped["recall_score"],
        beta**2 * grouped["precision_score"] + grouped["recall_score"],
    )
    grouped[f"fbeta_{beta}_score_no_hallucination"] = safe_divide(
        (1 + beta**2)
        * grouped["precision_score_no_hallucination"]
        * grouped["recall_score"],
        beta**2 * grouped["precision_score_no_hallucination"] + grouped["recall_score"],
    )
    return grouped


def main(inferences_dir: Path, results_dir: Path) -> None:
    """
    Evaluate the quality of JSON entity annotations.

    Parameters
    ----------
    inferences_dir : Path
        Directory containing the JSON inference files to evaluate.
    results_dir : Path
        Directory where evaluation results, logs, and reports will be written.
    """
    # Configure logging
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    logger = create_logger(
        f"logs/evaluate_entities_extraction_from_{sanitize_filename(str(inferences_dir))}_{timestamp}.log"
    )
    logger.info("Starting LLM annotation evaluation.")
    start_time = time.perf_counter()
    # Loading annotations with metadatas
    df = load_json_annotations_as_dataframe(inferences_dir)
    # Checking that the output format is correct
    # and the absence of hallucination
    df = add_quality_columns(df)
    # Build category-level dataset
    df_category = build_category_level_dataframe(df)
    # Compute confusion metrics (TP, FP, TN) by annotation file and category
    metrics = df_category.apply(compute_confusion_metrics_by_row, axis=1)
    df_with_conf_metrics = pd.concat([df_category, metrics], axis=1)
    # Save the detailed evaluation results DataFrame to a Parquet file
    save_df_to_parquet(
        df_with_conf_metrics,
        results_dir / "per_text_and_category_confusion_metrics.parquet",
    )
    # Compute grouped stats by model and framework
    df_grouped_stats = compute_grouped_stats(df, df_with_conf_metrics)
    # Saving into an excel
    output_path = results_dir / "grouped_evaluation_metrics.csv"
    df_grouped_stats.to_csv(output_path, index=False)
    elapsed_time = int(time.perf_counter() - start_time)
    logger.success(f"Evaluation duration: {timedelta(seconds=elapsed_time)} 🎉")


@click.command()
@click.option(
    "--inferences-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing the JSON annotation files to evaluate.",
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Target directory where evaluation results will be saved.",
    callback=ensure_dir,
)
def run_main_from_cli(
    inferences_dir: Path,
    results_dir: Path,
) -> None:
    """Evaluate the quality of JSON entity annotations from CLI."""
    main(
        inferences_dir=inferences_dir,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    run_main_from_cli()
