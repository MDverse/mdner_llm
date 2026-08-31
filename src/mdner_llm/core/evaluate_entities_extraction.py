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


def extract_predicted_entities_from_row(data_row) -> list[dict]:
    """Fallback from normalized_entities to formatted_response.

    Returns
    -------
    list[dict]
        List of predicted entities with their categories and hallucination flags.
    """
    if isinstance(data_row.get("normalized_entities"), dict):
        return data_row["normalized_entities"].get("entities", [])

    formatted = data_row.get("formatted_response")
    if formatted and hasattr(formatted, "entities"):
        return [
            {"category": ent.category, "text": ent.text, "is_hallucinated": False}
            for ent in formatted.entities
        ]
    elif isinstance(formatted, dict) and "entities" in formatted:
        return [
            {
                "category": ent.get("category"),
                "text": ent.get("text"),
                "is_hallucinated": False,
            }
            for ent in formatted["entities"]
        ]
    return []


def count_predicted_entities(data_row) -> int:
    """Count the number of predicted entities in a data row.

    Returns
    -------
    int
        Number of predicted entities in the data_row.
    """
    entities = extract_predicted_entities_from_row(data_row)
    return len(entities)


def count_hallucinated_entities(
    data_row,
    *,
    is_valid_output_format: bool,
) -> int:
    """Count hallucinated entities using pre-computed flags from normalization step.

    Returns
    -------
    int
        Number of hallucinated entities in the normalized_entities field.
    """
    if not is_valid_output_format or not isinstance(
        data_row.get("normalized_entities"), dict
    ):
        return 0
    entities = data_row["normalized_entities"].get("entities", [])
    return sum(
        1
        for ent in entities
        if isinstance(ent, dict) and ent.get("is_hallucinated", False)
    )


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
    # Count the number of predicted entities
    df["nb_predicted_entities_raw"] = df.apply(
        lambda row: count_predicted_entities(row), axis=1
    )
    # Count the number of hallucinated entities
    df["nb_hallucinated_entities"] = df.apply(
        lambda row: count_hallucinated_entities(
            row, is_valid_output_format=row["is_valid_output_format"]
        ),
        axis=1,
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
    for _, row in df.iterrows():
        gt_entities = row["groundtruth"].entities
        gt_by_category = group_texts_by_category(gt_entities)
        # Get the list of predicted entities
        pred_entities = extract_predicted_entities_from_row(row)
        # Collect all unique categories present in GT or Preds
        pred_categories = {
            ent.get("category") for ent in pred_entities if ent.get("category")
        }
        all_categories = set(gt_by_category) | pred_categories
        for category in all_categories:
            new_row = row.to_dict()
            # Split predictions into hallucinated and grounded for this category
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


def _compute_scores(
    tp: pd.Series, fp: pd.Series, fn: pd.Series, fp_clean: pd.Series
) -> dict[str, pd.Series]:
    """Compute precision/recall/F1/F-beta scores from confusion counts.

    Returns
    -------
    dict[str, pd.Series]
        Mapping of metric names to their computed Series.
    """
    beta = 0.5
    precision = safe_divide(tp, tp + fp)
    precision_clean = safe_divide(tp, tp + fp_clean)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    f1_clean = safe_divide(2 * precision_clean * recall, precision_clean + recall)
    fbeta = safe_divide(
        (1 + beta**2) * precision * recall, beta**2 * precision + recall
    )
    fbeta_clean = safe_divide(
        (1 + beta**2) * precision_clean * recall, beta**2 * precision_clean + recall
    )
    return {
        "precision": precision,
        "precision_no_hallucination": precision_clean,
        "recall": recall,
        "f1": f1,
        "f1_no_hallucination": f1_clean,
        f"fbeta_{beta}": fbeta,
        f"fbeta_{beta}_no_hallucination": fbeta_clean,
    }


def compute_grouped_stats(
    df: pd.DataFrame, df_categories: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute per-category, MICRO, and MACRO evaluation metrics.

    Rows are stacked by (model, framework, category), where `category` is either an
    actual entity category, "OVERALL_MICRO" (TP/FP/FN pooled across all categories
    before scoring, so frequent categories dominate), or "OVERALL_MACRO" (per-category
    scores averaged unweighted, so rare and frequent categories count equally).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per (model, framework, category).
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
    grouped_category["pct_hallucinations"] = 100 * safe_divide(
        grouped_category["nb_hallucinated_entities"],
        grouped_category["nb_predicted_entities"],
    )
    tp, fp, fp_clean, fn = (
        grouped_category["true_positives"],
        grouped_category["false_positives"],
        grouped_category["false_positives_no_hallucination"],
        grouped_category["false_negatives"],
    )
    grouped_category = grouped_category.assign(**_compute_scores(tp, fp, fn, fp_clean))
    # OVERALL MICRO row: pool text/entity counts and TP/FP/FN across all categories
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
            inference_date=("timestamp", "max"),
        )
        .reset_index()
    )
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
    micro = per_text_stats.merge(per_entity_stats, on=["model_name", "framework_name"])
    micro["pct_hallucinations"] = 100 * safe_divide(
        micro["nb_hallucinated_entities"], micro["nb_predicted_entities"]
    )
    tp, fp, fp_clean, fn = (
        micro["true_positives"],
        micro["false_positives"],
        micro["false_positives_no_hallucination"],
        micro["false_negatives"],
    )
    micro = micro.assign(
        **_compute_scores(tp, fp, fn, fp_clean), category="OVERALL_MICRO"
    )
    # MACRO row: unweighted mean of the per-category scores
    score_cols = [
        "precision",
        "precision_no_hallucination",
        "recall",
        "f1",
        "f1_no_hallucination",
        "fbeta_0.5",
        "fbeta_0.5_no_hallucination",
    ]
    macro_scores = (
        grouped_category.groupby(["model_name", "framework_name"])[score_cols]
        .mean()
        .reset_index()
        .assign(category="OVERALL_MACRO")
    )
    # Attach text-level metadata columns to macro summary.
    macro = per_text_stats.merge(
        macro_scores, on=["model_name", "framework_name"]
    ).assign(category="OVERALL_MACRO")
    # Concatenate the per-category, MICRO and MACRO DataFrames.
    return pd.concat([grouped_category, micro, macro], ignore_index=True)


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
