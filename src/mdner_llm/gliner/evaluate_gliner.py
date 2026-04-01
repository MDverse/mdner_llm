"""Evaluate the GLINER2 model on a test set."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import click
import loguru
import pandas as pd
from gliner2 import GLiNER2

from mdner_llm.core.logger import create_logger
from mdner_llm.utils.common import ensure_dir, sanitize_filename


def load_model(
    model_path: str | Path, logger: "loguru.Logger" = loguru.logger
) -> GLiNER2:
    """Load the GLINER2 model from the specified path.

    Parameters
    ----------
    model_path : str | Path
        Path to the trained model file.
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger

    Returns
    -------
    GLiNER2
        The loaded GLINER2 model.
    """
    try:
        # Load the model from the specified path
        model = GLiNER2.from_pretrained(model_path)
        logger.success(f"Loaded model from {model_path} successfully.")
    except Exception as exc:
        logger.error(f"Failed to load model from {model_path}: {exc}")
        raise
    else:
        return model


def load_test_dataset_from_jsonl(
    file_path: str | Path, logger: "loguru.Logger" = loguru.logger
) -> list[dict[str, Any]]:
    """Load a JSONL file and return a list of dictionaries.

    Parameters
    ----------
    file_path:
        Path to the JSONL file.
    logger:
        Logger for logging messages, by default loguru.logger

    Returns
    -------
        A list where each element is a dictionary corresponding to one line.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a line is not valid JSON.
    """
    logger.info(f"Loading test dataset from {file_path}.")
    # Ensure the file path is a Path object
    path = Path(file_path)
    # Check if the file exists
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    data = []
    # Read the file line by line and parse each line as JSON
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON at line {i}"
                raise ValueError(msg) from exc
    if data == []:
        logger.warning("No valid data found.")
    else:
        logger.debug("First example:")
        logger.debug(f"Text: {data[0].get('input', '')}")
        logger.debug(f"Ground Truth: {data[0].get('output', {}).get('entities', {})}")

    logger.success(f"Loaded {len(data)} test samples successfully!")
    return data


def load_test_dataset_entries(
    file_path: str | Path, logger: "loguru.Logger" = loguru.logger
) -> list[tuple[Path, str]]:
    r"""Load a text file containing (path, url) pairs.

    Each line is expected to be:
        <path>\t<url>

    Parameters
    ----------
    file_path : str | Path
        Path to the text file.
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger

    Returns
    -------
    list[tuple[Path, str]]
        List of (path, url) tuples.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a line is malformed.
    """
    logger.info(f"Loading test dataset metadata from {file_path}.")
    # Ensure the file path is a Path object
    path = Path(file_path)
    # Check if the file exists
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    entries = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                msg = f"Invalid format at line {i}: {line}"
                raise ValueError(msg)

            file_path_str, url = parts[0], parts[1]
            entries.append((Path(file_path_str), url))

    if entries == []:
        logger.warning("No valid metadata found.")
    else:
        logger.debug("First example metadata:")
        logger.debug(f"Path: {entries[0][0]}")
        logger.debug(f"Url: {entries[0][1]}")

    logger.success(f"Loaded metadata for {len(entries)} test samples successfully!")
    return entries


def split_gliner_output_with_scores(
    extracted_entities: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    """Split GLINER output into simplified texts and detailed (text + score).

    Parameters
    ----------
    extracted_entities : dict[str, Any]
        Raw output from model.extract_entities.

    Returns
    -------
    tuple
        - dict[label, list[text]]
        - dict[label, list[{text, score}]]
    """
    # Extract the "entities" dictionary from the model output
    entities = extracted_entities.get("entities", {})

    simplified = {}
    with_scores = {}
    # Iterate over each entity label and its corresponding items
    for label, items in entities.items():
        texts = []
        detailed = []
        for item in items:
            if "text" not in item:
                continue
            # Extract the text and confidence score
            text = item["text"]
            score = float(item.get("confidence"))
            # Add only the text to the simplified output
            texts.append(text)
            # Add the text and score to the detailed output
            detailed.append({"text": text, "score": round(score, 3)})

        simplified[label] = texts
        with_scores[label] = detailed

    return simplified, with_scores


def is_valid_output_format(pred: dict[str, Any]) -> bool:
    """Check whether model output follows expected schema.

    Expected format:
    {"entities": {"LABEL": [{"text": "..."}]}}

    Parameters
    ----------
    pred : dict[str, Any]
        The model output to validate.

    Returns
    -------
    bool
        True if the output format is valid, False otherwise.
    """
    # Check that the output is a dictionary
    if not isinstance(pred, dict):
        return False
    # Check that the "entities" key exists and is a dictionary
    entities = pred.get("entities")
    if not isinstance(entities, dict):
        return False
    # Check that each label is a string
    # and each item is a list of dictionaries with "text" keys
    for label, items in entities.items():
        if not isinstance(label, str) or not isinstance(items, list):
            return False
        for item in items:
            if not isinstance(item, dict):
                return False
            if "text" not in item:
                return False

    return True


def has_no_hallucination(
    prediction: dict[str, list[str]],
    text: str,
) -> bool:
    """Check that all predicted entities are substrings of the input text.

    Parameters
    ----------
    prediction : dict[str, list[str]]
        The predicted entities,
        where keys are labels and values are lists of entity texts.
    text : str
        The original input text that was annotated.

    Returns
    -------
    bool
        True if all predicted entities are found in the input text,
        False if any entity is not a substring of the input text
        (indicating potential hallucination).
    """
    # Iterate over all predicted entities
    for entities in prediction.values():
        for entity in entities:
            # Check if the entity is a substring of the input text
            if entity not in text:
                return False
    return True


def compute_metrics_per_class(
    prediction: dict[str, list[str]],
    groundtruth: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Compute TP, FP, FN per entity class.

    Parameters
    ----------
    prediction : dict[str, list[str]]
        The predicted entities, where keys are labels and values are lists
        of entity texts.
    groundtruth : dict[str, Any]
        The ground truth entities, where each item is a dictionary with "label" and
        "text" keys.

    Returns
    -------
    dict[str, dict[str, int]]
        A dictionary where keys are entity labels and values are dictionaries with
        counts of true positives, false positives, and false negatives.
    """
    # Convert groundtruth and prediction to dict[label -> set[text]]
    gt_dict = {key: set(value) for key, value in groundtruth.items()}
    pred_dict = {key: set(value) for key, value in prediction.items()}
    # Get the set of all labels present in either ground truth or prediction
    all_labels = set(gt_dict) | set(pred_dict)

    metrics = {}
    # Compute TP, FP, FN for each label
    for label in all_labels:
        gt = gt_dict.get(label, set())
        pred = pred_dict.get(label, set())

        tp_entities = gt & pred
        fp_entities = pred - gt
        fn_entities = gt - pred
        # Record the counts
        # and the specific entities
        # for TP, FP, FN in the metrics dictionary
        metrics[label] = {
            "true_positives": len(tp_entities),
            "false_positives": len(fp_entities),
            "false_negatives": len(fn_entities),
            "tp_entities": list(tp_entities),
            "fp_entities": list(fp_entities),
            "fn_entities": list(fn_entities),
        }

    return metrics


def process_sample(
    model: GLiNER2,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Run model + compute evaluation signals for one sample.

    Parameters
    ----------
    model : GLiNER2
        The GLiNER2 model to use for entity extraction.
    sample : dict[str, Any]
        A dictionary containing the input text and
        the expected output with ground truth.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the input text, ground truth, model prediction,
        and evaluation metrics.
    """
    # Extract the input text
    text = sample.get("input", "")
    # Extract the expected output, including ground truth entities
    # and the description of each entity class
    output = sample.get("output", {})
    groundtruth = output.get("entities", {})
    entity_desc = output.get("entity_descriptions", {})
    # Run the model to extract entities from the input text,
    # using the entity descriptions
    # including confidence scores in the raw output for later analysis
    raw_pred = model.extract_entities(
        text,
        entity_desc,
        include_confidence=True,
    )
    # Split the raw model output into a
    # simplified version with just the predicted entity texts per class
    # and a detailed version that includes confidence scores for each predicted entity
    prediction, prediction_with_scores = split_gliner_output_with_scores(raw_pred)
    # Check if the raw model output follows the expected format
    format_valid = is_valid_output_format(raw_pred)
    # Check for hallucinations
    no_hallucination = has_no_hallucination(prediction, text)
    # Compute the confusion metrics (TP, FP, FN) for each entity class
    metrics_per_class = compute_metrics_per_class(prediction, groundtruth)

    return {
        "text": text,
        "groundtruth": groundtruth,
        "prediction": prediction,
        "prediction_with_scores": prediction_with_scores,
        "is_format_valid": format_valid,
        "has_no_hallucination": no_hallucination,
        "metrics_per_class": metrics_per_class,
    }


def build_evaluation_dataframe(
    model_name: str,
    model: GLiNER2,
    dataset: list[dict[str, Any]],
    entries_metadata: list[tuple[Path, str]],
    logger: "loguru.Logger" = loguru.logger,
) -> pd.DataFrame:
    """Run evaluation on full dataset and return a flat DataFrame.

    Parameters
    ----------
    model_name : str
        The name of the model being evaluated,
        used for contextual information in the DataFrame.
    model : GLiNER2
        The GLiNER2 model to use for entity extraction.
    dataset : list[dict[str, Any]]
        The test dataset, where each item is a dictionary containing the input text
        and expected output.
    entries_metadata : list[tuple[Path, str]]
        A list of tuples containing the file path and URL for each sample
        in the dataset, used for contextual information in the DataFrame.
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to one entity class prediction for one
        sample, including contextual information and evaluation metrics.
    """
    logger.info("Building evaluation DataFrame from model predictions...")
    rows = []
    for index, (sample, (path, url)) in enumerate(
        zip(dataset, entries_metadata, strict=True)
    ):
        logger.info(f"Processing sample {index + 1}/{len(dataset)}: {path} ({url})")
        result = process_sample(model, sample)
        prediction = result["prediction"]
        groundtruth = result["groundtruth"]
        logger.debug(f"Prediction ({len(prediction)} entities): {prediction}")
        logger.debug(f"Ground Truth ({len(groundtruth)} entities): {groundtruth}")

        for label, metrics in result["metrics_per_class"].items():
            logger.debug(f"Metrics for {label}:")
            logger.debug(f"TP: {metrics['true_positives']}")
            logger.debug(f"FP: {metrics['false_positives']}")
            logger.debug(f"FN: {metrics['false_negatives']}")
            predictions_with_scores = result["prediction_with_scores"].get(label, [])
            avg_confidence = sum(
                item["score"] for item in predictions_with_scores
            ) / max(len(predictions_with_scores), 1)
            logger.debug(
                f"Average confidence for predicted entities: {avg_confidence:.3f}"
            )
            rows.append(
                {
                    # Add contextual information for grouping
                    "model_name": model_name,
                    "text": result["text"],
                    "json_path": str(path),
                    "url": url,
                    "label": label,
                    "groundtruth": groundtruth.get(label, []),
                    "prediction": prediction.get(label, []),
                    "prediction_with_scores": (
                        result["prediction_with_scores"].get(label, [])
                    ),
                    "avg_confidence_score": avg_confidence,
                    # Add check for valid output format and hallucination
                    # It should be 100% for encoders
                    "is_format_valid": result["is_format_valid"],
                    "has_no_hallucination": result["has_no_hallucination"],
                    # Add the confusion metrics for this label
                    "true_positives": metrics["true_positives"],
                    "false_positives": metrics["false_positives"],
                    "false_negatives": metrics["false_negatives"],
                    # Add the specific entities for TP, FP, FN for this label
                    "tp_entities": metrics["tp_entities"],
                    "fp_entities": metrics["fp_entities"],
                    "fn_entities": metrics["fn_entities"],
                }
            )
    logger.success(f"Built evaluation DataFrame of {len(rows)} rows successfully!")
    return pd.DataFrame(rows)


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


def compute_metrics(
    df: pd.DataFrame,
    beta: float = 0.5,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute precision, recall, F1-score, and F-beta score for each row.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing columns for true positives, false positives,
        and false negatives.
    beta : float, optional
        The beta value to use for the F-beta score calculation, by default 0.5
        (which weights precision more than recall).

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        A tuple containing four pandas Series for precision, recall, F1-score,
        and F-beta score, respectively.
    """
    # Get the summed TP, FP, FN for each group
    tp = df["true_positives"]
    fp = df["false_positives"]
    fn = df["false_negatives"]
    # Compute precision, recall, F1-score, and F-beta score for each group
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    fbeta_score = safe_divide(
        (1 + beta**2) * precision * recall,
        beta**2 * precision + recall,
    )
    return precision, recall, f1_score, fbeta_score


def compute_stats_per_label(
    df: pd.DataFrame, beta: float = 0.5, logger: "loguru.Logger" = loguru.logger
) -> pd.DataFrame:
    """Compute metrics per model and per label.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing evaluation results.
    beta : float, optional
        The beta value to use for the F-beta score calculation, by default 0.5
        (which weights precision more than recall).
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger

    Returns
    -------
    pd.DataFrame
        A DataFrame with computed metrics (precision, recall, F1-score, etc.)
        for each model and each label.
    """
    logger.info("Computing metrics per model and per label...")
    # Compute metrics per model and per label by grouping the DataFrame
    grouped = (
        # Group by model name and label
        df.groupby(["model_name", "label"])
        # and aggregate the relevant columns to compute metrics
        .agg(
            # Count the number of unique texts (samples) for this model and label
            nb_annotations=("text", "nunique"),
            # Compute the percentage of samples where the output format is valid
            pct_is_format_valid=("is_format_valid", lambda s: 100 * s.mean()),
            # Compute the percentage of samples where there is no hallucination
            pct_has_no_hallucination=("has_no_hallucination", lambda s: 100 * s.mean()),
            # Sum the true positives, false positives,
            # and false negatives for this model and label
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
        )
        .reset_index()
    )
    precision, recall, f1_score, fbeta_score = compute_metrics(grouped, beta=beta)
    for label in grouped["label"].unique():
        logger.debug(f"Evaluation metrics for label '{label}':")
        label_mask = grouped["label"] == label
        logger.debug(f"Precision: {precision[label_mask].to_numpy()}")
        logger.debug(f"Recall: {recall[label_mask].to_numpy()}")
        logger.debug(f"F1-score: {f1_score[label_mask].to_numpy()}")
        logger.debug(
            f"F-beta score (beta={beta}): {fbeta_score[label_mask].to_numpy()}"
        )
    # Add the computed metrics as new columns in the grouped DataFrame
    grouped["precision_score"] = precision
    grouped["recall_score"] = recall
    grouped["f1_score"] = f1_score
    grouped[f"fbeta_{beta}_score"] = fbeta_score
    logger.success(
        f"Computed metrics per model and per label for {len(grouped)} "
        "samples successfully!"
    )
    return grouped


def compute_stats_overall(
    df: pd.DataFrame, beta: float = 0.5, logger: "loguru.Logger" = loguru.logger
) -> pd.DataFrame:
    """Compute overall metrics per model (all labels merged).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing evaluation results.
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger

    Returns
    -------
    pd.DataFrame
        A DataFrame with computed metrics (precision, recall, F1-score, etc.)
        for each model (all labels merged).
    """
    logger.info("Computing overall metrics per model (all labels merged)...")
    grouped = (
        # Group by model name only (merging all labels together)
        df.groupby("model_name")
        .agg(
            nb_annotations=("text", "nunique"),
            pct_is_format_valid=("is_format_valid", lambda s: 100 * s.mean()),
            pct_has_no_hallucination=("has_no_hallucination", lambda s: 100 * s.mean()),
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
        )
        .reset_index()
    )
    precision, recall, f1_score, fbeta_score = compute_metrics(grouped, beta=beta)
    for model_name in grouped["model_name"].unique():
        logger.debug(f"Overall evaluation metrics for model '{model_name}':")
        model_mask = grouped["model_name"] == model_name
        logger.debug(f"Precision: {precision[model_mask].to_numpy()}")
        logger.debug(f"Recall: {recall[model_mask].to_numpy()}")
        logger.debug(f"F1-score: {f1_score[model_mask].to_numpy()}")
        logger.debug(f"F-beta score (beta=0.5): {fbeta_score[model_mask].to_numpy()}")
    # Add the computed metrics as new columns in the grouped DataFrame
    grouped["precision_score"] = precision
    grouped["recall_score"] = recall
    grouped["f1_score"] = f1_score
    grouped["fbeta_0.5_score"] = fbeta_score
    # Rename the label to "OVERALL" since this is the overall metrics for each model
    grouped["label"] = "OVERALL"
    logger.success(
        f"Computed overall metrics per model for {len(grouped)} samples successfully!"
    )
    return grouped


def compute_all_stats(
    df: pd.DataFrame, beta: float = 0.5, logger: "loguru.Logger" = loguru.logger
) -> pd.DataFrame:
    """Compute both per-label and overall metrics.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing evaluation results.
    beta : float, optional
        The beta value to use for the F-beta score calculation, by default 0.5
        (which weights precision more than recall).
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger

    Returns
    -------
    pd.DataFrame
        A DataFrame with computed metrics for each model and each label,
        as well as overall metrics for each model (with label "OVERALL").
    """
    per_label = compute_stats_per_label(df, beta=beta, logger=logger)
    overall = compute_stats_overall(df, beta=beta, logger=logger)

    return pd.concat([per_label, overall], ignore_index=True)


def save_per_text_metrics_to_parquet(
    df: pd.DataFrame,
    model_name: str,
    output_dir: str | Path,
    timestamp: str,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Save the detailed evaluation DataFrame to a Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing detailed evaluation results for each sample and label.
    model_name : str
        The name of the model being evaluated,
        used for contextual information in the filename.
    output_dir : str | Path
        The directory where the Parquet file will be saved.
    timestamp : str
        A timestamp string to include in the filename for versioning.
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger
    """
    output_path = (
        Path(output_dir)
        / f"per_text_metrics_{sanitize_filename(model_name)}_{timestamp}.parquet"
    )
    try:
        df.to_parquet(output_path, index=False)
        logger.success(
            f"Saved per-text evaluation metrics to {output_path} successfully!"
        )
    except Exception as exc:
        logger.error(f"Failed to save per-text metrics to {output_path}: {exc}")
        raise


def save_overall_stats_to_excel(
    df: pd.DataFrame,
    model_name: str,
    output_dir: str | Path,
    timestamp: str,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Save the overall evaluation metrics DataFrame to an Excel file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing overall evaluation metrics for each model and label.
    model_name : str
        The name of the model being evaluated,
        used for contextual information in the filename.
    output_dir : str | Path
        The directory where the Excel file will be saved.
    timestamp : str
        A timestamp string to include in the filename for versioning.
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger
    """
    output_path = (
        Path(output_dir)
        / f"overall_metrics_{sanitize_filename(model_name)}_{timestamp}.xlsx"
    )
    try:
        df.to_excel(output_path, index=False)
        logger.success(
            f"Saved overall evaluation metrics to {output_path} successfully!"
        )
    except Exception as exc:
        logger.error(f"Failed to save overall metrics to {output_path}: {exc}")
        raise


def main(
    model_name: str,
    model_path: str | Path,
    test_dataset_path: str | Path,
    test_metadata_path: str | Path,
    beta: float,
    output_dir: str | Path,
):
    """Evaluate GLINER2 model using the specified model path."""
    # Initialize logger
    timestamp = datetime.now(UTC).strftime("%Y%m%d")
    logger = create_logger(
        f"logs/eval_gliner_{sanitize_filename(model_name)}_{timestamp}.log"
    )
    logger.info("Starting GLINER2 evaluation...")
    start_time = datetime.now(UTC)
    # Load finetuned model
    model = load_model(model_path, logger)
    # Get the test dataset from the JSONL file
    # It contains the input texts and the expected outputs (ground truth)
    test_dataset = load_test_dataset_from_jsonl(test_dataset_path, logger)
    # Load the test dataset entries metadata from the text file
    # It contains the file paths and URLs corresponding to each sample
    test_entries_metadata = load_test_dataset_entries(test_metadata_path, logger)
    # Run evaluation and build the results DataFrame
    # Each row in the df corresponds to an entity class prediction for one sample,
    # and includes metadata and evaluation metrics for that sample and label
    df = build_evaluation_dataframe(
        model_name=model_name,
        model=model,
        dataset=test_dataset,
        entries_metadata=test_entries_metadata,
    )
    # Save the detailed evaluation results DataFrame to a Parquet file
    save_per_text_metrics_to_parquet(df, model_name, output_dir, timestamp, logger)
    # Compute the aggregated metrics per label and overall, and save to Excel
    stats = compute_all_stats(df, beta=beta, logger=logger)
    save_overall_stats_to_excel(stats, model_name, output_dir, timestamp, logger)
    elapsed_time = int((datetime.now(UTC) - start_time).total_seconds())
    logger.success(
        f"Evaluation completed successfully in: {timedelta(seconds=elapsed_time)}!"
    )


@click.command()
@click.option(
    "--model-name",
    type=str,
    default="GLiNER2_Base_205M_parameters",
    help="Name of the model.",
)
@click.option(
    "--model-path",
    type=click.Path(),
    default="fastino/gliner2-base-v1",
    help="Path to the trained model file.",
)
@click.option(
    "--test-dataset",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default="data/gliner/test.jsonl",
    help="Path to the test dataset file.",
)
@click.option(
    "--test-metadata-path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default="data/gliner/test_metadata.txt",
    help="Path to the test metadata file.",
)
@click.option(
    "--beta",
    type=float,
    default=0.5,
    help=(
        "Beta value for F-beta score calculation "
        "(default: 0.5, which weights precision more than recall)."
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default="results/gliner/evaluation_stats",
    help="Directory to save evaluation results.",
    callback=ensure_dir,
)
def run_main_from_cli(
    model_name: str,
    model_path: str | Path,
    test_dataset: str | Path,
    test_metadata_path: str | Path,
    beta: float,
    output_dir: str | Path,
):
    """Run evaluation of GLINER2 model from the command line."""
    main(
        model_name=model_name,
        model_path=model_path,
        test_dataset_path=test_dataset,
        test_metadata_path=test_metadata_path,
        output_dir=output_dir,
        beta=beta,
    )


if __name__ == "__main__":
    run_main_from_cli()
