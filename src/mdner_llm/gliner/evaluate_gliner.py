"""Evaluate the GLINER2 model on the test set using the best model from training."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import loguru
import pandas as pd
from gliner2 import GLiNER2

from mdner_llm.core.logger import create_logger
from mdner_llm.utils.common import sanitize_filename


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


def load_test_dataset_from_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file and return a list of dictionaries.

    Parameters
    ----------
        file_path: Path to the JSONL file.

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

    return data


def load_test_dataset_entries(file_path: str | Path) -> list[tuple[Path, str]]:
    r"""Load a text file containing (path, url) pairs.

    Each line is expected to be:
        <path>\t<url>

    Parameters
    ----------
    file_path : str | Path
        Path to the text file.

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
    path = Path(file_path)

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
        The predicted entities, where keys are labels and values are lists of entity texts.
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
    groundtruth: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Compute TP, FP, FN per entity class.

    Parameters
    ----------
    prediction : dict[str, list[str]]
        The predicted entities, where keys are labels and values are lists
        of entity texts.
    groundtruth : list[dict[str, Any]]
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
    """Run model + compute evaluation signals for one sample."""
    text = sample.get("input", "")
    output = sample.get("output", {})

    groundtruth = output.get("entities", {})
    entity_desc = output.get("entity_descriptions", {})

    raw_pred = model.extract_entities(
        text,
        entity_desc,
        include_confidence=True,
    )

    prediction, prediction_with_scores = split_gliner_output_with_scores(raw_pred)

    format_valid = is_valid_output_format(raw_pred)
    no_hallucination = has_no_hallucination(prediction, text)

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
) -> pd.DataFrame:
    """Run evaluation on full dataset and return a flat DataFrame."""
    rows = []

    for sample, (path, url) in zip(dataset, entries_metadata, strict=True):
        result = process_sample(model, sample)

        for label, metrics in result["metrics_per_class"].items():
            rows.append(
                {
                    # Add contextual information for grouping
                    "model_name": model_name,
                    "text": result["text"],
                    "json_path": str(path),
                    "url": url,
                    "label": label,
                    "groundtruth": result["groundtruth"].get(label, []),
                    "prediction": result["prediction"].get(label, []),
                    "prediction_with_scores": (
                        result["prediction_with_scores"].get(label, [])
                    ),
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

    return pd.DataFrame(rows)


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    """Safely divide two pandas Series."""
    return a / b.replace(0, pd.NA)


def compute_stats_per_label(df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics per model and per label."""
    grouped = (
        df.groupby(["model_name", "label"])
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


def compute_stats_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall metrics per model (all labels merged)."""
    grouped = (
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

    grouped["label"] = "OVERALL"

    return grouped


def compute_all_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute both per-label and overall metrics."""
    per_label = compute_stats_per_label(df)
    overall = compute_stats_overall(df)

    return pd.concat([per_label, overall], ignore_index=True)


def main(
    model_name: str,
    model_path: str | Path,
    test_dataset_path: str | Path,
    test_data_paths: str | Path,
):
    """Evaluate GLINER2 model using the specified model path."""
    # Initialize logger
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    logger = create_logger(f"logs/eval_gliner_{timestamp}.log")

    # Load finetuned model
    model = load_model(model_path, logger)

    # Get the test dataset from the JSONL file
    test_dataset = load_test_dataset_from_jsonl(test_dataset_path)
    # Load the test dataset entries and their metadata from the text file
    test_entries_metadata = load_test_dataset_entries(test_data_paths)

    # Run evaluation and build the results DataFrame
    df = build_evaluation_dataframe(
        model_name=model_name,
        model=model,
        dataset=test_dataset,
        entries_metadata=test_entries_metadata,
    )
    parquet_path = (
        Path("results/gliner/evaluation_stats")
        / f"per_text_class_metrics_{sanitize_filename(model_name)}_{timestamp}.parquet"
    )
    os.makedirs(parquet_path.parent, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    stats = compute_all_stats(df)
    stats.to_excel(
        f"results/gliner//evaluation_stats/overall_metrics_{sanitize_filename(model_name)}_{timestamp}.xlsx",
        index=False,
    )


@click.command()
@click.option(
    "--model-name",
    type=str,
    default="GLiNER2 Small (205M parameters) finetuned",
    help="Name of the model.",
)
@click.option(
    "--model-path",
    type=click.Path(),
    default="fastino/gliner2-large-v1",
    help="Path to the trained model file.",
)
@click.option(
    "--test-dataset",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default="data/gliner/test.jsonl",
    help="Path to the test dataset file.",
)
@click.option(
    "--test-data-paths",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default="data/gliner/test_paths.txt",
    help="Path to the test data paths file.",
)
def run_main_from_cli(
    model_name: str,
    model_path: str | Path,
    test_dataset: str | Path,
    test_data_paths: str | Path,
):
    """Run evaluation of GLINER2 model from the command line."""
    main(
        model_name=model_name,
        model_path=model_path,
        test_dataset_path=test_dataset,
        test_data_paths=test_data_paths,
    )


if __name__ == "__main__":
    run_main_from_cli()
