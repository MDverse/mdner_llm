"""Evaluate the GLINER2 model on the test set using the best model from training."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import loguru
import pandas as pd
from gliner2 import GLiNER2

from mdner_llm.core.logger import create_logger


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


def load_test_dataset_paths(file_path: str | Path) -> list[Path]:
    """Load a text file containing paths to test data files.

    Parameters
    ----------
        file_path: Path to the text file.

    Returns
    -------
        A list of Path objects corresponding to the paths in the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    # Ensure the file path is a Path object
    path = Path(file_path)
    # Check if the file exists
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    # Read the file line by line and return a list of Path objects
    with path.open("r", encoding="utf-8") as f:
        return [Path(line.strip()) for line in f if line.strip()]


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


def load_gliner_annotations_as_dataframe(
    model_name: str,
    model: GLiNER2,
    test_dataset: list[dict[str, Any]],
    test_dataset_paths: list[Path],
    logger: "loguru.Logger" = loguru.logger,
) -> pd.DataFrame:
    """Run GLINER2 on the test dataset and return a structured DataFrame.

    Parameters
    ----------
    model_name : str
        Name of the model for logging and DataFrame purposes.
    model : GLiNER2
        The GLINER2 model used for annotation.
    test_dataset : list[dict[str, Any]]
        The test dataset loaded from the JSONL file.
    test_dataset_paths : list[Path]
        List of paths to the test data files.
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger

    Returns
    -------
    pd.DataFrame
        A DataFrame with model predictions and ground truth.
    """
    # Check for length mismatch between dataset and paths
    if len(test_dataset) != len(test_dataset_paths):
        logger.error(
            "Mismatch between dataset size "
            f"({len(test_dataset)}) and paths ({len(test_dataset_paths)})"
        )
        # Return empty DataFrame on mismatch
        return pd.DataFrame()

    rows = []
    # Iterate over the test dataset and corresponding paths
    for sample, path in zip(test_dataset, test_dataset_paths, strict=True):
        # Extract text and ground truth from the sample
        text = sample.get("input", "")
        output = sample.get("output", {})
        groundtruth = output.get("entities", [])
        entities_description = output.get("entity_descriptions", {})
        # Extract entities with gliner2
        extracted_entities = model.extract_entities(
            text, entities_description, include_confidence=True
        )
        # Format the prediction as a list of dictionaries
        prediction, confidence_score = split_gliner_output_with_scores(
            extracted_entities
        )
        rows.append(
            {
                "model_name": model_name,
                "text_to_annotate": text,
                "json_path": str(path),
                "model_response": prediction,
                "model_response_with_confidence_score": confidence_score,
                "groundtruth": groundtruth,
            }
        )

    return pd.DataFrame(rows)


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
    # Load the test dataset paths from the text file
    test_dataset_paths = load_test_dataset_paths(test_data_paths)

    # Load test annotations into a DataFrame
    df = load_gliner_annotations_as_dataframe(
        model_name=model_name,
        model=model,
        test_dataset=test_dataset,
        test_dataset_paths=test_dataset_paths,
    )
    # Save the evaluation results to a CSV file
    df.to_csv("results/gliner/evaluation_results.csv", index=False)

    # Compute confusion metrics (TP, FP, TN) by text
    # df_with_conf_metrics = compute_confusion_metrics(df_with_text, results_dir)


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
    default="results/gliner/test_100_descriptions/best",
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
