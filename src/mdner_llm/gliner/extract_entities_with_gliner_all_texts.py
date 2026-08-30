"""Extract structured entities from text using GLiNER / GLiNER2 models."""

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import loguru
from gliner import GLiNER
from gliner2 import AutoExtractor
from gliner2.processor import WhitespaceTokenSplitter

from mdner_llm.common import ensure_dir, sanitize_filename
from mdner_llm.core.extract_entities_with_llm import (
    save_formated_response_with_metadata_to_json,
)
from mdner_llm.logger import create_logger
from mdner_llm.models.entities import ListOfEntities


def load_model(
    model_path: str | Path,
    adapter_path: str | Path | None,
    logger: "loguru.Logger" = loguru.logger,
):
    """Load GLiNER or GLiNER2 checkpoint with optional adapter.

    Returns
    -------
        Loaded model instance.

    Raises
    ------
    OSError
        If the model or adapter fails to load from the specified path.
    ValueError
        If the model configuration is invalid.
    """
    model_str = str(model_path)
    try:
        # Attempt to load GLiNER2/2.5 model first.
        model = AutoExtractor.from_pretrained(model_str)
    except (OSError, ValueError) as err_g2:
        try:
            # Fallback to GLiNER model if GLiNER2 fails.
            model = GLiNER.from_pretrained(model_str)
        except (OSError, ValueError) as err_gliner:
            logger.error(f"Model loading failed: {err_gliner}")
            raise err_g2 from err_gliner
    # Load LoRA adapter if specified and supported by the model.
    if adapter_path and hasattr(model, "load_adapter"):
        logger.info(f"Loading LoRA adapter from {adapter_path}.")
        model.load_adapter(str(adapter_path))
    logger.success(f"Model loaded from {model_path}.")
    return model


def load_metadata(path: Path) -> list[tuple[Path, str]]:
    """Load metadata mapping from a TSV file.

    Returns
    -------
        List of tuples mapping JSON paths to their source URLs.

    Raises
    ------
        OSError
            If the metadata file cannot be read.
        ValueError
            If a line does not contain the expected tab-separated values.
    """
    metadata = []
    # Read metadata TSV file.
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            # Split line into JSON path and URL
            # Expecting exactly two tab-separated values per line.
            json_path, url = line.split("\t")
            # Add it to the metadata list.
            metadata.append((Path(json_path), url))
    return metadata


def load_sample(
    jsonl_path: Path,
    metadata_path: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> list[tuple[str, dict[str, str], ListOfEntities, Path, str]]:
    """Load samples and ground truth annotations.

    Returns
    -------
        List of sample tuples containing text, entity descriptions,
        ground truth, path, and URL.

    Raises
    ------
        OSError
            If the JSONL or metadata file cannot be read.
        json.JSONDecodeError
            If a line in the JSONL file is not valid JSON.
        IndexError
            If metadata entries do not match the number of samples.
    """
    samples = []
    # Load path and URL metadata for each sample.
    metadata = load_metadata(metadata_path)
    # Read JSONL file containing text samples and their annotations.
    # Each line is expected to be a valid JSON object.
    with jsonl_path.open(encoding="utf-8") as file:
        for idx, line in enumerate(file):
            if not line.strip():
                continue
            data = json.loads(line)
            # Extract text, output annotations, and ground truth entities.
            text = data.get("input", "")
            output = data.get("output", {})
            groundtruth = output.get("entities", {})
            try:
                # Normalize ground truth into ListOfEntities.
                normalized_gt = ListOfEntities.model_validate(
                    {
                        "entities": [
                            {"category": category, "text": text}
                            for category, texts in groundtruth.items()
                            for text in texts
                        ]
                    }
                )
            except ValueError as exc:
                logger.error(f"Failed to normalize groundtruth: {exc}")
                # If normalization fails, use an empty ListOfEntities.
                normalized_gt = ListOfEntities(entities=[])
            # Extract entity descriptions for the sample.
            entity_desc = output.get("entity_descriptions", {})
            # Get the corresponding JSON path and URL from the metadata.
            json_path, url = metadata[idx]
            samples.append((text, entity_desc, normalized_gt, json_path, url))
    return samples


def run_gliner(model, text: str, entity_desc: dict) -> tuple[dict, float, int, int]:
    """Run extraction using the appropriate model API.

    Returns
    -------
        Predictions dictionary, elapsed time in seconds, input token count,
        and output token count.

    Raises
    ------
        AttributeError
            If the model object lacks the required prediction methods.
        ValueError
            If prediction inputs are malformed.
    """
    start = time.perf_counter()
    # Extract entities with gliner2/2.5
    if hasattr(model, "extract_entities"):
        predictions = model.extract_entities(text, entity_desc, include_confidence=True)
    # or extract entities with gliner
    else:
        labels = list(entity_desc.keys())
        ents = model.predict_entities(text, labels, threshold=0.5)
        entities_by_cat = {}
        for ent in ents:
            entities_by_cat.setdefault(ent["label"], []).append(
                {"text": ent["text"], "confidence": ent.get("score", 1.0)}
            )
        predictions = {"entities": entities_by_cat}
    # Measure elapsed time.
    elapsed = time.perf_counter() - start
    # Count tokens using whitespace splitting.
    # Example: "Hello world!" -> ["Hello", "world!"] -> 2 tokens
    tokenizer = WhitespaceTokenSplitter()
    input_tokens = len(list(tokenizer(text)))
    output_tokens = len(list(tokenizer(json.dumps(predictions))))
    return predictions, elapsed, input_tokens, output_tokens


def extract_entities_with_gliner(
    model,
    model_name_id: str,
    text: str,
    entity_desc: dict[str, str],
    groundtruth: ListOfEntities,
    text_path: Path,
    url: str,
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Run inference on one document and write output JSON.

    Raises
    ------
        OSError
            If writing the output file fails.
    """
    # Run GLiNER/GLiNER2 inference and measure performance.
    predictions, inference_time, in_tokens, out_tokens = run_gliner(
        model, text, entity_desc
    )
    # Format the predictions into a structured ListOfEntities model.
    try:
        formatted_response = ListOfEntities.model_validate(
            {
                "entities": [
                    {"category": category, "text": ent["text"]}
                    for category, ents in predictions.get("entities", {}).items()
                    for ent in ents
                ]
            }
        )
    except ValueError as exc:
        logger.error(f"Failed to format response: {exc}")
        # Fallback to an empty ListOfEntities if formatting fails.
        formatted_response = ListOfEntities(entities=[])
        # Set status to indicate a formatting error occurred.
        status = "format_error"
    else:
        status = "ok"
    # Prepare output directory and file path for saving results.
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    out_file = (
        output_dir / f"{text_path.stem}_{sanitize_filename(text_path.stem)}_{ts}.json"
    )
    response_metadata = {
        "timestamp": ts,
        "input_json_path": str(text_path),
        "text": text,
        "url": url,
        "model_name": model_name_id,
        "framework_name": "noframework",
        "groundtruth": groundtruth.model_dump(),
        "status": status,
        "formatted_response": formatted_response.model_dump(),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "inference_time_sec": inference_time,
        "inference_cost_usd": 0.0,
    }
    # Save the formatted response and metadata to a JSON file.
    save_formated_response_with_metadata_to_json(out_file, response_metadata, logger)


def extract_entities_with_gliner_all_texts(
    text_path: Path,
    metadata_path: Path,
    model_path: str,
    output_dir: Path,
    adapter_path: str | Path | None,
    model_name_id: str | None = None,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Run batch entity extraction over all loaded texts.

    Raises
    ------
        OSError
            If reading input paths or loading the model fails.
    """
    logger.info("Starting batch entity extraction.")
    test_samples = load_sample(text_path, metadata_path, logger=logger)
    model = load_model(model_path, adapter_path, logger=logger)
    # Determine a canonical name for the model.
    # Useful when the model_path is a local path.
    canonical_name = model_name_id or str(model_path)
    start_time = datetime.now(UTC)
    # Process each sample and extract entities.
    for idx, (text, entity_desc, groundtruth, json_path, url) in enumerate(
        test_samples, start=1
    ):
        try:
            extract_entities_with_gliner(
                model=model,
                model_name_id=canonical_name,
                text=text,
                entity_desc=entity_desc,
                groundtruth=groundtruth,
                text_path=json_path,
                output_dir=output_dir,
                url=url,
                logger=logger,
            )
        except (OSError, ValueError, TypeError) as exc:
            logger.error(f"Error processing {json_path.name}: {exc}")
        # Log progress.
        total_files = len(test_samples)
        logger.info(
            f"Processed {idx}/{total_files} files ({(idx / total_files) * 100:.1f}%)"
        )
    elapsed_time = int((datetime.now(UTC) - start_time).total_seconds())
    logger.success(f"Batch extraction completed in {timedelta(seconds=elapsed_time)}!")


@click.command()
@click.option(
    "--text-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to JSONL file containing text samples.",
)
@click.option(
    "--metadata-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to TSV file mapping JSON paths to URLs.",
)
@click.option(
    "--model-path",
    type=click.Path(),
    required=True,
    help="Path to GLiNER or GLiNER2 model checkpoint.",
)
@click.option(
    "--adapter-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to the adapter checkpoint if using LoRA fine-tuning.",
)
@click.option(
    "--model-name-id",
    type=str,
    default=None,
    help="Explicit label for model evaluation.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    callback=ensure_dir,
    required=True,
    help="Directory to save output JSON files.",
)
def run_main_from_cli(
    text_path: Path,
    metadata_path: Path,
    model_path: Path,
    output_dir: Path,
    adapter_path: str | Path | None,
    model_name_id: str | None,
) -> None:
    """Run CLI entrypoint for batch entity extraction."""
    logger = create_logger(level="INFO")
    extract_entities_with_gliner_all_texts(
        text_path=text_path,
        metadata_path=metadata_path,
        model_path=str(model_path),
        output_dir=output_dir,
        adapter_path=adapter_path,
        model_name_id=model_name_id,
        logger=logger,
    )


if __name__ == "__main__":
    run_main_from_cli()
