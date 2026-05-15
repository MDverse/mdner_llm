"""Extract structured entities from a text using GLINER2 model."""

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import loguru
from gliner2 import GLiNER2
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
) -> GLiNER2:
    """Load a GLiNER2 model from disk.

    Returns
    -------
        GLiNER2: The loaded model instance.
    """
    try:
        model = GLiNER2.from_pretrained(model_path)
        if adapter_path:
            logger.info(
                f"Loading LoRA adapter from {adapter_path} and applying to base model."
            )
            model.load_adapter(adapter_path)
        logger.success(f"Model loaded from {model_path}")
    except Exception as exc:
        logger.error(f"Model loading failed: {exc}")
        raise
    return model


def load_metadata(path: Path) -> list[tuple[str, str]]:
    """Load metadata file mapping json_path -> url.

    Returns
    -------
        list[tuple[str, str]]: List of (json_path, url) tuples.
    """
    metadata = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            json_path, url = line.split("\t")
            metadata.append((Path(json_path), url))
    return metadata


def load_sample(
    jsonl_path: Path,
    metadata_path: Path = Path("data/gliner/test_metadata.txt"),
    logger: "loguru.Logger" = loguru.logger,
) -> list[tuple[str, dict[str, str], dict[str, list[str]], str, str]]:
    """Load samples + metadata.

    Returns
    -------
        list[tuple[str, dict[str, str], dict[str, list[str]], str, str]]:
        List of tuples containing (text, entity_desc, groundtruth, json_path, url).
    """
    samples = []
    metadata = load_metadata(metadata_path)

    with jsonl_path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            data = json.loads(line)
            text = data.get("input", "")
            output = data.get("output", {})
            groundtruth = output.get("entities", {})
            try:
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
                logger.error(
                    f"Failed to normalize groundtruth in {metadata[idx][0].name}: {exc}"
                )
                normalized_gt = ListOfEntities(entities=[])
            entity_desc = output.get("entity_descriptions", {})
            json_path, url = metadata[idx]
            samples.append((text, entity_desc, normalized_gt, json_path, url))

    logger.info(f"Loaded {len(samples)} samples with metadata.")
    return samples


def run_gliner(
    model: GLiNER2,
    text: str,
    entity_desc: dict,
) -> tuple[list[dict], float, int, int]:
    """Run GLiNER inference.

    Returns
    -------
        tuple: A tuple containing the list of predicted entities with confidence scores,
        the time taken for inference in seconds and the input/output tokens counts.
    """
    start = time.perf_counter()
    predictions = model.extract_entities(
        text,
        entity_desc,
        include_confidence=True,
    )
    elapsed = time.perf_counter() - start
    # Count output tokens using a simple whitespace tokenizer
    tokenizer = WhitespaceTokenSplitter()
    input_tokens = len(list(tokenizer(text)))
    output_tokens = len(list(tokenizer(json.dumps(predictions))))
    return predictions, elapsed, input_tokens, output_tokens


def save_json(
    json_output_path: Path,
    json_data: dict,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Save structured output with metadata.

    Raises
    ------
    FileNotFoundError
        If the parent directory does not exist.
    OSError
        If a system-level error occurs during writing.
    ValueError
        If the data cannot be serialized to JSON.
    """
    try:
        json_output_path.write_text(
            json.dumps(json_data, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.success(
            f"Saved formated response with metadata to {json_output_path} successfully."
        )
    except FileNotFoundError as exc:
        msg = f"Directory does not exist for output file: {json_output_path}"
        raise FileNotFoundError(msg) from exc
    except OSError as exc:
        msg = f"Failed to write JSON to {json_output_path}: {exc}"
        raise OSError(msg) from exc
    except TypeError as exc:
        msg = f"Invalid data provided for JSON serialization: {exc}"
        raise ValueError(msg) from exc


def extract_entities_with_gliner(
    model: GLiNER2,
    model_path: str | Path,
    adapter_path: str | Path | None,
    text: str,
    entity_desc: dict[str, str],
    groundtruth: dict,
    text_path: Path,
    json_path: Path,
    url: str,
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Extract entities using GLiNER."""
    logger = create_logger()
    predictions, inference_time, in_tokens, out_tokens = run_gliner(
        model, text, entity_desc
    )
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
        logger.error(f"Failed to format GLiNER response: {exc}")
        formatted_response = ListOfEntities(entities=[])
        status = "format_error"
    else:
        status = "ok"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    json_path = (
        output_dir / f"{text_path.stem}_{sanitize_filename(text_path.stem)}_{ts}.json"
    )
    if adapter_path:
        path = Path(adapter_path)
        model_name = f"{path.parents[1].stem}_{path.parent.stem}"
    else:
        model_name = str(model_path)
    response_metadata = {
        "timestamp": ts,
        "input_json_path": str(text_path),
        "text": text,
        "url": url,
        "model_name": model_name,
        "framework_name": "noframework",
        "groundtruth": groundtruth.model_dump(),
        "status": status,
        "formatted_response": formatted_response.model_dump(),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "inference_time_sec": inference_time,
        "inference_cost_usd": 0.0,
    }
    save_formated_response_with_metadata_to_json(json_path, response_metadata, logger)


def extract_entities_with_gliner_all_texts(
    text_path: Path,
    model_path: str,
    output_dir: Path,
    adapter_path: str | Path | None,
    logger: "create_logger" = loguru.logger,
) -> None:
    """Run entity extraction on multiple annotation files."""
    logger.info("Starting batch entity extraction.")
    test_samples = load_sample(text_path, logger=logger)
    model = load_model(model_path, adapter_path, logger=logger)
    # Process each file and extract entities
    start_time = datetime.now(UTC)
    for idx, (text, entity_desc, groundtruth, json_path, url) in enumerate(
        test_samples, start=1
    ):
        try:
            extract_entities_with_gliner(
                model=model,
                model_path=model_path,
                adapter_path=adapter_path,
                text=text,
                entity_desc=entity_desc,
                groundtruth=groundtruth,
                text_path=Path(json_path),
                output_dir=output_dir,
                json_path=json_path,
                url=url,
                logger=logger,
            )

        except KeyError as exc:
            logger.error(f"Missing required field {exc} in {json_path.name}")
        except ValueError as exc:
            logger.error(
                f"Invalid configuration while processing {json_path.name}: {exc}"
            )
        except RuntimeError as exc:
            logger.error(f"Runtime failure while processing {json_path.name}: {exc}")

        # Log progress
        total_files = len(test_samples)
        percent_done = (idx / total_files) * 100
        logger.info(f"Processed {idx}/{total_files} files ({percent_done:.1f}%)")

    elapsed_time = int((datetime.now(UTC) - start_time).total_seconds())
    logger.success(
        f"Batch extraction completed successfully in {timedelta(seconds=elapsed_time)}!"
    )


@click.command()
@click.option(
    "--text-path",
    type=click.Path(exists=True, path_type=Path),
    default="data/gliner/test.jsonl",
    help=(
        "Path to a text file containing paths to JSON files with input texts. "
        "Each line in the text file should be a valid path to a JSON file. "
    ),
)
@click.option(
    "--model-path",
    type=click.Path(),
    default="fastino/gliner2-base-v1",
    help=(
        "Path to the trained model file or model identifier from Hugging Face Hub. "
        "Hugging Face example: 'fastino/gliner2-base-v1', 'fastino/gliner2-large-v1'."
        "Local file example: 'results/gliner/models/gliner2-finetuned-small/best'."
    ),
)
@click.option(
    "--adapter-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "Path to the LoRA adapter model. "
        "The script will attempt to load the adapter and apply it to the base model."
    ),
)
@click.option(
    "--output-dir",
    default="results/gliner/annotations",
    type=click.Path(path_type=Path),
    callback=ensure_dir,
)
def run_main_from_cli(
    text_path: Path, model_path: Path, output_dir: Path, adapter_path: str | Path | None
) -> None:
    """CLI entrypoint."""
    logger = create_logger(level="INFO")
    extract_entities_with_gliner_all_texts(
        text_path=text_path,
        model_path=model_path,
        output_dir=output_dir,
        adapter_path=adapter_path,
        logger=logger,
    )


if __name__ == "__main__":
    run_main_from_cli()
