"""Extract structured entities from text using GLiNER / GLiNER2 models."""

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import loguru
from gliner import GLiNER
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
):
    """Load GLiNER or GLiNER2 checkpoint with optional adapter."""
    model_str = str(model_path)
    try:
        try:
            model = GLiNER2.from_pretrained(model_str)
        except Exception as err_g2:
            if GLiNER is not None:
                try:
                    model = GLiNER.from_pretrained(model_str)
                except Exception:
                    raise err_g2
            else:
                raise err_g2

        if adapter_path and hasattr(model, "load_adapter"):
            logger.info(f"Loading LoRA adapter from {adapter_path}")
            model.load_adapter(str(adapter_path))

        logger.success(f"Model loaded from {model_path}")
        return model
    except Exception as exc:
        logger.error(f"Model loading failed: {exc}")
        raise


def load_metadata(path: Path) -> list[tuple[Path, str]]:
    """Load metadata mapping."""
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
    metadata_path: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> list[tuple[str, dict[str, str], ListOfEntities, Path, str]]:
    """Load samples and ground truth."""
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
                            {"category": category, "text": t}
                            for category, texts in groundtruth.items()
                            for t in texts
                        ]
                    }
                )
            except ValueError as exc:
                logger.error(f"Failed to normalize groundtruth: {exc}")
                normalized_gt = ListOfEntities(entities=[])
            entity_desc = output.get("entity_descriptions", {})
            json_path, url = metadata[idx]
            samples.append((text, entity_desc, normalized_gt, json_path, url))

    return samples


def run_gliner(model, text: str, entity_desc: dict) -> tuple[dict, float, int, int]:
    """Run extraction using appropriate API."""
    start = time.perf_counter()
    if hasattr(model, "extract_entities"):
        predictions = model.extract_entities(text, entity_desc, include_confidence=True)
    else:
        labels = list(entity_desc.keys())
        ents = model.predict_entities(text, labels, threshold=0.5)
        entities_by_cat: dict[str, list[dict]] = {}
        for ent in ents:
            entities_by_cat.setdefault(ent["label"], []).append(
                {"text": ent["text"], "confidence": ent.get("score", 1.0)}
            )
        predictions = {"entities": entities_by_cat}

    elapsed = time.perf_counter() - start
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
    """Run inference on one document and write output JSON."""
    predictions, inference_time, in_tokens, out_tokens = run_gliner(model, text, entity_desc)
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
        formatted_response = ListOfEntities(entities=[])
        status = "format_error"
    else:
        status = "ok"

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    out_file = output_dir / f"{text_path.stem}_{sanitize_filename(text_path.stem)}_{ts}.json"

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
    """Run batch entity extraction."""
    logger.info("Starting batch entity extraction.")
    test_samples = load_sample(text_path, metadata_path, logger=logger)
    model = load_model(model_path, adapter_path, logger=logger)

    canonical_name = model_name_id or str(model_path)
    start_time = datetime.now(UTC)

    for idx, (text, entity_desc, groundtruth, json_path, url) in enumerate(test_samples, start=1):
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
        except Exception as exc:
            logger.error(f"Error processing {json_path.name}: {exc}")

        total_files = len(test_samples)
        logger.info(f"Processed {idx}/{total_files} files ({(idx / total_files) * 100:.1f}%)")

    elapsed_time = int((datetime.now(UTC) - start_time).total_seconds())
    logger.success(f"Batch extraction completed in {timedelta(seconds=elapsed_time)}!")


@click.command()
@click.option("--text-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--metadata-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model-path", type=click.Path(), required=True)
@click.option("--adapter-path", type=click.Path(path_type=Path), default=None)
@click.option("--model-name-id", type=str, default=None, help="Explicit label for model in metadata CSV.")
@click.option("--output-dir", type=click.Path(path_type=Path), callback=ensure_dir, required=True)
def run_main_from_cli(
    text_path: Path,
    metadata_path: Path,
    model_path: Path,
    output_dir: Path,
    adapter_path: str | Path | None,
    model_name_id: str | None,
) -> None:
    """CLI entrypoint."""
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