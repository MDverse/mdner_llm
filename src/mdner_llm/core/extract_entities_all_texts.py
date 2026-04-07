"""Batch extraction of structured entities from multiple texts using an LLM."""

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click

from mdner_llm.core.extract_entities import extract_entities
from mdner_llm.core.logger import create_logger
from mdner_llm.utils.common import (
    ensure_dir,
    list_json_files_from_txt,
    sanitize_filename,
)


@click.command()
@click.option(
    "--texts-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="Text file containing annotation JSON files.",
)
@click.option(
    "--model",
    required=True,
    type=str,
    help="LLM model name to use for extraction."
    "Find available models in OpenRouter (https://openrouter.ai/models).",
)
@click.option(
    "--framework",
    default="none",
    type=click.Choice(["instructor", "llamaindex", "pydanticai", "none"]),
    help="Validation framework to apply to model outputs."
    "Choices: 'instructor', 'llamaindex', 'pydanticai'. "
    "If 'none', no validation is applied and the raw model response is returned.",
)
@click.option(
    "--prompt-tag",
    default="json",
    type=click.Choice(["json", "json_with_positions"]),
    help="Descriptor indicating the format of the expected LLM output."
    "Choices: 'json' or 'json_with_positions'.",
)
@click.option(
    "--prompt-file",
    default="json_few_shot.txt",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to a text file containing the extraction prompt.",
)
@click.option(
    "--output-dir",
    default="results/llm/annotations",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path),
    help="Directory to save output files.",
    callback=ensure_dir,
)
@click.option(
    "--max-retries",
    default=3,
    type=int,
    help="Maximum number of retries in case of API or validation failure.",
)
def main(
    texts_path: Path,
    prompt_file: Path,
    model: str,
    framework: str,
    prompt_tag: str,
    output_dir: Path,
    max_retries: int,
) -> None:
    """Run entity extraction on multiple annotation files."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    logger = create_logger(
        f"logs/extract_entities_all_texts_{sanitize_filename(model)}_{framework}_{timestamp}.log"
    )
    logger.info("Starting batch entity extraction.")
    selected_files = list_json_files_from_txt(texts_path=texts_path, logger=logger)
    total_files = len(selected_files)
    # Process each file and extract entities
    start_time = datetime.now(UTC)
    for idx, file_path in enumerate(selected_files, start=1):
        try:
            extract_entities(
                prompt_tag=prompt_tag,
                prompt_file=prompt_file,
                model=model,
                text_path=file_path,
                framework=framework,
                output_dir=output_dir,
                max_retries=max_retries,
                logger=logger,
            )
        except FileNotFoundError as exc:
            logger.error(f"Input file not found ({file_path.name}): {exc}")
        except json.JSONDecodeError as exc:
            logger.error(f"Invalid JSON in {file_path.name}: {exc}")
        except KeyError as exc:
            logger.error(f"Missing required field {exc} in {file_path.name}")
        except ValueError as exc:
            logger.error(
                f"Invalid configuration while processing {file_path.name}: {exc}"
            )
        except RuntimeError as exc:
            logger.error(f"Runtime failure while processing {file_path.name}: {exc}")

        # Log progress
        percent_done = (idx / total_files) * 100
        logger.info(f"Processed {idx}/{total_files} files ({percent_done:.1f}%)")

    elapsed_time = int((datetime.now(UTC) - start_time).total_seconds())
    logger.success(
        f"Batch extraction completed successfully in {timedelta(seconds=elapsed_time)}!"
    )


if __name__ == "__main__":
    # Run entity extraction on multiple annotation files
    main()
