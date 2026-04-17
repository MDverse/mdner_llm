"""Batch extraction of structured entities from multiple texts using an LLM."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import loguru

from mdner_llm.core.extract_entities_with_llm import extract_entities
from mdner_llm.utils.common import (
    ensure_dir,
    list_json_files_from_txt,
    sanitize_filename,
)
from mdner_llm.utils.logger import create_logger


def extract_entities_all_texts(
    texts_path: Path,
    prompt_file: Path,
    model: str,
    framework: str,
    output_dir: Path,
    max_retries: int,
    logger: "create_logger" = loguru.logger,
) -> None:
    """Run entity extraction on multiple annotation files."""
    logger.info("Starting batch entity extraction.")
    selected_files = list_json_files_from_txt(texts_path=texts_path, logger=logger)
    total_files = len(selected_files)
    # Process each file and extract entities
    start_time = datetime.now(UTC)
    for idx, file_path in enumerate(selected_files, start=1):
        try:
            extract_entities(
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
    type=click.Choice(["instructor", "pydanticai", "noframework"]),
    help="Validation framework to apply to model outputs."
    "Choices: 'instructor', 'pydanticai', 'noframework'.",
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
def run_main_from_cli(
    texts_path: Path,
    model: str,
    framework: str,
    prompt_file: Path,
    output_dir: Path,
    max_retries: int,
) -> None:
    """CLI entrypoint."""
    # Initialize logger with a unique log file for this run
    logger = create_logger(
        f"logs/extract_entities_all_texts_{sanitize_filename(model)}_{framework}.log"
    )
    extract_entities_all_texts(
        texts_path=texts_path,
        model=model,
        framework=framework,
        prompt_file=prompt_file,
        output_dir=output_dir,
        max_retries=max_retries,
        logger=logger,
    )


if __name__ == "__main__":
    # Run entity extraction on multiple annotation files
    run_main_from_cli()
