"""
Batch extraction of structured entities from multiple texts using an LLM.

This script iterates over a subset of annotation JSON files stored in a folder,
selects the most informative ones according to simple heuristics, and applies
the entity-extraction pipeline implemented in ``extract_entities`` to each file.

The selection strategy prioritizes:
1. Files containing at least one entity of each available type.
2. Files containing a moderate number of molecules (between 2 and 5).
3. The most recently modified files as a fallback.

For each selected input text, the script runs the chosen LLM model and framework,
then produces two output files in the specified output directory:
- a ``.json`` file containing metadata and the structured response,
- a ``.txt`` file containing the raw model output.

Both files share the same base name, which includes the input filename, model,
framework, and a timestamp.

Usage:
======
uv run src/extract_entities_all_texts.py --path-prompt PATH --model STR --path-texts PATH
                                    [--tag-prompt STR] [--framework STR]
                                    [--output-dir PATH] [--max-retries INT]

Arguments:
==========
    --path-prompt: Path
        Path to a text file containing the extraction prompt.

    --model: str
        Language model name to use for extraction find in OpenRouter page model
        (https://openrouter.ai/models). Example: "openai/gpt-4o-mini".

    --path-texts: Path
        Path to the text file containing text to annotate JSON files.
        Must include a key "raw_text" with the text content.

    --tag-prompt: str (Optional)
        Descriptor indicating the format of the expected LLM output.
        Choices: "json" or "json_with_positions".
        Default: "json"

    --framework: str (Optional)
        Validation framework to apply to model outputs.
        Choices: "instructor", "llamaindex", "pydanticai".
        Default: None (no framework)

    --output-dir: Path (Optional)
        Directory where the output JSON and text files will be saved.
        Default: "results/llm_annotations"

    --max-retries: int (Optional)
        Maximum number of retries in case of API or validation failure.
        Default: 3


Example:
========
uv run src/extract_entities_all_texts.py \
        --path-prompt prompts/json_few_shot.txt \
        --model openai/gpt-4o \
        --path-texts  results/50_selected_files_20260103_002043.txt \
        --tag-prompt json \
        --framework instructor \
        --output-dir results/llm_annotations \
        --max-retries 3

This command processes up to annotation files from
``results/50_selected_files_20260103_002043.txt`` and
saves the corresponding ``.json`` and ``.txt`` outputs
in `results/llm_annotations/{file_name}_openai_gpt-4o_instructor_YYYYMMDD_HHMMSS``.
"""

# METADATAS
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "AGPL-3.0 license"
__date__ = "2025"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import json
import sys
import time
from pathlib import Path
from typing import Any

import click
from loguru import logger
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from extract_entities import extract_entities


# FUNCTIONS
def setup_logger(loguru_logger: Any, log_dir: str | Path = "logs") -> None:
    """Configure a Loguru logger to write logs into a rotating daily log file.

    Parameters
    ----------
    loguru_logger : Any
        A Loguru logger instance (typically `loguru.logger`).
    log_dir : str or Path, optional
        Directory where log files will be stored. Default is "logs".
    """
    # Ensure log directory exists
    log_folder = Path(log_dir)
    log_folder.mkdir(parents=True, exist_ok=True)
    # Reset any previous configuration
    loguru_logger.remove()
    # Define log format
    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss}"
        "| <level>{level:<8}</level> "
        "| <level>{message}</level>"
    )
    loguru_logger.add(
        log_folder / "evaluate_json_annotations_{time:YYYY-MM-DD}.log",
        format=fmt,
        level="DEBUG",
    )
    loguru_logger.add(
        sys.stdout,
        format=fmt,
        level="DEBUG",
    )


@click.command()
@click.option(
    "--path-texts",
    required=True,
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="Text file containing annotation JSON files.",
)
@click.option(
    "--path-prompt",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the prompt file.",
)
@click.option(
    "--model",
    required=True,
    type=str,
    help="LLM model name.",
)
@click.option(
    "--framework",
    default="instructor",
    type=click.Choice(["instructor", "llamaindex", "pydanticai"]),
    help="Validation framework.",
)
@click.option(
    "--tag-prompt",
    default="json",
    type=click.Choice(["json", "json_with_positions"]),
    show_default=True,
    help="Expected output format.",
)
@click.option(
    "--output-dir",
    default="results/llm_annotations_batch",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Directory where outputs will be saved.",
)
@click.option(
    "--max-retries",
    default=3,
    type=int,
    show_default=True,
    help="Maximum number of retries for LLM calls.",
)
def extract_entities_all_texts(
    path_texts: Path,
    path_prompt: Path,
    model: str,
    framework: str,
    tag_prompt: str,
    output_dir: Path,
    max_retries: int,
) -> None:
    """
    Run entity extraction on multiple annotation files.

    This function selects up to ``nb_files`` annotation JSON files from a directory
    and applies the entity extraction pipeline to each of them using a specified
    LLM, prompt, and validation framework. Results are written to disk as both
    JSON metadata files and raw text outputs.

    Parameters
    ----------
    path_texts : Path
        Text file containing input annotation JSON files to process.
    path_prompt : Path
        Path to the text file containing the extraction prompt.
    model : str
        Identifier of the language model to use (for example
        ``"openai/gpt-4o"``).
    framework : str
        Validation framework used to guide or validate model outputs.
        Supported values are ``"instructor"``, ``"llamaindex"``,
        and ``"pydanticai"``.
    tag_prompt : str
        Descriptor indicating the expected output format of the LLM.
        Supported values are ``"json"`` and ``"json_with_positions"``.
    output_dir : Path
        Directory where extracted entities and metadata files are written.
    max_retries : int, optional
        Maximum number of retries in case of API errors or validation failures.
        Default is 3.
    """
    logger.info("Starting batch entity extraction...")

    selected_files = [
        Path(line.strip())
        for line in path_texts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print()
    start_time = time.time()
    for file_path in tqdm(
        selected_files,
        desc="Annotating texts...",
        total=len(selected_files), unit="file"
        ):
        try:
            extract_entities(
                tag_prompt=tag_prompt,
                path_prompt=path_prompt,
                model=model,
                path_text=file_path,
                framework=framework,
                output_dir=output_dir,
                max_retries=max_retries
            )
        except FileNotFoundError as exc:
            logger.error(
                f"Input file not found ({file_path.name}): {exc}"
            )

        except json.JSONDecodeError as exc:
            logger.error(
                f"Invalid JSON in {file_path.name}: {exc}"
            )

        except KeyError as exc:
            logger.error(
                f"Missing required field {exc} in {file_path.name}"
            )

        except ValueError as exc:
            logger.error(
                f"Invalid configuration while processing {file_path.name}: {exc}"
            )

        except RuntimeError as exc:
            logger.error(
                f"Runtime failure while processing {file_path.name}: {exc}"
            )
    elapsed_time: int | float = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logger.success(
        f"Batch extraction completed successfully in "
        f"{int(minutes // 60)}h {int(minutes % 60)}m {seconds:.2f}s!"
    )


# MAIN PROGRAM
if __name__ == "__main__":
    # Run entity extraction on multiple annotation files
    extract_entities_all_texts()
