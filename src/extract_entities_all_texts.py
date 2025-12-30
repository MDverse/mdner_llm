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
uv run src/extract_entities_all_texts.py --path-prompt PATH --model STR
                                    --path-folder-texts PATH [--tag-prompt STR]
                                    [--framework STR] [--output-dir PATH]
                                    [--max-retries INT] [--nb-files INT]

Arguments:
==========
    --path-prompt: Path
        Path to a text file containing the extraction prompt.

    --model: str
        Language model name to use for extraction find in OpenRouter page model
        (https://openrouter.ai/models). Example: "openai/gpt-4o-mini".

    --path-folder-texts: Path
        Path to the directory containing text to annotate JSON files.
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

    --nb-files: int (Optional)
        maximum number of files to process.
        Default: 50


Example:
========
uv run src/extract_entities_all_texts.py \
        --path-prompt prompts/json_few_shot.txt \
        --model openai/gpt-4o \
        --path-folder-texts annotations/v2 \
        --tag-prompt json \
        --framework instructor \
        --output-dir results/llm_annotations \
        --max-retries 3
        --nb-files 50

This command processes up to 50 annotation files from ``annotations/v2`` and
saves the corresponding ``.json`` and ``.txt`` outputs in `results/llm_annotations`.
{file_name}_openai_gpt-4o_instructor_YYYYMMDD_HHMMSS`
"""

# METADATAS
import time
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "AGPL-3.0 license"
__date__ = "2025"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import json
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd
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


def select_annotation_files(
    annotations_dir: Path,
    nb_files: int,
    tsv_path: Path = Path("results/all_annotations_entities_count.tsv"),
) -> list[Path]:
    """
    Select informative annotation JSON files from a directory.

    Priority:
    1. Files with all entity types present.
    2. Files with 2-5 molecules.
    3. Most recent files.

    Parameters
    ----------
    annotations_dir : Path
        Directory containing annotation JSON files.
    nb_files : int
        Maximum number of files to select.
    tsv_path : Path
        TSV file containing entity counts.

    Returns
    -------
    list[Path]
        Selected annotation file paths.

    Raises
    ------
    ValueError
        If no JSON files are found or the TSV file is invalid.
    """
    logger.info(f"Selecting text to annotate from {annotations_dir}...")
    # Load entity count table (one row per annotation file)
    df = pd.read_csv(tsv_path, sep="\t")

    # Ensure the TSV can be matched to JSON filenames
    if "filename" not in df.columns:
        msg = "TSV file must contain a 'filename' column"
        raise ValueError(msg)

    # List all available annotation JSON files, sorted by recency
    json_files = sorted(
        annotations_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not json_files:
        msg = f"No JSON files found in {annotations_dir}"
        raise ValueError(msg)

    # Map filenames to paths for fast lookup
    file_map = {p.name: p for p in json_files}

    # Accumulator for selected filenames (keeps insertion order)
    selected: list[str] = []

    # Identify entity-count columns (excluding SOFTVERS)
    entity_cols = [
        col for col in df.columns
        if col.endswith("_nb") and col != "SOFTVERS_nb"
    ]

    # Priority 1: files containing at least one instance of each entity type
    if entity_cols:
        df_all = df[(df[entity_cols] > 0).all(axis=1)]
        selected.extend(
            fname for fname in df_all["filename"]
            if fname in file_map
        )

    # Priority 2: files with a moderate number of molecules (2-5)
    # Applied only if the selection is still incomplete
    if "MOLECULE_nb" in df.columns and len(selected) < nb_files:
        df_mol = df[
            (df["MOLECULE_nb"] >= 2) & (df["MOLECULE_nb"] <= 5)
        ]
        selected.extend(
            fname for fname in df_mol["filename"]
            if fname in file_map and fname not in selected
        )

    # Priority 3: fill remaining slots with the most recent files
    if len(selected) < nb_files:
        selected.extend(
            fname for fname in file_map
            if fname not in selected
        )

    selected_files = [file_map[name] for name in selected[:nb_files]]
    logger.success(f"Selected {len(selected_files)} files successfully!")
    # Return paths, truncated to the requested number of files
    return selected_files


@click.command()
@click.option(
    "--path-folder-texts",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing annotation JSON files.",
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
    "--nb-files",
    default=50,
    type=int,
    show_default=True,
    help="Number of annotation files to process.",
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
    path_folder_texts: Path,
    path_prompt: Path,
    model: str,
    framework: str,
    nb_files: int,
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
    path_folder_texts : Path
        Directory containing input annotation JSON files to process.
    path_prompt : Path
        Path to the text file containing the extraction prompt.
    model : str
        Identifier of the language model to use (for example
        ``"openai/gpt-4o"``).
    framework : str
        Validation framework used to guide or validate model outputs.
        Supported values are ``"instructor"``, ``"llamaindex"``,
        and ``"pydanticai"``.
    nb_files : int
        Maximum number of annotation files to process.
        Default is 50.
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

    selected_files = select_annotation_files(
        annotations_dir=path_folder_texts,
        nb_files=nb_files
    )
    start_time = time.time()
    for file_path in tqdm(selected_files, total=len(selected_files)):
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
