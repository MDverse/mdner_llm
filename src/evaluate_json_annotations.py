"""
Evaluate and benchmark annotations produced by several LLMs on molecular-dynamics texts.

This script loads the *N most recent* annotation files from `annotations/v2` and
evaluates how well different language models (e.g., GPT-4, Gemini, MoonshotAI Kimik2,
Qwen, Meta Llama 3.1/3.3) extract structured entities from molecular-dynamics descriptions.

Each model is tested under four configurations:
- `no_validation`
- `validation_instructor`
- `validation_llamaindex`
- `validation_pydanticai`

For every model output, the script checks:
1. **JSON format validity**
   The response must match the expected schema:
   `{"entities": [ {"label": <str>, "text": <str>, ...}, ... ]}`
   (and optionally character positions when using `json_with_positions`).

2. **Hallucination detection**
   Each extracted entity must correspond to text actually present in the source document.

3. **Annotation correctness**
   The model’s entities are compared against expert-validated annotations.
   Each prediction is tagged as correct or incorrect.

All detailed per-response results for each model are saved as **Parquet files**.
A final **Excel (.xlsx)** summary aggregates statistics across models, enabling comparisons on:
- format adherence,
- hallucination rate,
- annotation accuracy.

This tool is designed to benchmark LLM reliability when producing structured, domain-specific annotations.

Usage:
=======
    uv run src/evaluate_json_annotations.py [--annotations-dir PATH]
                                            [--nb-annotations INT]
                                            [--tag-prompt 'json|json_with_positions']
                                            [--results-dir PATH]

Arguments:
==========
    --annotations-dir: PATH
        Directory containing the annotation JSON files to evaluate.
        The script automatically selects the most recent samples.
        Default: "annotations/v2"

    --nb-annotations: INT
        Number of recent annotation files to process.
        Default: 10

    --tag-prompt: STR
        Output format expected from the model:
        - "json": only label and text
        - "json_with_positions": label, text, start, end
        Default: "json"

    --results-dir: PATH
        Directory where all evaluation outputs (Parquet files+summary Excel) are saved.
        Default: "results/json_evaluation_stats/<timestamp>"

Example:
========
    uv run src/evaluate_json_annotations.py \
        --annotations-dir annotations/v2 \
        --nb-annotations 50 \
        --tag-prompt json \
        --results-dir results/json_evaluation_stats/test

This command will evaluate the 50 most recent annotation files found in
`annotations/v2`, run all models on them using the JSON prompt format, and save
all per-model parquet files plus a global XLSX summary inside
`results/json_evaluation_stats/test`.
"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "AGPL-3.0 license"
__date__ = "2025"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import click
import instructor
import pandas as pd
from dotenv import load_dotenv
from instructor.core import (
    InstructorRetryException,
)
from instructor.core import (
    ValidationError as InstructorValidationError,
)
from instructor.core.client import Instructor
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from loguru import logger
from openai.types.chat import ChatCompletion
from pydantic import ValidationError as PydanticValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_core import ValidationError as CoreValidationError
from tqdm import tqdm

# UTILITY IMPORTS
from utils import (
    PROMPT_JSON,
    PROMPT_POSITIONS,
    ListOfEntities,
    ListOfEntitiesPositions
)

# CONSTANTS
# To adapt on your needs :
MODELS_OPENAI = [
    # "o3-mini-2025-01-31",
    "gpt-4o-mini-2024-07-18",
    "gpt-5-mini-2025-08-07",
]
MODELS_OPENROUTER = [
    "meta-llama/llama-3.1-8b-instruct",
    # "meta-llama/llama-3-70b-instruct",
    "moonshotai/kimi-k2-thinking",
    "google/gemini-2.5-flash",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-chat-v3-0324",
]
MODELS_OPENROUTER = []


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


def ensure_dir(ctx, param, value: Path) -> Path:
    """
    Create the directory if it does not already exist.

    Callback for Click options to ensure the provided path
    is a valid directory. Behaves like `mkdir -p`.

    Parameters
    ----------
    ctx : click.Context
        The Click context for the current command invocation.
        (Required by Click callbacks but unused in this function.)
    param : click.Parameter
        The Click parameter associated with this callback.
        (Required by Click callbacks but unused in this function.)
    value : Path
        The directory path provided by the user, already converted
        into a `pathlib.Path` object by Click.

    Returns
    -------
    Path
        The same path, after ensuring the directory exists.
    """
    value.mkdir(parents=True, exist_ok=True)
    return value


def assign_all_instructor_clients() -> dict[str, Instructor]:
    """
    Assign a client to each model.

    Returns
    -------
    Dict[str, Instructor]
        Dictionary mapping model names to Instructor clients.
    """
    load_dotenv()
    models_with_providers = dict.fromkeys(MODELS_OPENAI, "openai")
    models_with_providers.update(dict.fromkeys(MODELS_OPENROUTER, "openrouter"))
    return {
        model: instructor.from_provider(
            f"{provider}/{model}", mode=instructor.Mode.JSON
        )
        for model, provider in models_with_providers.items()
    }


def assign_all_llamaindex_clients() -> dict[str, OpenAI | OpenRouter]:
    """
    Assign a LlamaIndex client to each model.

    Returns
    -------
    Dict[str, OpenAI | OpenRouter]
        Dictionary mapping model names to LlamaIndex OpenAI | OpenRouter clients.
    """
    load_dotenv()
    clients = {}
    for model_name in MODELS_OPENROUTER:
        llm = OpenRouter(
            model=model_name,
        )
        clients[model_name] = llm.as_structured_llm(output_cls=ListOfEntities)

    for model_name in MODELS_OPENAI:
        llm = OpenAI(
            model=model_name,
        )
        clients[model_name] = llm.as_structured_llm(output_cls=ListOfEntities)

    return clients


def assign_all_pydanticai_clients() -> dict[str, OpenAIChatModel]:
    """
    Assign a PydanticAI for each PydanticAI model.

    Returns
    -------
    Dict[str, OpenAIChatModel]
        Dictionary mapping model names to PydanticAI clients.
    """
    load_dotenv()
    clients = {}
    api_key = os.getenv("OPENROUTER_API_KEY")
    for model_name in MODELS_OPENROUTER:
        llm = OpenAIChatModel(model_name, provider=OpenRouterProvider(api_key=api_key))
        clients[model_name] = llm

    for model_name in MODELS_OPENAI:
        llm = OpenAIChatModel(model_name)
        clients[model_name] = llm

    return clients


def load_recent_annotations(
    annotations_dir: Path,
    nb_files: int,
    tag_prompt: str,
) -> list[dict[str, Any]]:
    """
    Load the `nb_files` most recently modified annotation JSON files.

    The function:
        - lists JSON files in the directory,
        - sorts them by modification date (descending),
        - loads at most `nb_files`,
        - extracts:
            * file_path: Path
            * text_to_annotate: str
            * groundtruth: ListOfEntities or ListOfEntitiesPositions
        - removes position fields if `tag_prompt == "json"`.

    Parameters
    ----------
    annotations_dir : Path
        Directory containing annotation JSON files.
    nb_files : int
        Maximum number of recent files to load.
    tag_prompt : str
        Determines whether returned entities contain character positions.
        Must be one of: "json", "json_with_positions".

    Returns
    -------
    list[dict]
        Annotation records sorted by modification date (newest first).

    Raises
    ------
    ValueError
        If the directory contains no JSON files, or
        if a file is missing required fields.
    """
    logger.info("Loading annotations and groundtruth records.")
    json_files = sorted(
        (p for p in annotations_dir.glob("*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not json_files:
        msg = f"No JSON files found in {annotations_dir}"
        raise ValueError(msg)

    selected_files = json_files[:nb_files]
    records = []

    for file_path in selected_files:
        try:
            with file_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.warning(f"Invalid JSON in {file_path}: {exc}")
            continue

        raw_text = data.get("raw_text")
        entities_with_positions = data.get("entities")

        # Remove start/end if tag_prompt = "json"
        if tag_prompt == "json":
            entities = [
                {
                    "label": ent.get("label"),
                    "text": ent.get("text"),
                }
                for ent in entities_with_positions
            ]
            groundtruth = ListOfEntities(entities)
        else:  # json_with_positions
            groundtruth = ListOfEntitiesPositions(entities_with_positions)

        records.append(
            {
                "file_path": file_path,
                "text_to_annotate": raw_text,
                "groundtruth": groundtruth,
            }
        )

    return records


def annotate(
    text: str,
    model: str,
    client: Instructor | OpenAI | OpenRouter | OpenAIChatModel,
    tag_prompt: str,
    validator: str = "instructor",
    max_retries: int = 3,
    validation: bool = True,
) -> ChatCompletion | str:
    """Annotate the given text using the specified model.

    If validation is True, the output will be validated against the GlobalResponse schema.

    Parameters
    ----------
    text : str
        The text to annotate.
    model : str
        The name of the LLM model to use.
    client : Union[Instructor | OpenAI | OpenRouter | OpenAIChatModel]
        The LLM client to use (either from Instructor, llamaindex or pydantic_ai).
    validator: str, optional
        The name of the output validator package between "instructor", "llamaindex", "pydanticai" (Default is "instructor").
    validation : bool, optional
        Whether to validate the output against the schema, by default True
    max_retries : int, optional
        Maximum number of retries for the API call in case of failure, by default 3

    Returns
    -------
    Union[ListOfEntities,ChatCompletion]
        The response from the LLM model, either validated or raw output.
    """
    # Set response model and retries based on validation flag
    if validation:
        response_model = ListOfEntities
    else:
        response_model = None
        max_retries = 0

    # Set prompt based on positions of the start and end or without
    prompt = PROMPT_JSON if tag_prompt == "json" else PROMPT_POSITIONS

    result = None
    # Query the LLM client for annotation
    if validator == "instructor":
        try:
            result = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Extract entities as structured JSON."},
                    {"role": "user", "content": f"{prompt}\nThe text to annotate:\n{text}"}
                ],
                response_model=response_model,
                max_retries=max_retries,
            )
        except InstructorRetryException as e:
            logger.warning(
                f"    ⚠️ Validated annotation failed after {e.n_attempts} attempts."
            )
            # logger.warning(f"Total usage: {e.total_usage.total_tokens} tokens")
            return str(e.last_completion)

        except InstructorValidationError as e:
            # logger.error(e.errors)
            return str(e.raw_output)

    elif validator == "llamaindex":
        input_msg = ChatMessage.from_str(f"{prompt}\nThe text to annotate:\n{text}")
        try:
            response = client.chat([input_msg])
            result = response.raw
        except (PydanticValidationError, CoreValidationError, ValueError) as e:
            return str(e)

    elif validator == "pydanticai":
        agent = Agent(
            model=client,
            output_type=ListOfEntities,
            retries=max_retries,
            system_prompt=("Extract entities as structured JSON."),
        )
        try:
            response = agent.run_sync(f"{prompt}\nThe text to annotate:\n{text}")
            result = response.output
        except (
            PydanticValidationError,
            CoreValidationError,
            UnexpectedModelBehavior,
        ) as e:
            return str(e)

    return result


def append_annotation_result(
    df: pd.DataFrame,
    *,
    model_name: str,
    provider: str,
    validator: str,
    prompt_tag: str,
    text_to_annotate: str,
    json_path: Path,
    model_response: dict[str, Any],
    groundtruth: Any,
) -> pd.DataFrame:
    """
    Evaluate a model response and append a result row to the evaluation DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Existing dataframe where the new row will be added.
    model_name : str
        Name of the LLM used.
    provider : str
        Backend provider (OpenAI or OpenRouter).
    validator : str
        Name of the validation method ('no_validation', 'instructor', 'llamaindex', 'pydanticai').
    prompt_tag : str
        Tag defining expected JSON format ('json' or 'json_with_positions').
    text_to_annotate : str
        Original text provided to the model.
    json_path : Path
        Path to the groundtruth annotation file.
    model_response : dict
        The model prediction processed by `annotate()`.
    groundtruth : Any
        Expert annotations used to check correctness.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with the appended row.
    """
    # Evaluation of the model's response
    is_correct_output_format = is_valid_output_format(model_response, prompt_tag)
    is_without_hallucination = has_no_hallucination(model_response, text_to_annotate)
    is_correct = is_annotation_correct(model_response, groundtruth)

    # Append the row
    new_row = {
        "model_name": model_name,
        "provider": provider,
        "validator": validator,
        "prompt": prompt_tag,
        "text_to_annotate": text_to_annotate,
        "json_path": str(json_path),
        "is_correct_output_format": is_correct_output_format,
        "is_without_hallucination": is_without_hallucination,
        "is_correct": is_correct,
    }

    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


def summarize_model_stats(parquet_path: Path) -> dict[str, dict[str, float]]:
    """
    Load a parquet evaluation file for a model and aggregate statistics per validator.

    Returns a structure:
    {
        "no_validation": {
            "correct_format": 80.0,
            "no_hallucination": 70.0,
            "correct_answer": 60.0,
        },
        ...
    }
    """
    df = pd.read_parquet(parquet_path)

    results = {}

    for validator in df["validator"].unique():
        sub = df[df["validator"] == validator]

        total = len(sub)
        if total == 0:
            results[validator] = {
                "correct_format": 0.0,
                "no_hallucination": 0.0,
                "correct_answer": 0.0,
            }
            continue

        results[validator] = {
            "correct_format": round(100 * sub["is_correct_output_format"].mean(), 1),
            "no_hallucination": round(100 * sub["is_without_hallucination"].mean(), 1),
            "correct_answer": round(100 * sub["is_correct"].mean(), 1),
        }

    return results


@click.command()
@click.option(
    "--annotations-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("annotations/v2"),
    show_default=True,
    help="Directory containing the JSON annotation files to evaluate."
)
@click.option(
    "--nb-annotations",
    type=int,
    default=10,
    show_default=True,
    help="Number of annotation samples to evaluate."
)
@click.option(
    "--tag-prompt",
    type=click.Choice(["json", "json_with_positions"], case_sensitive=False),
    default="json",
    show_default=True,
    help=(
        "Tag determining the expected format of the LLM output. "
        "'json' requests only labels and text, while 'json_with_positions' "
        "also includes start and end character positions."
    )
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path(f"results/json_evaluation_stats/{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    show_default=True,
    help="Target directory where evaluation results will be saved.",
    callback=ensure_dir
)
def evaluate_json_annotations(
    annotations_dir: Path,
    nb_annotations: int,
    tag_prompt: str,
    results_dir: Path,
) -> None:
    """
    Evaluate the quality of JSON entity annotations produced by multiple LLM models.

    This command loads JSON annotation files, sends the content to several
    LLM-based processors (Instructor, LlamaIndex, and PydanticAI clients), and
    compares the predicted annotations against the ground truth. It then
    aggregates the evaluation metrics and writes the results to the specified
    output directory.

    Parameters
    ----------
    annotations_dir : Path
        Directory containing the JSON annotation files to evaluate.
    nb_annotations : int
        Number of annotation samples to process from the dataset.
    tag_prompt : str
        Descriptor indicating the format of the expected LLM output
        (e.g., 'json' or 'json_with_positions').
    results_dir : Path
        Directory where evaluation results, logs, and reports will be written.
    """
    # Configure logging
    setup_logger(logger, results_dir)
    logger.info("Starting evaluation of JSON annotation outputs...")
    logger.debug(f"Annotations directory: {annotations_dir}")
    logger.debug(f"Number of annotations to process: {nb_annotations}")
    logger.debug(f"Tag prompt: {tag_prompt}")
    logger.debug(f"Results directory: {results_dir}")

    # Initialize
    all_summary_rows = []

    # Assign each models to instructor/llamaindex/pydanticai clients
    instructor_clients = assign_all_instructor_clients()
    llama_clients = assign_all_llamaindex_clients()
    py_clients = assign_all_pydanticai_clients()

    # Retrieve MD annotated texts with their verified annotations
    annotations = load_recent_annotations(
        annotations_dir,
        nb_annotations,
        tag_prompt,
    )

    # Loop through LLMs
    for model_name in MODELS_OPENAI + MODELS_OPENROUTER:
        logger.info(
            f"=================== 🤖 Evaluating model: {model_name} ==================="
        )
        provider = "OpenAI" if model_name in MODELS_OPENAI else "OpenRouter"

        # Create an empty dataframe for this model
        eval_df = pd.DataFrame(
            columns=[
                "model_name",
                "provider",
                "validator",
                "prompt",
                "text_to_annotate",
                "json_path",
                "is_correct_output_format",
                "is_without_hallucination",
                "is_correct",
            ]
        )

        # Loop through MD texts to annotate
        for record in annotations:
            file_path = record["file_path"]
            text_to_annotate = record["text_to_annotate"]
            groundtruth = record["groundtruth"]

            # ------------------------------------------------------
            # 1. Annotation without validation
            # ------------------------------------------------------
            response_no_val = annotate(
                                text_to_annotate,
                                model_name,
                                instructor_clients[model_name],
                                validation=False)

            eval_df = append_annotation_result(
                eval_df,
                model_name=model_name,
                provider=provider,
                validator="no_validation",
                prompt_tag=tag_prompt,
                text_to_annotate=text_to_annotate,
                json_path=file_path,
                model_response=response_no_val,
                groundtruth=groundtruth,
            )

            # ------------------------------------------------------
            # 2. Annotation with INSTRUCTOR validation
            # ------------------------------------------------------
            response_instructor_val = annotate(
                                text_to_annotate,
                                model_name,
                                instructor_clients[model_name],
                                validation=True)

            eval_df = append_annotation_result(
                eval_df,
                model_name=model_name,
                provider=provider,
                validator="instructor",
                prompt_tag=tag_prompt,
                text_to_annotate=text_to_annotate,
                json_path=file_path,
                model_response=response_instructor_val,
                groundtruth=groundtruth,
            )

            # ------------------------------------------------------
            # 3. Annotation with LLAMAINDEX validation
            # ------------------------------------------------------
            response_llamaindex_val = annotate(
                                text_to_annotate,
                                model_name,
                                llama_clients[model_name],
                                validation=True)

            eval_df = append_annotation_result(
                eval_df,
                model_name=model_name,
                provider=provider,
                validator="llamaindex",
                prompt_tag=tag_prompt,
                text_to_annotate=text_to_annotate,
                json_path=file_path,
                model_response=response_llamaindex_val,
                groundtruth=groundtruth,
            )
            # ------------------------------------------------------
            # 4. Annotation with PyDANTICAI validation
            # ------------------------------------------------------
            response_pydanticai_val = annotate(
                                text_to_annotate,
                                model_name,
                                py_clients[model_name],
                                validation=True)

            eval_df = append_annotation_result(
                eval_df,
                model_name=model_name,
                provider=provider,
                validator="pydanticai",
                prompt_tag=tag_prompt,
                text_to_annotate=text_to_annotate,
                json_path=file_path,
                model_response=response_pydanticai_val,
                groundtruth=groundtruth,
            )

        # Save model's evaluation
        model_out_path = results_dir / f"{model_name}_{len(annotations)}_annotations_stats.parquet"
        eval_df.to_parquet(model_out_path, index=False)

        parquet_path = results_dir / f"{model_name}_{len(annotations)}_annotations_stats.parquet"
        model_key = f"{model_name} ({provider})"
        stats = summarize_model_stats(parquet_path)

        # Add row with 12 values (4 validator with 3 things to check : format, hallucination, correctness)
        all_summary_rows.append([
            model_key,

            stats["no_validation"]["correct_format"],
            stats["no_validation"]["no_hallucination"],
            stats["no_validation"]["correct_answer"],

            stats["instructor"]["correct_format"],
            stats["instructor"]["no_hallucination"],
            stats["instructor"]["correct_answer"],

            stats["llamaindex"]["correct_format"],
            stats["llamaindex"]["no_hallucination"],
            stats["llamaindex"]["correct_answer"],

            stats["pydanticai"]["correct_format"],
            stats["pydanticai"]["no_hallucination"],
            stats["pydanticai"]["correct_answer"],
        ])

    # Create simple dataframe
    df_simple = pd.DataFrame(
        all_summary_rows,
        columns=[
            "Model (Provider)",

            "nof_val_correct_format",
            "nof_val_no_hallu",
            "nof_val_correct_answer",

            "instr_correct_format",
            "instr_no_hallu",
            "instr_correct_answer",

            "llama_correct_format",
            "llama_no_hallu",
            "llama_correct_answer",

            "pyd_correct_format",
            "pyd_no_hallu",
            "pyd_correct_answer",
        ]
    )

    multi_columns = pd.MultiIndex.from_tuples([
        ("JSON without format validation", "Correct output format (%)"),
        ("JSON without format validation", "No hallucination (%)"),
        ("JSON without format validation", "Correct answer (%)"),
        ("JSON + Instructor", "Correct output format (%)"),
        ("JSON + Instructor", "No hallucination (%)"),
        ("JSON + Instructor", "Correct answer (%)"),
        ("JSON + LlamaIndex", "Correct output format (%)"),
        ("JSON + LlamaIndex", "No hallucination (%)"),
        ("JSON + LlamaIndex", "Correct answer (%)"),
        ("JSON + PydanticAI", "Correct output format (%)"),
        ("JSON + PydanticAI", "No hallucination (%)"),
        ("JSON + PydanticAI", "Correct answer (%)")
    ])
    df_results = pd.DataFrame(
        df_simple.drop(columns=["Model (Provider)"]).values,
        columns=multi_columns,
        index=df_simple["Model (Provider)"],
    )
    path = results_dir / f"evaluation_summary_{len(annotations)}_annotations_{results_dir.split("/")[-1]}.xlsx"
    df_results.to_excel(path, index=True)
    logger.success(f"Evaluation stats saved to: {path} successfully!")


# MAIN PROGRAM
if __name__ == "__main__":
    # Evaluate json annotations trough all models
    evaluate_json_annotations()
