"""
Extract structured entities from a text using a specified LLM and framework.

This script applies a language model to a single text and extracts structured
entities based on a provided prompt. It supports multiple frameworks to guide
or validate the extraction.

The output is a JSON object containing the extracted entities, which may
include labels, text, and optional character positions depending on the prompt.

Usage:
=======
    uv run src/extract_entities.py --tag-prompt str
                                   --prompt PATH
                                   --model MODEL_NAME
                                   --text PATH
                                   --framework {none,instructor,llamaindex,pydanticai}

Arguments:
==========
    --tag-prompt: str
        Descriptor indicating the format of the expected LLM output \
        (e.g., 'json' or 'json_with_positions').

    --prompt: PATH
        Path to a text file containing the extraction prompt.

    --model: STR
        Language model name to use for extraction from OpenRouter (https://openrouter.ai/models).

    --text: PATH
        Path to a JSON file with the source text.

    --framework: STR
        Validation framework to apply to model outputs. Default: "none".
        Choices: "none", "instructor", "llamaindex", "pydanticai"

Example:
========
    uv run src/extract_entities.py \
        --tag-prompt json \
        --path-prompt prompts/json_few_shot.txt \
        --model  openai/gpt-4o \
        --text annotations/v2/figshare_121241.json \
        --framework instructor

This command will use GPT-4o to extract entities
from `annotations/v2/figshare_121241.json` according to the instructions in
`prompts/json_few_shot.txt`, applying the "instructor" validation framework.
"""

# METADATAS
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "AGPL-3.0 license"
__date__ = "2025"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import instructor
from dotenv import load_dotenv
from llama_index.llms.openrouter import OpenRouter
from loguru import logger
from openai.types.chat import ChatCompletion
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

# UTILITY IMPORTS
from models.pydantic_output_models import ListOfEntities, ListOfEntitiesPositions
from utils import annotate, sanitize_filename


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


def serialize_response(resp: Any) -> str:
    """
    Serialize various response objects into a JSON-safe string representation.

    Parameters
    ----------
    resp : Any
        The object to serialize. This may be a string, a custom class instance,
        or a model response object such as ChatCompletion.

    Returns
    -------
    str
        A JSON-compatible string representation of the input object.
    """
    # If it's already a string, nothing to do.
    if isinstance(resp, str):
        return resp

    if isinstance(resp, (ListOfEntities, ListOfEntitiesPositions)):
        return resp.model_dump_json(indent=2)

    # Specific handling for ChatCompletion-like objects
    if isinstance(resp, ChatCompletion):
        return json.dumps(resp.__dict__, default=str)

    return str(resp)


@click.command()
@click.option("--tag-prompt", required=True, type=click.Choice(["json", "json_with_positions"]),
              help="Descriptor indicating the format of the expected LLM output \
                (e.g., 'json' or 'json_with_positions').")
@click.option("--path-prompt", required=True, type=click.Path(exists=True),
              help="Path to the prompt file.")
@click.option("--model", required=True, type=str,
              help="Model name to use for extraction.")
@click.option("--path-text", required=True, type=click.Path(exists=True),
              help="Path to the JSON text to process.")
@click.option("--framework", default=None,
              type=click.Choice(["instructor", "llamaindex", "pydanticai"]),
              help="Validation framework.")
@click.option("--output-dir", default="results/llm_annotations", type=click.Path(),
              help="Directory to save output files.")
def extract_entities(
    tag_prompt: str,
    path_prompt: Path,
    model: str,
    path_text: Path,
    framework: str,
    output_dir: Path
) -> None:
    """
    Extract structured entities from a text using a specified LLM and framework.

    Parameters
    ----------
    path_prompt : Path
        Path to the prompt file.
    model : str
        Model name to use for extraction.
    text : str
        Path to the JSON text to process.
    framework : str
        Validation framework.
    output_dir : str
        Directory to save output files.
    """
    setup_logger(logger, log_dir="logs")
    logger.info("Starting the extraction of entities...")
    logger.debug(tag_prompt)
    logger.debug(path_prompt)
    logger.debug(model)
    logger.debug(path_text)
    logger.debug(framework)
    logger.debug(output_dir)

    # Load text to annotate
    text_path = Path(path_text)
    with open(text_path, encoding="utf-8") as f:
        data = json.load(f)
        text_to_annotate = data["raw_text"]

    # Retrive the openrouter api key
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None:
        msg = "OPENROUTER_API_KEY must be set in the environment"
        raise ValueError(msg)

    # Assign clients based on framework
    validation = True
    if framework == "instructor" or framework is None:
        client = instructor.from_provider(model, mode=instructor.Mode.JSON)
    elif framework == "llamaindex":
        llm = OpenRouter(model=model, api_key=api_key)
        output_model = ListOfEntities if tag_prompt == "json" else ListOfEntitiesPositions
        client = llm.as_structured_llm(output_cls=output_model)
    elif framework == "pydanticai":
        client = OpenAIChatModel(model, provider=OpenRouterProvider(api_key=api_key))
    else:
        validation = False
        msg = f"Unknown framework '{framework}'." \
            "Valid options: 'instructor', 'llamaindex', 'pydanticai'"
        raise ValueError(msg)

    # Run annotation and time it
    start_time = time.time()
    path_prompt = Path(path_prompt)
    response = annotate(text_to_annotate, model, client, tag_prompt, framework, path_prompt=path_prompt, validation=validation)
    elapsed_time = time.time() - start_time

    # Prepare output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = sanitize_filename(model)
    base_name = f"{text_path.stem}_{model_safe}_{framework}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output_path = output_dir / f"{base_name}.json"
    txt_output_path = output_dir / f"{base_name}.txt"

    # Save JSON metadata + response
    json_data = {
        "model": model,
        "framework": framework,
        "text_file": str(text_path),
        "output_file": str(txt_output_path),
        "elapsed_time_sec": elapsed_time,
        "timestamp": timestamp,
        "response": serialize_response(response)
    }
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    # Save raw model response
    txt_output_path.write_text(
        serialize_response(response),
        encoding="utf-8",
    )

    logger.info(f"Saved JSON output to {json_output_path}")
    logger.info(f"Saved raw response to {txt_output_path}")
    logger.info(f"Annotation completed in {elapsed_time:.2f} seconds.")


# MAIN PROGRAM
if __name__ == "__main__":
    extract_entities()