"""
Extract structured entities from a text using a specified LLM and framework.

This script applies a language model to a single text file and extracts
structured entities based on a provided prompt. It supports multiple
frameworks to guide or validate the extraction.

The output consists of:
1. A JSON file containing metadata about the extraction and the serialized
   model response.
2. A plain text file containing the raw response from the model.

Usage:
=======
uv run src/extract_entities.py --path-prompt PATH --model STR --path-text PATH
                               [--tag-prompt STR] [--framework STR]
                               [--output-dir PATH] [--max-retries INT]

Arguments:
==========
    --path-prompt: Path
        Path to a text file containing the extraction prompt.

    --model: str
        Language model name to use for extraction find in OpenRouter page model
        (https://openrouter.ai/models). Example: "openai/gpt-4o-mini".

    --path-text: Path
        Path to a JSON file containing the text to annotate.
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
uv run src/extract_entities.py \
    --path-prompt prompts/json_few_shot.txt \
    --model openai/gpt-4o \
    --path-text annotations/v2/figshare_121241.json \
    --tag-prompt json \
    --framework instructor \
    --output-dir results/llm_annotations \
    --max-retries 3

This command will extract entities from `annotations/v2/figshare_121241.json`
using the prompt in `prompts/json_few_shot.txt` and the "instructor"
validation framework, saving results in `results/llm_annotations` with base
filename `figshare_121241_openai_gpt-4o_instructor_YYYYMMDD_HHMMSS`. Two files
will be generated: a JSON metadata file (`.json`) and a text file with the raw
model response (`.txt`). The command will retry up to 3 times in case of API
errors.
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
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

import click
import instructor
from dotenv import load_dotenv
from instructor.core import InstructorRetryException
from instructor.core.exceptions import ModeError, ProviderError
from instructor.core.exceptions import ValidationError as InstructorValidationError
from llama_index.llms.openai import OpenAI as llamaOpenAI
from llama_index.llms.openrouter import OpenRouter
from loguru import logger
from openai.types.chat import ChatCompletion
from pydantic import ValidationError as PydanticValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_core import ValidationError as CoreValidationError

# UTILITY IMPORTS
from models.pydantic_output_models import ListOfEntities, ListOfEntitiesPositions


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

    # If it's a ListOfEntities or ListOfEntitiesPositions object
    if isinstance(resp, (ListOfEntities, ListOfEntitiesPositions)):
        return resp.model_dump_json(indent=2)

    # Specific handling for ChatCompletion-like objects
    if isinstance(resp, ChatCompletion):
        return json.dumps(resp.__dict__, default=str)

    return str(resp)


def sanitize_filename(s: str) -> str:
    """Replace unsafe characters for filenames.

    This function replaces any character that is not a letter, digit,
    underscore, hyphen, or dot with an underscore. It helps prevent issues
    with filesystem restrictions across different operating systems.

    Parameters
    ----------
    s : str
        The input string to sanitize.

    Returns
    -------
    str
        A sanitized string safe for use as a filename.
    """
    return re.sub(r"[^\w\-_.]", "_", s)


def annotate_with_instructor(
    text: str,
    model: str,
    api_key: str | None,
    prompt: str,
    response_model: ListOfEntities | ListOfEntitiesPositions | None,
    max_retries: int = 3,
) -> tuple[ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions,
            float | int]:
    """
    Annotate a text using the Instructor framework.

    This function queries an LLM via Instructor to extract structured entities
    from the input text. When validation is enabled, the output is validated
    against a Pydantic schema and returned as a structured object.

    Parameters
    ----------
    text : str
        Input text to annotate.
    model : str
        Identifier of the LLM model to use.
    api_key : str | None
        API key used to authenticate requests to OpenRouter
        (typically provided via the ``OPENROUTER_API_KEY`` environment variable).
    prompt : str
        Instruction prompt provided to the model.
    response_model : ListOfEntities | ListOfEntitiesPositions | None
        Pydantic model used to validate and parse the LLM output.
        If ``None``, no validation is applied and the raw output is returned.
    max_retries : int (Default is 3)
        Maximum number of retries in case of API or validation failure.

    Returns
    -------
    ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions
        Annotation result returned by the LLM. The returned type depends on
        whether validation is enabled and on the chosen response model.
    float | int:
        The time elapsed for the inference.

    Raises
    ------
    ValueError
        If the OpenRouter API key is missing (``OPENROUTER_API_KEY`` is not set),
        preventing initialization of the Instructor client.
    """
    if api_key is None:
        msg = "OPENROUTER_API_KEY is not set. Unable to initialize Instructor client."
        logger.error(msg)
        raise ValueError(msg)

    # Instantiate an Instructor client for the requested model.
    model_entry_point = (
        f"openrouter/{model}" if not model.startswith("openai") else model)
    client = instructor.from_provider(model_entry_point, async_client=False,
                                                mode=instructor.Mode.JSON,
                                                base_url="https://openrouter.ai/api/v1",
                                                api_key=api_key)

    try:
        # Query the LLM
        start_time = time.time()
        llm_response = client.create(
            messages=[
                {
                    "role": "system",
                    "content": "Extract entities as structured JSON.",
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n{text}",
                },
            ],
            # The response is optionally validated against the provided Pydantic model.
            response_model=response_model,
            max_retries=max_retries,
        )
        elapsed_time: int | float = time.time() - start_time
        return llm_response, elapsed_time

    except InstructorValidationError as exc:
        # Raised when the LLM output does not conform to the expected schema.
        logger.warning(f"Validation failed: {exc}")
        return str(exc), 0

    except InstructorRetryException as exc:
        # Raised when all retry attempts fail.
        logger.warning(f"Failed after {exc.n_attempts} attempts")
        logger.warning(f"Last completion: {exc.last_completion}")
        return str(exc.last_completion), 0

    except (ProviderError, ModeError) as exc:
        # Catch-all for provider-level, mode-related, or parsing errors.
        logger.warning(f"Instructor error: {exc}")
        return str(exc), 0


def annotate_with_llamaindex(
    text: str,
    model: str,
    api_key: str | None,
    prompt: str,
    response_model: ListOfEntities | ListOfEntitiesPositions | None,
) -> tuple[ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions,
            float | int]:
    """
    Annotate a text using the Llamaindex framework.

    This function queries an LLM via Llamaindex to extract structured entities
    from the input text. When validation is enabled, the output is validated
    against a Pydantic schema and returned as a structured object.

    Parameters
    ----------
    text : str
        Input text to annotate.
    model : str
        Identifier of the LLM model to use.
    api_key : str | None
        API key used to authenticate requests to OpenRouter
        (typically provided via the ``OPENROUTER_API_KEY`` environment variable).
    prompt : str
        Instruction prompt provided to the model.
    response_model : ListOfEntities | ListOfEntitiesPositions | None
        Pydantic model used to validate and parse the LLM output.
        If ``None``, no validation is applied and the raw output is returned.

    Returns
    -------
    ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions
        Annotation result returned by the LLM. The returned type depends on
        whether validation is enabled and on the chosen response model.
    float | int:
        The time elapsed for the inference.

    Raises
    ------
    ValueError
        If the environment variable ``OPENROUTER_API_KEY`` is not set or
        if a ValueError occurs during the LLM call.
    """
    if api_key is None:
        msg = "OPENROUTER_API_KEY must be set in the environment"
        logger.error(msg)
        raise ValueError(msg)

    # Instantiate an Llamaindex client for the requested model
    llm = OpenRouter(model=model, api_key=api_key)
    client = llm.as_structured_llm(output_cls=response_model)

    try:
        # Query the LLM
        start_time = time.time()
        if model.startswith("openai"):
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key is None:
                msg = "OPENAI_API_KEY must be set in the environment"
                raise ValueError(msg)
            model_name = model.split("/")[1]
            client = llamaOpenAI(model=model_name, api_key=openai_api_key)
            client = client.as_structured_llm(output_cls=response_model)

        llm_response = client.complete(f"{prompt}\n{text}").raw
        elapsed_time: int | float = time.time() - start_time
        return llm_response, elapsed_time

    except ValueError as e:
        logger.error(f"ValueError during LLM call: {e}")
        return str(e), 0

    except PydanticValidationError as e:
        logger.error(f"Pydantic validation error: {e}")
        return str(e), 0


def annotate_with_pydanticai(
    text: str,
    model: str,
    api_key: str | None,
    prompt: str,
    response_model: ListOfEntities | ListOfEntitiesPositions | None,
    max_retries: int = 3,
) -> tuple[ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions,
            float | int]:
    """
    Annotate a text using the PydanticAI framework.

    This function queries an LLM via PydanticAI to extract structured entities
    from the input text. When validation is enabled, the output is validated
    against a Pydantic schema and returned as a structured object.

    Parameters
    ----------
    text : str
        Input text to annotate.
    model : str
        Identifier of the LLM model to use.
    api_key : str | None
        API key used to authenticate requests to OpenRouter
        (typically provided via the ``OPENROUTER_API_KEY`` environment variable).
    prompt : str
        Instruction prompt provided to the model.
    response_model : ListOfEntities | ListOfEntitiesPositions | None
        Pydantic model used to validate and parse the LLM output.
        If ``None``, no validation is applied and the raw output is returned.
    max_retries : int (Default is 3)
        Maximum number of retries in case of API or validation failure.

    Returns
    -------
    ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions
        Annotation result returned by the LLM. The returned type depends on
        whether validation is enabled and on the chosen response model.
    float | int:
        The time elapsed for the inference.

    Raises
    ------
    ValueError
        If the environment variable ``OPENROUTER_API_KEY`` is not set or
        if a ValueError occurs during the LLM call.
    """
    if api_key is None:
        msg = "OPENROUTER_API_KEY must be set in the environment"
        logger.error(msg)
        raise ValueError(msg)

    # Instantiate an PydanticAI client for the requested model
    client = OpenAIChatModel(model, provider=OpenRouterProvider(api_key=api_key))

    try:
        # Query the LLM
        start_time = time.time()
        agent = Agent(
            model=client,
            output_type=response_model,
            retries=max_retries,
            system_prompt=("Extract entities as structured JSON."),
        )
        llm_response = agent.run_sync(f"{prompt}\n{text}").output
        elapsed_time: int | float = time.time() - start_time
        return llm_response, elapsed_time

    except PydanticValidationError as e:
        logger.error(f"Pydantic validation error: {e}")
        return str(e), 0

    except CoreValidationError as e:
        logger.error(f"Core validation error: {e}")
        return str(e), 0

    except UnexpectedModelBehavior as e:
        logger.error(f"Unexpected model behavior: {e}")
        return str(e), 0


def extract_entities(
    tag_prompt: str,
    path_prompt: Path,
    model: str,
    path_text: Path,
    framework: str,
    output_dir: Path,
    max_retries: int = 3
) -> None:
    """
    Extract structured entities from a text using a specified LLM and framework.

    Parameters
    ----------
    path_prompt : Path
        Path to the prompt file.
    model : str
        Model name to use for extraction.
    path_text : str
        Path to the JSON text to process.
    framework : str
        Validation framework.
    output_dir : str
        Directory to save output files.
    max_retries : int (Default is 3)
        Maximum number of retries in case of API or validation failure.

    Raises
    ------
    FileNotFoundError
        If either the text file or prompt file does not exist.
    json.JSONDecodeError
        If the JSON text file cannot be parsed.
    KeyError
        If the key "raw_text" is missing from the JSON text file.
    """
    setup_logger(logger, log_dir=output_dir)
    logger.info("Starting the extraction of entities...")
    logger.debug(f"===================================================== "
                 f"📝 Text to annotate path: {path_text} "
                 f"=====================================================")
    logger.debug(
        f"🤖 Model: {model} | 🛠️ Framework: {framework} | 🏷️ Tag: {tag_prompt} | "
        f"💬 Prompt path: {path_prompt} | 📂 Output dir: {output_dir} | "
        f"🔁 Max retries: {max_retries}\n"
    )

    # Load text to annotate
    try:
        with open(path_text, encoding="utf-8") as f:
            data = json.load(f)
            text_to_annotate = data["raw_text"]
            # Retrieve the groundtruth annotation
            entities = data["entities"]
            if tag_prompt == "json":
                # We remove the "start" and "end" keys
                normalized = [
                    {"label": ent.get("label"), "text": ent.get("text")}
                    for ent in entities
                ]
                groundtruth = ListOfEntities(entities=normalized)
            else:
                groundtruth = ListOfEntitiesPositions(entities=entities)
    except FileNotFoundError:
        logger.error("File not found: %s", path_text)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in file %s: %s", path_text, e)
        raise
    except KeyError:
        logger.error("Expected key 'raw_text' not found in %s", path_text)
        raise
    preview = text_to_annotate[:75].replace("\n", " ")
    logger.debug(f"Loaded text ({len(text_to_annotate)} chars): {preview}...")

    # Load prompt from txt file
    try:
        prompt = path_prompt.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error(f"File not found: {path_prompt}")
        raise
    preview = prompt[:75].replace("\n", " ")
    logger.debug(f"Loaded prompt ({len(prompt)} chars): {preview}...\n")

    # Set response model and retries based on tag and framework
    if framework:
        if tag_prompt == "json":
            response_model = ListOfEntities
        else:
            response_model = ListOfEntitiesPositions
    else:
        response_model = None
        max_retries = 0

    # Retrive the openrouter api key
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    # Run annotation and time it
    if framework == "instructor" or framework is None:
        llm_response, inference_time = annotate_with_instructor(
            text_to_annotate,
            model,
            api_key,
            prompt,
            response_model,
            max_retries
        )

    elif framework == "llamaindex":
        llm_response, inference_time = annotate_with_llamaindex(
            text_to_annotate,
            model,
            api_key,
            prompt,
            response_model,
        )

    elif framework == "pydanticai":
        llm_response, inference_time = annotate_with_pydanticai(
            text_to_annotate,
            model,
            api_key,
            prompt,
            response_model,
            max_retries
        )

    # Prepare output paths
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    model_safe = sanitize_filename(model)
    base_name = f"{path_text.stem}_{model_safe}_{framework}_{timestamp}"
    json_output_path = output_dir / f"{base_name}.json"
    txt_output_path = output_dir / f"{base_name}.txt"

    # Save JSON metadata + response
    json_data = {
        "timestamp": timestamp,
        "output_file": str(txt_output_path),
        "text_file": str(path_text),
        "framework_name": framework,
        "model_name": model,
        "prompt_path": str(path_prompt),
        "tag_prompt": str(tag_prompt),
        "inference_time_sec": inference_time,
        "raw_llm_response": serialize_response(llm_response),
        "groundtruth": serialize_response(groundtruth)
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(json_data, indent=4, ensure_ascii=False),
                                 encoding="utf-8")
    logger.debug(f"Saved JSON output to {json_output_path}")

    # Save raw model response
    txt_output_path.write_text(
        serialize_response(llm_response),
        encoding="utf-8",
    )
    logger.debug(f"Saved raw response to {txt_output_path}")

    logger.success(f"Completed the extraction of entities in {inference_time:.2f} "
                   "seconds successfully!")


@click.command()
@click.option(
    "--tag-prompt",
    default="json",
    type=click.Choice(["json", "json_with_positions"]),
    help="Descriptor indicating the format of the expected LLM output "
    "(e.g., 'json' or 'json_with_positions')."
)
@click.option(
    "--path-prompt",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the prompt file."
)
@click.option(
    "--model",
    required=True,
    type=str,
    help="Model name to use for extraction."
)
@click.option(
    "--path-text",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the JSON text to process."
)
@click.option(
    "--framework",
    default=None,
    type=click.Choice(["instructor", "llamaindex", "pydanticai"]),
    help="Validation framework."
)
@click.option(
    "--output-dir",
    default="results/llm_annotations",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path),
    help="Directory to save output files.",
    callback=ensure_dir
)
@click.option(
    "--max-retries",
    default=3,
    type=int,
    help="Maximum number of retries in case of API or validation failure."
)
def extract_entities_from_cli(
    tag_prompt: str,
    path_prompt: Path,
    model: str,
    path_text: Path,
    framework: str,
    output_dir: Path,
    max_retries: int = 3
) -> None:
    """CLI entrypoint."""
    extract_entities(
        tag_prompt=tag_prompt,
        path_prompt=path_prompt,
        model=model,
        path_text=path_text,
        framework=framework,
        output_dir=output_dir,
        max_retries=max_retries,
    )


# MAIN PROGRAM
if __name__ == "__main__":
    extract_entities_from_cli()
