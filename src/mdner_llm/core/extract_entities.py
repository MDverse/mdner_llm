"""
Extract structured entities from a text using a specified LLM and framework.

This script applies a language model to a single text file and extracts
structured entities based on a provided prompt. It supports multiple
frameworks to guide or validate the extraction.

The output consists of:
1. A JSON file containing metadata about the extraction and the serialized
   model response.
2. A plain text file containing the raw response from the model.
"""

import json
import time
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import click
import instructor
import loguru
from instructor.core import InstructorRetryException
from instructor.core.exceptions import ModeError, ProviderError
from instructor.core.exceptions import ValidationError as InstructorValidationError
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from pydantic import ValidationError as PydanticValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_core import ValidationError as CoreValidationError

from mdner_llm.models.entities import ListOfEntities
from mdner_llm.models.entities_with_positions import ListOfEntitiesPositions
from mdner_llm.utils.common import (
    ensure_dir,
    load_api_key,
    sanitize_filename,
    serialize_response,
)
from mdner_llm.utils.logger import create_logger


def load_text_and_metadata(
    path_text: str | Path, prompt_tag: str, logger: "loguru.Logger" = loguru.logger
) -> tuple[str, ListOfEntities | ListOfEntitiesPositions, str]:
    """Load raw text and ground truth annotations from a JSON file.

    The JSON file must contain a ``raw_text``, ``entities`` field and a ``url`` field.
    Depending on the value of ``prompt_tag``, entities are normalized or kept
    with positional information.

    Parameters
    ----------
    path_text
        Path to the JSON file containing the text and annotations.
    prompt_tag
        Annotation format selector. If equal to `json`, entity positions
        are removed.
    logger
        Logger instance for logging messages.


    Returns
    -------
    tuple[str, str, ListOfEntities | ListOfEntitiesPositions, str]
        The raw text to annotate and the corresponding ground truth object parsed from
        the JSON file, and the URL if available.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    json.JSONDecodeError
        If the file content is not valid JSON.
    KeyError
        If required keys are missing from the JSON structure.
    """
    # Load the JSON file containing the text and annotations
    try:
        with open(path_text, encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        logger.error(f"File not found: {path_text}")
        raise
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid JSON in file {path_text}: {exc}")
        raise
    except KeyError as exc:
        logger.error(f"Missing expected key {exc} in {path_text}")
        raise
    # Extract raw text and entities from the loaded JSON
    text_to_annotate = data["raw_text"]
    entities = data["entities"]
    url = data.get("url", "N/A")
    # Normalize entities based on the prompt_tag value
    if prompt_tag == "json":
        # Remove positional information from entities
        normalized = [
            {"category": ent.get("category"), "text": ent.get("text")}
            for ent in entities
        ]
        # Create a ListOfEntities object for the ground truth
        groundtruth = ListOfEntities(entities=normalized)
    else:
        # Create a ListOfEntitiesPositions object for the ground truth
        groundtruth = ListOfEntitiesPositions(entities=entities)
    preview = text_to_annotate[:75].replace("\n", " ")
    logger.debug(f"Loaded text ({len(text_to_annotate)} chars): {preview}...")
    return text_to_annotate, groundtruth, url


def load_prompt(prompt_file: Path, logger: "loguru.Logger" = loguru.logger) -> str:
    """Load the JSON few-shot prompt from the mdner_llm package.

    Parameters
    ----------
    prompt_file : Path
        Path to the text file containing the prompt.
    logger : loguru.Logger
        Logger instance for logging messages.

    Returns
    -------
    str
        The prompt content.

    Raises
    ------
    FileNotFoundError
        If the prompt file does not exist in the package resources.
    """
    # Ensure the prompt file path is a Path object
    prompt_file = Path(prompt_file)
    # Load prompt content
    # from the specified file within the package resources
    try:
        prompt = (
            files("mdner_llm.prompt_templates")
            .joinpath(prompt_file)
            .read_text(encoding="utf-8")
        )
    # Handle when the prompt file is not found in the package resources
    except FileNotFoundError:
        logger.error("Prompt file not found in mdner_llm.prompt_templates")
        raise

    preview = prompt[:75].replace("\n", " ")
    logger.debug(f"Loaded prompt ({len(prompt)} chars): {preview}...")
    return prompt


def annotate_without_framework(
    text: str,
    model: str,
    api_key: str | None,
    prompt: str,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[ChatCompletion | str, float | int, dict[str, float | int]]:
    """
    Annotate a text without applying any validation framework.

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
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger

    Returns
    -------
    ChatCompletion | str
        Raw annotation result returned by the LLM.
    float | int:
        The time elapsed for the inference.
    dict[str, float | int]:
        A dictionary containing cost usage and token counts.
    """
    # Instantiate an OpenAI client for the requested model
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    no_response = (
        None,
        0,
        {"cost": 0, "input_tokens": 0, "output_tokens": 0},
    )
    try:
        # Query the LLM and time the inference
        start_time = time.perf_counter()
        llm_response = client.chat.completions.create(
            model=model,
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
        )
        elapsed_time = time.perf_counter() - start_time

    # Handle common OpenAI API exceptions
    except RateLimitError as exc:
        logger.warning(f"Rate limit exceeded: {exc}")
        return no_response
    except APIConnectionError as exc:
        logger.warning(f"Connection error: {exc}")
        return no_response
    except APIError as exc:
        logger.warning(f"API error: {exc}")
        return no_response
    # If no exception was raised, return the response and usage details
    else:
        # Get the cost and token usage from the response
        usage = {
            "cost": round(
                llm_response.usage.cost_details["upstream_inference_cost"], 2
            ),
            "input_tokens": llm_response.usage.prompt_tokens,
            "output_tokens": llm_response.usage.completion_tokens,
        }
        return llm_response, elapsed_time, usage


def annotate_with_instructor(
    text: str,
    model: str,
    api_key: str | None,
    prompt: str,
    response_model: ListOfEntities | ListOfEntitiesPositions | None,
    max_retries: int = 3,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[
    ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions | None,
    float | int,
    dict[str, float | int],
]:
    """
    Annotate a text using the Instructor framework.

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
    ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions | None
        Annotation result returned by the LLM. The returned type depends on
        whether validation is enabled and on the chosen response model.
    float | int:
        The time elapsed for the inference.
    dict[str, float | int]:
        A dictionary containing cost usage and token counts.
    """
    # Instantiate an Instructor client for the requested model.
    model_entry_point = (
        f"openrouter/{model}" if not model.startswith("openai") else model
    )
    client = instructor.from_provider(
        model_entry_point,
        async_client=False,
        mode=instructor.Mode.JSON,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    no_response = (
        None,
        0,
        {"cost": 0, "input_tokens": 0, "output_tokens": 0},
    )
    try:
        # Query the LLM
        start_time = time.perf_counter()
        llm_response, completion = client.create_with_completion(
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
            # The response is validated against the provided Pydantic model.
            response_model=response_model,
            max_retries=max_retries,
        )
        elapsed_time = time.perf_counter() - start_time

    except InstructorValidationError as exc:
        # Raised when the LLM output does not conform to the expected schema.
        logger.warning(f"Validation failed: {exc}")
        return no_response
    except InstructorRetryException as exc:
        # Raised when all retry attempts fail.
        logger.warning(f"Failed after {exc.n_attempts} attempts")
        logger.warning(f"Last completion: {exc.last_completion}")
        return no_response
    except (ProviderError, ModeError) as exc:
        # Catch-all for provider-level, mode-related, or parsing errors.
        logger.warning(f"Instructor error: {exc}")
        return no_response
    else:
        usage = {
            "cost": round(completion.usage.cost_details["upstream_inference_cost"], 2),
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }
        return llm_response, elapsed_time, usage


def annotate_with_pydanticai(
    text: str,
    model: str,
    api_key: str | None,
    prompt: str,
    response_model: ListOfEntities | ListOfEntitiesPositions | None,
    max_retries: int = 3,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[
    ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions | None,
    float | int,
    dict[str, float | int],
]:
    """
    Annotate a text using the PydanticAI framework.

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
    dict[str, float | int]:
        A dictionary containing cost usage and token counts.
    """
    # Instantiate an PydanticAI client for the requested model
    client = OpenRouterModel(model, provider=OpenRouterProvider(api_key=api_key))
    no_response = (
        None,
        0,
        {"cost": 0, "input_tokens": 0, "output_tokens": 0},
    )
    try:
        # Query the LLM
        start_time = time.perf_counter()
        agent = Agent(
            model=client,
            output_type=response_model,
            retries=max_retries,
            system_prompt=("Extract entities as structured JSON."),
        )
        raw_llm_response = agent.run_sync(f"{prompt}\n{text}")
        llm_response = raw_llm_response.output
        elapsed_time = time.perf_counter() - start_time

    except PydanticValidationError as e:
        logger.error(f"Pydantic validation error: {e}")
        return no_response
    except CoreValidationError as e:
        logger.error(f"Core validation error: {e}")
        return no_response
    except UnexpectedModelBehavior as e:
        logger.error(f"Unexpected model behavior: {e}")
        return no_response
    else:
        return (
            llm_response,
            elapsed_time,
            {"cost": None, "input_tokens": None, "output_tokens": None},
        )


def extract_content(raw: ChatCompletion) -> str | None:
    """
    Extract message content from a ChatCompletion-like object.

    Parameters
    ----------
    raw : ChatCompletion
        The raw output from the LLM, expected to be a ChatCompletion object
        containing a list of choices, where each choice has a message with content.

    Returns
    -------
    str | None
        The content string extracted from the first choice's message, or None if
        the expected structure is not present (e.g., in tool-call scenarios).

    Raises
    ------
    ValueError
        If the expected structure is not present.
    """
    if hasattr(raw, "choices"):
        choices = getattr(raw, "choices", None)
        if not choices:
            msg = "ChatCompletion has no choices"
            raise ValueError(msg)
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            msg = "Missing message in first choice"
            raise ValueError(msg)

        return getattr(message, "content", None)

    return raw


def normalize_llm_output(
    raw: ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions | None,
) -> str | ListOfEntities | ListOfEntitiesPositions | None:
    """
    Normalize heterogeneous LLM outputs into a consistent format.

    Rules
    -----
    - If input is a Pydantic model (e.g. ListOfEntities), return as-is.
    - If input is a ChatCompletion, extract message.content.
    - If input is already dict, return as-is.
    - Otherwise return input unchanged.

    Parameters
    ----------
    raw : ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions | None
        The raw output from the LLM, which can be in various formats depending on
        the framework used.

    Returns
    -------
    str | ListOfEntities | ListOfEntitiesPositions
        If the input was a ChatCompletion, returns the content string.
        For the other types, returns the input as-is
        (which could be a Pydantic model or a raw string).
    """
    # OpenAI ChatCompletion object
    if hasattr(raw, "choices"):
        return extract_content(raw)
    else:
        return raw


def save_json_output(
    json_output_path: Path,
    json_data: dict[str, str],
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """
    Save structured annotation metadata to a JSON file.

    Parameters
    ----------
    json_output_path : Path
        Path where the JSON file will be written.
    json_data : dict[str, str]
        Dictionary containing metadata and model outputs.
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger

    Raises
    ------
    FileNotFoundError
        If the parent directory does not exist.
    PermissionError
        If writing permissions are insufficient.
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
        logger.debug(
            f"Saved JSON output to {json_output_path} "
            "with metadata and model response successfully."
        )
    except FileNotFoundError as exc:
        msg = f"Directory does not exist for output file: {json_output_path}"
        raise FileNotFoundError(msg) from exc
    except PermissionError as exc:
        msg = f"Permission denied when writing to {json_output_path}"
        raise PermissionError(msg) from exc
    except OSError as exc:
        msg = f"Failed to write JSON to {json_output_path}: {exc}"
        raise OSError(msg) from exc
    except TypeError as exc:
        msg = f"Invalid data provided for JSON serialization: {exc}"
        raise ValueError(msg) from exc


def save_txt_output(
    txt_output_path: Path,
    content: str,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """
    Save raw LLM response to a text file.

    Parameters
    ----------
    txt_output_path : Path
        Path where the text file will be written.
    content : str
        Raw string content to save.
    logger : loguru.Logger, optional
        Logger for logging messages, by default loguru.logger

    Raises
    ------
    FileNotFoundError
        If the parent directory does not exist.
    PermissionError
        If writing permissions are insufficient.
    OSError
        If a system-level error occurs during writing.
    """
    try:
        txt_output_path.write_text(content, encoding="utf-8")
        logger.debug(f"Saved raw response to {txt_output_path} successfully.")
    except FileNotFoundError as exc:
        msg = f"Directory does not exist for output file: {txt_output_path}"
        raise FileNotFoundError(msg) from exc
    except PermissionError as exc:
        msg = f"Permission denied when writing to {txt_output_path}"
        raise PermissionError(msg) from exc
    except OSError as exc:
        msg = f"Failed to write raw response to {txt_output_path}: {exc}"
        raise OSError(msg) from exc


def extract_entities(
    prompt_tag: str,
    prompt_file: Path,
    model: str,
    text_path: Path,
    framework: str,
    output_dir: Path,
    max_retries: int,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """
    Extract structured entities from a text using a specified LLM and framework.

    Parameters
    ----------
    prompt_file : Path
        Path to a text file containing the extraction prompt.
    model : str
        Model name to use for extraction.
    text_path : str
        Path to the JSON text to process.
    framework : str
        Validation framework.
    output_dir : str
        Directory to save output files.
    max_retries : int (Default is 3)
        Maximum number of retries in case of API or validation failure.
        If either the text file or prompt file does not exist.
    logger : loguru.Logger
        Logger instance for logging messages.
    """
    # Log input parameters
    logger.debug(f"Text to annotate: {text_path}")
    logger.debug(f"Model: {model}")
    logger.debug(f"Framework: {framework}")
    logger.debug(f"Tag prompt: {prompt_tag}")
    logger.debug(f"Prompt file: {prompt_file}")
    logger.debug(f"Output directory: {output_dir}")
    logger.debug(f"Max retries: {max_retries}")
    # Load info from the JSON file:
    # raw text, ground truth entities and URL if available
    text_to_annotate, groundtruth, url = load_text_and_metadata(
        text_path, prompt_tag, logger
    )
    # Load prompt from txt file
    prompt = load_prompt(prompt_file, logger)
    # Set response model and retries based on tag and framework
    if framework:
        if prompt_tag == "json":
            response_model = ListOfEntities
        else:
            response_model = ListOfEntitiesPositions
    # Retrieve the openrouter api key
    api_key = load_api_key("OPENROUTER_API_KEY")
    # Run annotation and time it
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_T%H-%M-%S")
    if framework == "none":
        raw_llm_response, inference_time, usage = annotate_without_framework(
            text_to_annotate,
            model,
            api_key,
            prompt,
            logger,
        )
    if framework == "instructor":
        raw_llm_response, inference_time, usage = annotate_with_instructor(
            text_to_annotate,
            model,
            api_key,
            prompt,
            response_model,
            max_retries,
            logger,
        )

    elif framework == "pydanticai":
        raw_llm_response, inference_time, usage = annotate_with_pydanticai(
            text_to_annotate,
            model,
            api_key,
            prompt,
            response_model,
            max_retries,
            logger=logger,
        )
    # Normalize the LLM output to extract the content string if it's a ChatCompletion,
    llm_response = normalize_llm_output(raw_llm_response)
    llm_response_str = serialize_response(llm_response)
    if raw_llm_response is None:
        logger.warning(f"LLM did not return a response for text '{text_path.name}'.")
    else:
        logger.debug(f"LLM response: {llm_response}")
        logger.debug(f"Inference time: {inference_time:.2f} seconds")
        logger.debug(f"Input tokens: {usage['input_tokens']}")
        logger.debug(f"Output tokens: {usage['output_tokens']}")
        logger.debug(f"Cost usage: {usage['cost']} $")
    # Prepare output paths
    model_safe = sanitize_filename(model)
    base_name = f"{text_path.stem}_{model_safe}_{framework}_{timestamp}"
    json_output_path = output_dir / f"{base_name}.json"
    txt_output_path = output_dir / f"{base_name}.txt"
    # Save JSON metadata + response
    response_metadata = {
        "timestamp": timestamp,
        "json_path": str(text_path),
        "text": serialize_response(text_to_annotate),
        "url": url,
        "model_name": model,
        "framework_name": framework,
        "prompt_path": str(prompt_file),
        "prompt_tag": prompt_tag,
        "groundtruth": serialize_response(groundtruth),
        "raw_llm_response": serialize_response(raw_llm_response),
        "llm_response": llm_response_str,
        "inference_time_sec": inference_time,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "inference_cost_usd": usage["cost"],
        "response_file": str(txt_output_path),
    }
    save_json_output(json_output_path, response_metadata, logger)
    # Save parsed response in a txt file
    save_txt_output(txt_output_path, llm_response_str, logger)


@click.command()
@click.option(
    "--text-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the JSON annotation file to process.",
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
    prompt_tag: str,
    prompt_file: Path,
    model: str,
    text_path: Path,
    framework: str,
    output_dir: Path,
    max_retries: int,
) -> None:
    """CLI entrypoint."""
    logger = create_logger(level="DEBUG")
    logger.info("Starting the extraction of entities.")
    extract_entities(
        prompt_tag=prompt_tag,
        prompt_file=prompt_file,
        model=model,
        text_path=text_path,
        framework=framework,
        output_dir=output_dir,
        max_retries=max_retries,
        logger=logger,
    )
    logger.success("Completed the extraction of entities successfully!")


if __name__ == "__main__":
    main()
