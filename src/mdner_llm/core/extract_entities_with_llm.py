"""Extract structured entities from a text using a specified LLM and framework."""

import json
import time
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import click
import instructor
import loguru
from instructor.core.exceptions import (
    ClientError,
    ConfigurationError,
    InstructorRetryException,
    ModeError,
    ProviderError,
    ResponseParsingError,
    ValidationError,
)
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage
from pydantic import ValidationError as PydanticValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_core import ValidationError as CoreValidationError

from mdner_llm.common import ensure_dir, load_api_key, sanitize_filename
from mdner_llm.logger import create_logger
from mdner_llm.models.entities import ListOfEntities


def load_text_and_metadata(
    path_text: str | Path, logger: "loguru.Logger" = loguru.logger
) -> tuple[str, ListOfEntities, str]:
    """Load raw text and ground truth annotations from a JSON file.

    Returns
    -------
    tuple[str, str, ListOfEntities, str]
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
    logger.debug(f"Loading text and metadata from {path_text}.")
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
    text, entities = data["raw_text"], data["entities"]
    # Remove positional information from entities
    normalized = [
        {"category": ent.get("category"), "text": ent.get("text")} for ent in entities
    ]
    logger.debug(f"Loaded text ({len(text)} chars): {text[:75].replace('\n', ' ')}...")
    return text, ListOfEntities(entities=normalized), data.get("url")


def load_prompt(prompt_file: Path, logger: "loguru.Logger" = loguru.logger) -> str:
    """Load the JSON few-shot prompt from the mdner_llm package.

    Returns
    -------
    str
        The prompt content.

    Raises
    ------
    FileNotFoundError
        If the prompt file does not exist in the package resources.
    """
    logger.debug(f"Loading prompt from {prompt_file}.")
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
    logger.debug(
        f"Loaded prompt ({len(prompt)} chars) : {prompt[:75].replace('\n', ' ')}..."
    )
    return prompt


def normalize_to_pydantic_model(
    raw_llm_response: ChatCompletion | None,
    metadata: dict,
) -> tuple[ListOfEntities, dict]:
    """
    Normalize raw LLM response into a ListOfEntities Pydantic model.

    Returns
    -------
    tuple[ListOfEntities, dict]
        A tuple containing the normalized Pydantic model and the updated metadata.
    """
    # Default empty model
    empty_model = ListOfEntities(entities=[])
    # If API already failed, don't go further
    if metadata.get("status") == "api_error" or raw_llm_response is None:
        return empty_model, metadata
    # Check JSON format validity
    try:
        # Extract content safely
        content = raw_llm_response.choices[0].message.content
        # Attempt to parse the content as JSON
        parsed = json.loads(content)
    # Return empty model if content is not valid JSON
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
        metadata["status"] = "format_error"
        return empty_model, metadata
    # Check Pydantic validation
    try:
        model = ListOfEntities.model_validate(parsed)
    except PydanticValidationError:
        metadata["status"] = "format_error"
        return empty_model, metadata
    # Success
    metadata["status"] = "ok"
    return model, metadata


def update_metadata(
    metadata: dict,
    start_time: float,
    raw_llm_response: ChatCompletion,
) -> dict:
    """
    Update the metadata dictionary with status, raw LLM response, and usage details.

    Returns
    -------
    dict
        The updated metadata dictionary with usage details added.
    """
    # Add inference time to usage details
    metadata["usage"]["inference_time_sec"] = time.perf_counter() - start_time
    # Store the raw LLM response in metadata for later reference
    metadata["raw_llm_response"] = raw_llm_response
    usage = getattr(raw_llm_response, "usage", None)
    if isinstance(usage, CompletionUsage):
        # Get the token and cost usage from the response
        metadata["usage"]["input_tokens"] = usage.prompt_tokens
        metadata["usage"]["output_tokens"] = usage.completion_tokens
        metadata["usage"]["cost"] = usage.cost_details["upstream_inference_cost"]
    return metadata


def annotate_without_framework(
    model: str,
    api_key: str | None,
    metadata: dict,
    messages: list[dict[str, str]],
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[ListOfEntities, dict[str, ChatCompletion | float | None]]:
    """
    Annotate a text without applying any validation framework.

    Returns
    -------
    ListOfEntities
        The normalized Pydantic model with extracted entities
        (may be empty if format error).
    dict[str, ChatCompletion | float | None]
        The updated metadata dictionary with inference details and status.
    """
    # Instantiate an OpenAI client for the requested model
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    try:
        # Query the LLM and time the inference
        start_time = time.perf_counter()
        llm_response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    # Handle common OpenAI API exceptions
    except (APIError, RateLimitError, APIConnectionError) as exc:
        logger.warning(f"API error: {exc}")
        metadata["status"] = "api_error"
        return normalize_to_pydantic_model(None, metadata)
    else:
        updated_metadata = update_metadata(metadata, start_time, llm_response)
        return normalize_to_pydantic_model(llm_response, updated_metadata)


def annotate_with_instructor(
    model: str,
    api_key: str | None,
    metadata: dict,
    messages: list[dict[str, str]],
    response_model: ListOfEntities | None,
    max_retries: int = 3,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[
    ListOfEntities,
    dict[str, float | ChatCompletion | None],
]:
    """
    Annotate a text using the Instructor framework.

    Returns
    -------
    ListOfEntities
        The normalized Pydantic model with extracted entities.
    dict[str, float | ChatCompletion | None]
        The updated metadata dictionary with inference details and status.
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
    try:
        # Query the LLM
        start_time = time.perf_counter()
        llm_response, completion = client.create_with_completion(
            messages=messages,
            # The response is validated against the provided Pydantic model.
            response_model=response_model,
            max_retries=max_retries,
        )
    # Handle common Instructor exceptions
    # related to validation (format errors, retries, and response parsing issues)
    except (ValidationError, InstructorRetryException, ResponseParsingError) as exc:
        logger.error(f"Validation error: {exc}")
        metadata["status"] = "format_error"
        return normalize_to_pydantic_model(None, metadata)
    # related to API (provider issues, and mode errors)
    except (
        ConfigurationError,
        ClientError,
        ProviderError,
        ModeError,
    ) as exc:
        logger.error(f"API error: {exc}")
        metadata["status"] = "api_error"
        return normalize_to_pydantic_model(None, metadata)
    else:
        updated_metadata = update_metadata(metadata, start_time, completion)
        updated_metadata["status"] = "ok"
        return llm_response, updated_metadata


def annotate_with_pydanticai(
    model: str,
    api_key: str | None,
    metadata: dict,
    messages: tuple[str, str],
    response_model: ListOfEntities | None,
    max_retries: int = 3,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[
    ListOfEntities,
    dict[str, float | ChatCompletion | None],
]:
    """
    Annotate a text using the PydanticAI framework.

    Returns
    -------
    ListOfEntities
        The normalized Pydantic model with extracted entities.
    dict[str, float | ChatCompletion | None]
        The updated metadata dictionary with inference details and status.
    """
    # Instantiate an PydanticAI client for the requested model
    client = OpenRouterModel(model, provider=OpenRouterProvider(api_key=api_key))
    try:
        # Query the LLM
        start_time = time.perf_counter()
        agent = Agent(
            model=client,
            output_type=response_model,
            retries=max_retries,
            system_prompt=(messages[0]),
        )
        raw_llm_response = agent.run_sync(messages[1])
        llm_response = raw_llm_response.output

    except (
        PydanticValidationError,
        CoreValidationError,
        UnexpectedModelBehavior,
    ) as exc:
        logger.error(f"Validation error: {exc}")
        metadata["status"] = "format_error"
        return normalize_to_pydantic_model(None, metadata)
    else:
        updated_metadata = update_metadata(metadata, start_time, raw_llm_response)
        updated_metadata["status"] = "ok"
        return llm_response, updated_metadata


def annotate_with_llm_and_framework(
    framework: str,
    text_to_annotate: str,
    model: str,
    api_key: str | None,
    prompt: str,
    response_model=ListOfEntities,
    max_retries: int = 3,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[ListOfEntities, dict[str, float | ChatCompletion | None]]:
    """
    Annotate a text using a specified LLM and validation framework.

    Returns
    -------
    ListOfEntities
        The normalized Pydantic model with extracted entities.
        (may be empty if format error).
    dict[str, float | ChatCompletion | None]
        The updated metadata dictionary with inference details and status.
    """
    logger.debug(f"Starting annotation with model {model} using {framework}.")
    # Initialize metadata dictionary to store inference details
    metadata = {
        "raw_llm_response": None,
        "usage": {
            "inference_time_sec": None,
            "cost": None,
            "input_tokens": None,
            "output_tokens": None,
        },
        "status": None,
    }
    # Define the system and user messages for the LLM prompt
    system_msg = "Extract entities as structured JSON."
    user_msg = f"{prompt}\n{text_to_annotate}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    # Route to the appropriate annotation function based on the specified framework
    if framework == "noframework":
        return annotate_without_framework(model, api_key, metadata, messages, logger)
    if framework == "instructor":
        return annotate_with_instructor(
            model,
            api_key,
            metadata,
            messages,
            response_model,
            max_retries,
            logger,
        )
    if framework == "pydanticai":
        return annotate_with_pydanticai(
            model,
            api_key,
            metadata,
            (system_msg, user_msg),
            response_model,
            max_retries,
            logger,
        )


def save_raw_response_to_txt(
    txt_output_path: Path,
    content: ChatCompletion | None,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Save raw LLM response to a text file."""
    try:
        # Case PydanticAI
        content = content.output if hasattr(content, "output") else content
        # Serialize the content to JSON string
        serialized_content = content.model_dump_json(indent=None) if content else ""
        # Write the serialized content to the specified text file
        txt_output_path.write_text(serialized_content, encoding="utf-8")
        logger.debug(f"Saved raw response to {txt_output_path} successfully.")
    except FileNotFoundError:
        logger.error(f"Directory does not exist for output file: {txt_output_path}")

    except OSError as exc:
        logger.error(f"Failed to write raw response to {txt_output_path}: {exc}")


def save_formated_response_with_metadata_to_json(
    json_output_path: Path,
    json_data: dict[str, str],
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """
    Save structured annotation metadata to a JSON file.

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
        logger.debug(
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


def extract_entities(
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
    # Load info from the JSON file:
    # raw text, ground truth entities and URL if available
    text_to_annotate, groundtruth, url = load_text_and_metadata(text_path, logger)
    # Load prompt from txt file
    prompt = load_prompt(prompt_file, logger)
    # Retrieve the openrouter api key
    api_key = load_api_key("OPENROUTER_API_KEY")
    # Run annotation and time it
    formatted_llm_response, inference_metadata = annotate_with_llm_and_framework(
        framework,
        text_to_annotate,
        model,
        api_key,
        prompt,
        response_model=ListOfEntities,
        max_retries=max_retries,
        logger=logger,
    )
    usage = inference_metadata.get("usage", {})
    logger.debug(f"Response status: {inference_metadata['status']}.")
    logger.debug(f"Formatted LLM response: {formatted_llm_response}")
    logger.debug(f"Inference time: {usage.get('inference_time_sec')} seconds.")
    logger.debug(f"Input tokens: {usage.get('input_tokens')}.")
    logger.debug(f"Output tokens: {usage.get('output_tokens')}.")
    logger.debug(f"Cost usage: {usage.get('cost')} $.")
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prepare output path
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    txt_output_path = Path(
        output_dir / f"{text_path.stem}_{sanitize_filename(model)}_{framework}_{ts}.txt"
    )
    # Save raw response into a txt file
    save_raw_response_to_txt(
        txt_output_path, inference_metadata["raw_llm_response"], logger
    )
    # Save formated response with metadata in a JSON file
    response_metadata = {
        "timestamp": ts,
        "input_json_path": str(text_path),
        "text": text_to_annotate,
        "url": url,
        "model_name": model,
        "framework_name": framework,
        "prompt_path": str(prompt_file),
        "groundtruth": groundtruth.model_dump(),
        "status": inference_metadata["status"],
        "formatted_response": formatted_llm_response.model_dump(),
        "inference_time_sec": usage.get("inference_time_sec"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "inference_cost_usd": usage.get("cost"),
        "raw_llm_response_file": str(txt_output_path),
    }
    save_formated_response_with_metadata_to_json(
        txt_output_path.with_suffix(".json"), response_metadata, logger
    )


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
    help="LLM model name to use for extraction."
    "Find available models in OpenRouter (https://openrouter.ai/models).",
)
@click.option(
    "--framework",
    default="noframework",
    type=click.Choice(["instructor", "pydanticai", "noframework"]),
    help="Validation framework to apply to model outputs.",
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
    run_main_from_cli()
