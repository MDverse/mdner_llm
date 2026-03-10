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
from instructor.core.exceptions import \
    ValidationError as InstructorValidationError
from llama_index.llms.openai import OpenAI as llamaOpenAI
from llama_index.llms.openrouter import OpenRouter
from openai.types.chat import ChatCompletion
from pydantic import ValidationError as PydanticValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_core import ValidationError as CoreValidationError

from mdner_llm.core.logger import create_logger
from mdner_llm.models.entities import ListOfEntities
from mdner_llm.models.entities_with_positions import ListOfEntitiesPositions
from mdner_llm.utils.common import (ensure_dir, load_api_key,
                                    sanitize_filename, serialize_response)


def load_text_and_groundtruth(
    path_text: str | Path, tag_prompt: str, logger: "loguru.Logger" = loguru.logger
) -> tuple[str, ListOfEntities | ListOfEntitiesPositions]:
    """Load raw text and ground truth annotations from a JSON file.

    The JSON file must contain a ``raw_text`` field and an ``entities`` field.
    Depending on the value of ``tag_prompt``, entities are normalized or kept
    with positional information.

    Parameters
    ----------
    path_text
        Path to the JSON file containing the text and annotations.
    tag_prompt
        Annotation format selector. If equal to `json`, entity positions
        are removed.
    logger
        Logger instance for logging messages.


    Returns
    -------
    tuple[str, str, ListOfEntities | ListOfEntitiesPositions]
        The raw text to annotate and the corresponding ground truth object.

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
    # Normalize entities based on the tag_prompt value
    if tag_prompt == "json":
        # Remove positional information from entities
        normalized = [
            {"label": ent.get("label"), "text": ent.get("text")} for ent in entities
        ]
        # Create a ListOfEntities object for the ground truth
        groundtruth = ListOfEntities(entities=normalized)
    else:
        # Create a ListOfEntitiesPositions object for the ground truth
        groundtruth = ListOfEntitiesPositions(entities=entities)
    preview = text_to_annotate[:75].replace("\n", " ")
    logger.debug(f"Loaded text ({len(text_to_annotate)} chars): {preview}...")
    return text_to_annotate, groundtruth


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


def annotate_with_instructor(
    text: str,
    model: str,
    api_key: str | None,
    prompt: str,
    response_model: ListOfEntities | ListOfEntitiesPositions | None,
    max_retries: int = 3,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[
    ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions | None, float | int
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

    except InstructorValidationError as exc:
        # Raised when the LLM output does not conform to the expected schema.
        logger.warning(f"Validation failed: {exc}")
        return None, 0

    except InstructorRetryException as exc:
        # Raised when all retry attempts fail.
        logger.warning(f"Failed after {exc.n_attempts} attempts")
        logger.warning(f"Last completion: {exc.last_completion}")
        return exc.last_completion, 0

    except (ProviderError, ModeError) as exc:
        # Catch-all for provider-level, mode-related, or parsing errors.
        logger.warning(f"Instructor error: {exc}")
        return None, 0

    else:
        # Only executes if no exception was raised
        elapsed_time: int | float = time.time() - start_time
        return llm_response, elapsed_time


def annotate_with_llamaindex(
    text: str,
    model: str,
    api_key: str | None,
    prompt: str,
    response_model: ListOfEntities | ListOfEntitiesPositions | None,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[
    ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions | None, float | int
]:
    """
    Annotate a text using the Llamaindex framework.

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
    """
    # Instantiate an Llamaindex client
    # Using openai api key for openai models
    if model.startswith("openai"):
        openai_api_key = load_api_key("OPENAI_API_KEY")
        model_name = model.split("/")[1]
        base_llm = llamaOpenAI(model=model_name, api_key=openai_api_key)
    else:
        # Using openrouter api key for other models
        base_llm = OpenRouter(model=model, api_key=api_key)

    # Structured response
    if response_model is not None:
        try:
            # Query the LLM
            start_time = time.time()
            structured_llm = base_llm.as_structured_llm(output_cls=response_model)
            llm_response = structured_llm.complete(f"{prompt}\n{text}").raw
            elapsed_time: int | float = time.time() - start_time

        except (ValueError, PydanticValidationError) as exc:
            logger.warning(
                "Structured parsing failed, falling back "
                f"to the raw llm response: {exc}"
            )
        else:
            # Only executes if no exception was raised
            elapsed_time: int | float = time.time() - start_time
            return llm_response, elapsed_time

    # Raw LLM response
    start_time = time.time()
    raw_response = base_llm.complete(f"{prompt}\n{text}").text
    elapsed = time.time() - start_time
    return raw_response, elapsed


def annotate_with_pydanticai(
    text: str,
    model: str,
    api_key: str | None,
    prompt: str,
    response_model: ListOfEntities | ListOfEntitiesPositions | None,
    max_retries: int = 3,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[
    ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions | None, float | int
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
    """
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

    except PydanticValidationError as e:
        logger.error(f"Pydantic validation error: {e}")
        return None, 0

    except CoreValidationError as e:
        logger.error(f"Core validation error: {e}")
        return None, 0

    except UnexpectedModelBehavior as e:
        logger.error(f"Unexpected model behavior: {e}")
        return None, 0
    else:
        # Only executes if no exception was raised
        elapsed_time: int | float = time.time() - start_time
        return llm_response, elapsed_time


def extract_entities(
    tag_prompt: str,
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
    path_text : str
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
    logger.debug(f"Text to annotate: {text_path}")
    logger.debug(f"Model: {model}")
    logger.debug(f"Framework: {framework}")
    logger.debug(f"Tag prompt: {tag_prompt}")
    logger.debug(f"Prompt file: {prompt_file}")
    logger.debug(f"Output directory: {output_dir}")
    logger.debug(f"Max retries: {max_retries}")
    
    # Load text to annotate
    text_to_annotate, groundtruth = load_text_and_groundtruth(
        text_path, tag_prompt, logger
    )

    # Load prompt from txt file
    prompt = load_prompt(prompt_file, logger)

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
    api_key = load_api_key("OPENROUTER_API_KEY")

    # Run annotation and time it
    if framework in {"instructor", "none"}:
        if framework == "none":
            response_model = None
            max_retries = 0
        llm_response, inference_time = annotate_with_instructor(
            text_to_annotate,
            model,
            api_key,
            prompt,
            response_model,
            max_retries,
            logger,
        )

    elif framework == "llamaindex":
        llm_response, inference_time = annotate_with_llamaindex(
            text_to_annotate, model, api_key, prompt, response_model, logger=logger
        )

    elif framework == "pydanticai":
        llm_response, inference_time = annotate_with_pydanticai(
            text_to_annotate,
            model,
            api_key,
            prompt,
            response_model,
            max_retries,
            logger=logger,
        )

    if llm_response is None:
        logger.warning(
            f"LLM did not return a response for text '{text_path.name}'. "
            "Skipping saving of annotation results."
        )
        return
    else:
        logger.debug(f"LLM response: {serialize_response(llm_response)}")
        logger.debug(f"Inference time: {inference_time:.2f} seconds")
    # Prepare output paths
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    model_safe = sanitize_filename(model)
    base_name = f"{text_path.stem}_{model_safe}_{framework}_{timestamp}"
    json_output_path = output_dir / f"{base_name}.json"
    txt_output_path = output_dir / f"{base_name}.txt"

    # Save JSON metadata + response
    json_data = {
        "timestamp": timestamp,
        "output_file": str(txt_output_path),
        "text_file": str(text_path),
        "framework_name": framework,
        "model_name": model,
        "prompt_path": str(prompt_file),
        "tag_prompt": tag_prompt,
        "inference_time_sec": inference_time,
        "raw_llm_response": serialize_response(llm_response),
        "groundtruth": serialize_response(groundtruth),
    }
    try:
        json_output_path.write_text(
            json.dumps(json_data, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        logger.debug(f"Saved JSON output to {json_output_path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Directory does not exist for output file: {json_output_path}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied when writing to {json_output_path}") from e
    except OSError as e:
        raise OSError(f"Failed to write JSON to {json_output_path}: {e}") from e
    except TypeError as e:
        raise ValueError(f"Invalid data provided for JSON serialization: {e}") from e
    
    # Save raw model response
    try:
        txt_output_path.write_text(
            serialize_response(llm_response),
            encoding="utf-8",
        )
        logger.debug(f"Saved raw response to {txt_output_path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Directory does not exist for output file: {txt_output_path}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied when writing to {txt_output_path}") from e
    except OSError as e:
        raise OSError(f"Failed to write raw response to {txt_output_path}: {e}") from e


@click.command()
@click.option(
    "--text-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the JSON annotation file to process.",
)
@click.option(
    "--model", required=True, type=str, 
    help="LLM model name to use for extraction." \
    "Find available models in OpenRouter (https://openrouter.ai/models)."
)
@click.option(
    "--framework",
    default="none",
    type=click.Choice(["instructor", "llamaindex", "pydanticai", "none"]),
    help="Validation framework to apply to model outputs." \
    "Choices: 'instructor', 'llamaindex', 'pydanticai'. " \
    "If 'none', no validation is applied and the raw model response is returned.",
)
@click.option(
    "--tag-prompt",
    default="json",
    type=click.Choice(["json", "json_with_positions"]),
    help="Descriptor indicating the format of the expected LLM output." \
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
    default="results/llm_annotations",
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
    tag_prompt: str,
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
        tag_prompt=tag_prompt,
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
