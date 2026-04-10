"""Common utility functions used across the project."""

import json
import os
import re
import sys
from pathlib import Path

import loguru
from dotenv import load_dotenv
from loguru import logger
from openai.types.chat import ChatCompletion

from mdner_llm.models.entities import ListOfEntities
from mdner_llm.models.entities_with_positions import ListOfEntitiesPositions


def create_logger(
    logpath: str | Path | None = None, level: str = "INFO"
) -> "loguru.Logger":
    """Create the logger with optional file logging.

    Parameters
    ----------
    logpath : str | Path | None, optional
        Path to the log file. If None, no file logging is done.
    level : str, optional
        Logging level. Default is "INFO".

    Returns
    -------
    loguru.Logger
        Configured logger instance.
    """
    # Define log format.
    logger_format = (
        "{time:YYYY-MM-DD HH:mm:ss} "
        "| <level>{level:<8}</level> "  # noqa: RUF027
        "| <level>{message}</level>"
    )
    # Remove default logger.
    logger.remove()
    # Add logger to path (if path is provided).
    if logpath:
        # Create parent directories.
        Path(logpath).parent.mkdir(parents=True, exist_ok=True)
        # Add logger to file.
        logger.add(logpath, format=logger_format, level="DEBUG", mode="w")
    # Add logger to stdout.
    logger.add(sys.stdout, format=logger_format, level=level)
    return logger


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


def serialize_response(
    resp: ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str | dict,
) -> str:
    """
    Serialize various response objects into a JSON-safe string representation.

    Parameters
    ----------
    resp : ListOfEntities | ListOfEntitiesPositions | ChatCompletion | str | dict
        The object to serialize. This may be a string, a custom class instance,
        or a model response object such as ChatCompletion.

    Returns
    -------
    str
        A JSON-compatible string representation of the input object.
    """
    # Handle the case where the response is None
    if resp is None:
        return ""
    # If it's already a string, nothing to do.
    if isinstance(resp, str):
        return resp
    # If it's a Pydantic model (like ListOfEntities or ListOfEntitiesPositions),
    # we can use the model's built-in serialization method
    if hasattr(resp, "model_dump_json"):
        return resp.model_dump_json(indent=2)
    # If it's a dict, we can serialize it to JSON
    if isinstance(resp, dict):
        return json.dumps(resp, indent=2)
    # Otherwise, we can try to convert it to a string directly
    return str(resp)


def load_api_key(key: str) -> str:
    """
    Load an API key from .env file or environment variables.

    Parameters
    ----------
    key : str
        The name of the environment variable containing the API key.

    Returns
    -------
    str
        The API key string.

    Raises
    ------
    ValueError
        If the API key is not found in the environment variables.
    """
    # Load all environment variables from .env file (if it exists)
    load_dotenv()
    # to ensure we can access them via os.getenv
    api_key = os.getenv(key)
    # If the key is not found in environment variables
    if api_key is None:
        msg = f"{key} must be set in the environment."
        # raise an error to prevent further execution without a valid API key
        raise ValueError(msg)
    return api_key


def list_json_files_from_txt(
    texts_path: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> list[Path]:
    """Read a text file containing paths to JSON files.

    Parameters
    ----------
    texts_path : Path
        Path to a text file where each line is a path to a JSON file.

    Returns
    -------
    list[Path]
        A list of Path objects corresponding to the JSON files listed in the text file.
    """
    logger.info(f"Reading list of JSON files from {texts_path}.")
    # Read the list of annotation text files from the provided path
    selected_files = [
        Path(line.strip())
        for line in texts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total_files = len(selected_files)
    logger.success(f"Found {total_files} JSON files successfully.")
    return selected_files
