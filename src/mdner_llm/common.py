"""Common utility functions used across the project."""

import os
import re
from pathlib import Path

import loguru
from dotenv import load_dotenv


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
