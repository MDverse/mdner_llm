"""Common utility functions used across the project."""

import os
import re
from pathlib import Path

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

    Returns
    -------
    str
        A sanitized string safe for use as a filename.
    """
    return re.sub(r"[^\w\-_.]", "_", s)


def load_api_key(key: str) -> str:
    """
    Load an API key from .env file or environment variables.

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
