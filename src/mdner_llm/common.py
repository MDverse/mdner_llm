"""Common utility functions used across the project."""

import os
import re
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

GREEK_LETTERS = {
    "α": "alpha",  # noqa: RUF001
    "β": "beta",
    "γ": "gamma",  # noqa: RUF001
    "δ": "delta",
    "ε": "epsilon",
    "ω": "omega",
}


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


def clean_text(text: str) -> str:
    """Normalize text for dedup, keeping the case.

    Returns
    -------
    str
        Text in NFKC form, with Greek letters spelled out, special dashes and
        quotes in ASCII and whitespace collapsed.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    text = text.strip()  # remove leading/trailing whitespace
    # Spell out Greek letters (β -> beta).
    for name, letter in GREEK_LETTERS.items():
        text = re.sub(rf"\b{name}\b", letter, text)
    # Replace special characters with their ASCII equivalents or remove them.
    table = str.maketrans(
        {
            # Non-breaking hyphen
            "‑": "-",  # e.g., "sars‑cov‑2" -> "sars-cov-2"  # noqa: RUF001, RUF003
            "\u2010": "-",  # NFKC converts the non-breaking hyphen into this one
            # Mathematical minus sign
            "−": "-",  # e.g., "U(VI)−complex" -> "U(VI)-complex"  # noqa: RUF001, RUF003
            # En dash (commonly used for charges in literature)
            "–": "-",  # e.g., "(-)-METH" -> "(-)-METH"  # noqa: RUF001
            # Em dash
            "—": "-",  # e.g., "protein—ligand" -> "protein-ligand"
            # Typographic/curly apostrophe
            "’": "'",  # e.g., "glycol’s" -> "glycol's"  # noqa: RUF001, RUF003
            # Soft hyphen / Invisible discretionary hyphen
            "\xad": None,  # e.g., "poly\xad(ethylene)" -> "poly(ethylene)"
        }
    )
    return text.translate(table)
