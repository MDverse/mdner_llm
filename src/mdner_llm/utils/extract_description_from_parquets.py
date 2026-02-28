#!/usr/bin/env python3
"""
Extract title and description from parquet files.

This script scans a directory containing parquet files, extracts the
columns `title`, `description`, `dataset_repository_name`,
and `dataset_id_in_repository`, and creates one text file per row.

Each output file is named:
    <dataset_repository_name>_<dataset_id_in_repository>.txt

The file content format is:
    title

    description

Usage:
    python src/extract_description_from_parquets.py --input-dir data/ \
                                                    --output-dir annotations/v3

Arguments:
    --input-dir   Path to directory containing parquet files.
    --output-dir  Path to directory where text files will be written.

Example:
    uv run src/extract_description_from_parquets.py \
        --input-dir data/ \
        --output-dir annotations/v3
"""

import re
from collections.abc import Iterable
from pathlib import Path

import click
import pandas as pd
from loguru import logger

REQUIRED_COLUMNS = {
    "title",
    "description",
    "dataset_repository_name",
    "dataset_id_in_repository",
}


def sanitize_filename(value: str) -> str:
    """Return a filesystem-safe version of a string.

    This function replaces any character that is not a word character, hyphen, or dot
    with an underscore. It also trims leading and trailing whitespace.

    Parameters
    ----------
    value : str
        The string to sanitize.

    Returns
    -------
    str
        A sanitized version of the input string, safe for use as a filename.
    """
    value = re.sub(r"[^\w\-\.]+", "_", value.strip())
    return value


def iter_parquet_files(directory: Path) -> Iterable[Path]:
    """Yield all parquet files in a directory recursively.

    Parameters
    ----------
    directory : Path
        The directory to search for parquet files.

    Yields
    ------
    Path
        The path to each parquet file found.
    """
    yield from directory.rglob("*.parquet")


def process_parquet_file(parquet_path: Path, output_dir: Path) -> int:
    """Process a single parquet file and write extracted text files.

    This function reads the specified parquet file, checks for required columns,
    and for each row, creates a text file containing the title and description.

    Parameters
    ----------
    parquet_path : Path
        The path to the parquet file to process.
    output_dir : Path
        The directory where output text files will be written.

    Returns
    -------
    int
        The number of descriptions extracted from the parquet file.

    Raises
    ------
    ValueError
        If the parquet file is missing any of the required columns.
    """
    # Read the parquet file into a DataFrame
    df = pd.read_parquet(parquet_path)
    # Check for required columns
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        msg = f"Missing required columns in {parquet_path}: {missing_columns}"
        raise ValueError(msg)
    # Iterate over each row (dataset)
    for _, row in df.iterrows():
        # Sanitize repository name and dataset ID for safe filenames
        repo_name = sanitize_filename(str(row["dataset_repository_name"]))
        dataset_id = sanitize_filename(str(row["dataset_id_in_repository"]))
        # Create the output filename and path
        filename = f"{repo_name}_{dataset_id}.txt"
        output_path = output_dir / filename
        # Extract title and description, handling missing values
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        description = str(row["description"]) if pd.notna(row["description"]) else ""
        # Combine title and description into the content to write
        content = f"{title}\n\n{description}"
        # Save the content to a text file
        output_path.write_text(content, encoding="utf-8")
    # Return the number of descriptions extracted
    # (number of rows in the DataFrame)
    return len(df)


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing parquet files.",
    default="data/",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory where output text files will be written.",
)
def main(input_dir: Path, output_dir: Path) -> None:
    """CLI entry point.

    This function validates input arguments, ensures the output directory exists,
    and processes each parquet file found in the input directory.

    Parameters
    ----------
    input_dir : Path
        Path to the directory containing parquet files.
    output_dir : Path
        Path to the directory where output text files will be written.
    """
    logger.info("Starting the extraction process...")
    output_dir.mkdir(parents=True, exist_ok=True)
    nb_descriptions = 0
    parquet_files = list(iter_parquet_files(input_dir))
    if not parquet_files:
        logger.warning(f"No parquet files found in `{input_dir}`.")
        logger.info("Exiting without processing.")

    logger.info(f"Processing {len(parquet_files)} parquet files:")
    for parquet_file in parquet_files:
        logger.info(f"{parquet_file.name}")
        nb_descriptions += process_parquet_file(parquet_file, output_dir)
    logger.success(
        f"Saved {nb_descriptions} dataset descriptions "
        f"into `{output_dir}` successfully!"
    )


if __name__ == "__main__":
    main()
