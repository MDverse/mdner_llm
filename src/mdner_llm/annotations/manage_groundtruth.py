"""Build the groundtruth corpus from a CSV list of JSON file names.

For each file name listed in the input CSV, this script:
1. Logs an info message if the file already exists in data/groundtruth.
2. Moves the file from data/draft to data/groundtruth if found there.
3. Otherwise, creates it from the raw Zenodo/Figshare parquet datasets.
"""

import json
import shutil
from pathlib import Path

import click
import pandas as pd
from loguru import logger


def load_raw_datasets() -> pd.DataFrame:
    """Load and concatenate Zenodo and Figshare parquet datasets.

    Returns
    -------
    pd.DataFrame
        Combined dataframe with repository metadata.
    """
    zenodo = pd.read_parquet(Path("data/raw_datasets/zenodo_datasets.parquet"))
    figshare = pd.read_parquet(Path("data/raw_datasets/figshare_datasets.parquet"))
    return pd.concat([zenodo, figshare], ignore_index=True)


def build_entry(row: pd.Series) -> dict:
    """Build a groundtruth JSON entry from a raw dataset row.

    Returns
    -------
    dict
        Groundtruth entry with raw_text, entities and url.
    """
    raw_text = f"{row['title']}\n{row['description']}"
    return {
        "raw_text": raw_text,
        "entities": [],
        "url": row["dataset_url_in_repository"],
    }


def is_processed_file(
    file_name: str,
    raw_datasets: pd.DataFrame,
    groundtruth_dir: Path = Path("data/groundtruth"),
    draft_dir: Path = Path("data/draft"),
) -> bool:
    """Ensure a single groundtruth file exists, creating or moving it if needed.

    Returns
    -------
    bool
        True if the file was found, moved, or created; False if it could not be
        created from the raw datasets.
    """
    groundtruth_path = groundtruth_dir / file_name
    if groundtruth_path.exists():
        logger.info(f"{file_name} already in {groundtruth_dir}")
        return True

    draft_path = draft_dir / file_name
    if draft_path.exists():
        shutil.move(str(draft_path), str(groundtruth_path))
        logger.warning(f"{file_name} moved from {draft_dir} to {groundtruth_dir}")
        return True

    stem = Path(file_name).stem
    repository_name, dataset_id = stem.split("_", 1)
    match = raw_datasets[
        (raw_datasets["dataset_repository_name"] == repository_name)
        & (raw_datasets["dataset_id_in_repository"].astype(str) == dataset_id)
    ]
    if match.empty:
        logger.error(f"{file_name} not found in raw datasets, skipping")
        return False

    entry = build_entry(match.iloc[0])
    groundtruth_path.write_text(json.dumps(entry, ensure_ascii=False, indent=4))
    logger.warning(f"{file_name} created in {groundtruth_dir} from raw datasets")
    return True


@click.command()
@click.option(
    "--csv_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to CSV file containing a list of groundtruth JSON file names to build.",
)
def main(csv_path: Path) -> None:
    """Build missing groundtruth JSON files from a CSV of file names."""
    logger.info("Building groundtruth files...")
    df = pd.read_csv(csv_path)
    logger.success(f"Loaded {len(df)} file names from {csv_path}.")
    # Check for unexpected files in data/groundtruth
    csv_files = set(df["file_name"])
    groundtruth_dir = Path("data/groundtruth")
    draft_dir = Path("data/draft")
    existing_files = {f.name: f for f in groundtruth_dir.glob("*.json")}
    for file_name in existing_files.keys() - csv_files:
        shutil.move(existing_files[file_name], draft_dir / file_name)
        logger.warning(f"Moved {file_name} to {draft_dir} (not in CSV).")
    # Check for missing files in data/groundtruth
    raw_datasets = load_raw_datasets()
    counter = 0
    for file_name in df["file_name"]:
        if is_processed_file(file_name, raw_datasets, groundtruth_dir, draft_dir):
            counter += 1
    logger.success(f"Successfully processed {counter}/{len(df)} groundtruth files.")


if __name__ == "__main__":
    main()
