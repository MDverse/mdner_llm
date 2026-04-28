"""Build a vocabulary of named entities from JSON annotation files.

This script scans a directory of JSON annotation files, aggregates named
entities by class, normalizes entity text to lowercase, counts total
occurrences across all files, and generates one vocabulary file per entity class.

Each output file contains:
- A header reporting the number of unique entities for that class.
- One normalized entity per line with its total occurrence count.
"""

import json
import math
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from mdner_llm.annotations.colors import COLORS
from mdner_llm.logger import create_logger


def collect_entities(
    texts_path: Path,
) -> list[dict]:
    """
    Collect normalized entity counts per class from annotation files.

    Parameters
    ----------
    texts_path : Path
        Path to a directory containing JSON annotation files.

    Returns
    -------
    list[dict]
        List of entities.
    """
    logger = create_logger()
    logger.info("Collecting entities.")
    entities_list = []
    json_files = list(texts_path.glob("*.json"))
    logger.success(f"Found {len(json_files)} JSON files successfully.")

    if json_files == []:
        logger.warning(f"No JSON files found in {texts_path}")
    # Handle relative paths if the text file is located in a different directory
    if str(texts_path).startswith("../../"):
        json_files = [Path("../../") / json_file for json_file in json_files]

    for json_file in json_files:
        try:
            with json_file.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse JSON file {json_file.name}: {exc}")
            continue

        for entity in data.get("entities", []):
            # Extract category and text
            category = entity.get("category")
            text = entity.get("text")
            # Skip if either category or text is missing
            if not category or not text:
                continue
            # Create entity dictionnary
            entity_dict = {
                "entity": text.lower(),
                "category": category,
                "json_file": Path(json_file).name,
            }
            entities_list.append(entity_dict)
    logger.success(f"Collected {len(entities_list)} entities.")
    return entities_list


def write_inventory(
    entities_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Write a single TSV file containing all entity counts.

    Parameters
    ----------
    entities_df : pd.DataFrame
        DataFrame containing all entities.
    out_path : Path
        Path where the output TSV file will be written.
    """
    logger.info("Writing entity inventory TSV file.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to TSV
    entities_df.to_csv(out_path, sep="\t", index=False)
    logger.success(f"Saved entity inventory in: {out_path}")


def plot_category_distribution(df: pd.DataFrame) -> None:
    """Plot a bar chart showing the total number of entities per category."""
    total_texts = df["json_file"].nunique()
    summary = df["category"].value_counts().sort_values(ascending=False)
    categories = summary.index.tolist()
    counts = summary.to_numpy()
    colors = [COLORS.get(cat, "#cccccc") for cat in categories]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    bars = ax.bar(x, counts, color=colors, edgecolor="dimgrey")
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title(
        f"Category distribution ({total_texts} texts / {counts.sum():,} entities)",
        fontsize=15,
    )
    ax.set_ylabel("Total count", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    file_path = Path("plots/annotations/entity_distribution.png")
    os.makedirs(file_path.parent, exist_ok=True)
    fig.savefig(file_path, bbox_inches="tight", dpi=200)
    logger.success(f"Saved entity distribution plot in '{file_path}'.")


def plot_entity_distribution_by_category(df: pd.DataFrame) -> None:
    """Plot histograms of entity counts per category from a flat entity DataFrame."""
    categories = sorted(df["category"].unique())
    n_cols = 2
    n_rows = math.ceil(len(categories) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(18, 5 * n_rows), constrained_layout=True
    )
    axes = axes.flatten()

    for i, cat in enumerate(categories):
        ax = axes[i]
        data = df[df["category"] == cat].groupby("json_file").size()
        ax.hist(data, bins=15, color=COLORS.get(cat, "#cccccc"), edgecolor="black")
        ax.set_title(
            f"Category {cat}\nmin: {data.min()} max: {data.max()}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Number of entities", fontsize=11)
        ax.set_ylabel("Number of files", fontsize=11)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Entity distributions by category", fontsize=16, fontweight="bold")
    file_path = Path("plots/annotations/entity_distribution_by_category.png")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, bbox_inches="tight", dpi=200)
    logger.success(f"Saved entities distribution by category plot in '{file_path}'.")


@click.command()
@click.option(
    "--annotations-path",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    required=True,
    help="Folder containing the list of JSON files with annotations.",
)
@click.option(
    "--out-path",
    type=click.Path(file_okay=True, path_type=Path),
    required=True,
    help="Path of the TSV file with the entities.",
)
def run_cli(
    annotations_path: Path,
    out_path: Path,
) -> None:
    """
    Run the QC entity inventory process.

    Parameters
    ----------
    annotations_path : Path
        Folder containing the JSON files with annotations.
    out_path : Path
        Path of the TSV file with the entities.
    """
    logger = create_logger()
    logger.info("Starting entity inventory.")
    entities = collect_entities(annotations_path)
    # Create the dataframe
    df_entities = pd.DataFrame(entities)
    write_inventory(df_entities, out_path)
    plot_category_distribution(df_entities)
    plot_entity_distribution_by_category(df_entities)
    logger.success("Entity inventory completed successfully!")


if __name__ == "__main__":
    run_cli()
