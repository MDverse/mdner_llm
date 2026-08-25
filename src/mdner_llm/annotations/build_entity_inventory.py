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
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from openai import OpenAI

from mdner_llm.annotations.colors import COLORS
from mdner_llm.common import load_api_key
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
    texts_dict = {}
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

        raw_text = data.get("raw_text", "")
        if raw_text:
            texts_dict[json_file.name] = raw_text

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
    return entities_list, texts_dict


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

    # Non-redundant counts: unique entity text (case-insensitive) per category.
    df_norm = df.assign(entity_norm=df["entity"].str.lower().str.strip())
    unique_counts = (
        df_norm.groupby("category")["entity_norm"]
        .nunique()
        .reindex(categories)
        .to_numpy()
    )

    colors = [COLORS.get(cat, "#cccccc") for cat in categories]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))

    bars = ax.bar(x, counts, color=colors, edgecolor="dimgrey")
    ax.bar(
        x,
        unique_counts,
        color=colors,
        edgecolor="dimgrey",
        hatch="///",
        alpha=0.7,
    )

    for bar, total, unique in zip(bars, counts, unique_counts, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            total,
            f"{total:.0f}\n({unique:.0f})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    total_entities = counts.sum()
    ax.set_title(
        f"Category distribution ({total_texts} texts / {total_entities:,} entities)",
        fontsize=15,
    )
    ax.set_ylabel("Total count", fontsize=13)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.legend(
        handles=[
            mpatches.Patch(
                facecolor="white",
                edgecolor="dimgrey",
                hatch="///",
                label="Non-redundant entities",
            )
        ],
        loc="upper right",
        fontsize=9,
    )
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


def plot_categories_per_text_distribution(df: pd.DataFrame) -> None:
    """Plot distribution of unique entity categories per text."""
    cat_per_file = df.groupby("json_file")["category"].nunique()
    total_texts = df["json_file"].nunique()
    total_categories = df["category"].nunique()
    # Outlier alert for texts covering very few categories
    for filename, count in cat_per_file.items():
        if count <= 1:
            logger.warning(
                f"Text with low category coverage in '{filename}': {count} category"
            )
    counts = cat_per_file.to_numpy()
    fig, axis = plt.subplots(figsize=(10, 5))
    _, _, bars = axis.hist(
        counts,
        bins=np.arange(1, total_categories + 2) - 0.5,
        rwidth=0.8,
        color="#B873C9",
        edgecolor="black",
    )
    axis.bar_label(bars, padding=2)
    axis.set(
        title="Category diversity distribution "
        f"({total_texts} texts / {total_categories} total categories)",
        xlabel="Unique categories per text",
        ylabel="Total count",
    )
    axis.set_xticks(range(1, total_categories + 1))
    # Save the plot
    out_file = Path("plots/annotations/categories_per_text_distribution.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", dpi=200)
    logger.success(f"Saved categories per text distribution plot in '{out_file}'.")


def plot_text_length_distribution(texts_dict: dict[str, str]) -> None:
    """Plot histogram of text word lengths."""
    # Compute word counts for each text
    counts = []
    for name, text in texts_dict.items():
        length = len(text.split())
        counts.append(length)
        # Log a warning for outlier text lengths
        if length < 10 or length > 500:
            logger.warning(f"Outlier text length in '{name}': {length} words")
    # Plot histogram of text lengths
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(counts, bins=20, color="#4C6EF5", edgecolor="black")
    axis.set(
        title="Text length distribution in words "
        f"({len(counts)} texts/ {sum(counts):,} words)",
        xlabel="Word count",
        ylabel="Total count",
    )
    # Add median, min, and max statistics to the plot
    stats = f"Median: {int(np.median(counts))}\nMin: {min(counts)}\nMax: {max(counts)}"
    axis.text(
        0.85,
        0.95,
        stats,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "whitesmoke",
            "edgecolor": "lightgrey",
            "alpha": 0.8,
        },
    )
    # Save the plot
    out_file = Path("plots/annotations/text_length_distribution.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", dpi=200)
    logger.success(f"Saved text length distribution plot in '{out_file}'.")


def plot_text_similarity_distribution(texts_dict: dict[str, str]) -> None:
    """Compute embeddings via OpenRouter and plot pairwise cosine similarities."""
    filenames = list(texts_dict.keys())
    text_list = list(texts_dict.values())
    # Initialize OpenAI client with OpenRouter API key
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=load_api_key("OPENROUTER_API_KEY"),
    )
    # Compute embeddings for all texts using OpenRouter
    response = client.embeddings.create(
        model="openai/text-embedding-3-large",
        input=text_list,
    )
    embeddings = np.array([item.embedding for item in response.data])
    # Normalize the embeddings
    normalized_embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    # Compute pairwise cosine similarity matrix
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    row_indices, col_indices = np.triu_indices(len(text_list), k=1)
    pair_similarities = similarity_matrix[row_indices, col_indices]

    # Outlier alerts for identical/near-duplicate texts
    for r_idx, c_idx, sim in zip(
        row_indices, col_indices, pair_similarities, strict=False
    ):
        if sim >= 0.98:
            logger.warning(
                f"Near-duplicate text pair ({sim:.3f}): "
                f"'{filenames[r_idx]}' and '{filenames[c_idx]}'"
            )
    # Plot histogram of pairwise cosine similarities
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(
        pair_similarities,
        bins=25,
        color="#7048E8",
        edgecolor="black",
    )
    axis.set_xlim(0, 1)
    axis.set(
        title=f"Text similarity distribution ({len(text_list)} texts / "
        f"{len(pair_similarities):,} pairs)",
        xlabel="Cosine similarity (0 = dissimilar, 1 = identical)",
        ylabel="Total count",
    )
    # Add median, min, and max statistics to the plot
    stats = f"Median: {np.median(pair_similarities):.2f}\n"
    stats += f"Min: {pair_similarities.min():.2f}\nMax: {pair_similarities.max():.2f}"
    axis.text(
        0.85,
        0.95,
        stats,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "whitesmoke",
            "edgecolor": "lightgrey",
            "alpha": 0.8,
        },
    )
    # Save the plot
    out_file = Path("plots/annotations/text_similarity_distribution.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", dpi=200)
    logger.success(f"Saved text similarity distribution plot in '{out_file}'.")


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
    entities, texts_dict = collect_entities(annotations_path)
    # Create the dataframe
    df_entities = pd.DataFrame(entities)
    write_inventory(df_entities, out_path)
    # Generate plots
    plot_category_distribution(df_entities)
    plot_entity_distribution_by_category(df_entities)
    plot_categories_per_text_distribution(df_entities)
    plot_text_length_distribution(texts_dict)
    plot_text_similarity_distribution(texts_dict)
    logger.success("Entity inventory completed successfully!")


if __name__ == "__main__":
    run_cli()
