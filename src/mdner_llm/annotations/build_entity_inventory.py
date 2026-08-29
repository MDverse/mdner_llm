"""Build a vocabulary of named entities from JSON annotation files.

This script scans a directory of JSON annotation files, aggregates named
entities by class, normalizes entity text to lowercase, counts total
occurrences across all files, and generates one vocabulary file per entity class.

Each output file contains:
- A header reporting the number of unique entities for that class.
- One normalized entity per line with its total occurrence count.
"""

import json
from pathlib import Path

import click
import loguru
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from openai import OpenAI

from mdner_llm.annotations.colors import COLORS
from mdner_llm.common import load_api_key
from mdner_llm.logger import create_logger


def collect_entities(
    texts_path: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> list[dict]:
    """
    Collect normalized entity counts per class from annotation files.

    Returns
    -------
    list[dict]
        List of entities.
    """
    logger.info("Collecting entities.")
    texts_dict = {}
    entities_list = []
    # Scan the directory for JSON files.
    json_files = list(texts_path.glob("*.json"))
    logger.success(f"Found {len(json_files)} JSON files successfully.")
    # Handle relative paths.
    if str(texts_path).startswith("../../"):
        json_files = [Path("../../") / json_file for json_file in json_files]
    # Process each JSON file.
    for json_file in json_files:
        try:
            with json_file.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse JSON file {json_file.name}: {exc}")
            continue
        # Extract raw text for similarity analysis.
        texts_dict[json_file.name] = data.get("raw_text", "")
        # Extract entities and normalize them.
        for entity in data.get("entities", []):
            # Extract category and text
            category = entity.get("category")
            text = entity.get("text")
            # Create entity dictionnary
            entity_dict = {
                "entity": text.lower(),
                "category": category,
                "json_file": Path(json_file).name,
            }
            entities_list.append(entity_dict)
    logger.success(f"Collected {len(entities_list)} entities.")
    return entities_list, texts_dict


def plot_category_distribution(
    df: pd.DataFrame, logger: "loguru.Logger" = loguru.logger
) -> None:
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
    # Plot distribution with total counts and non-redundant counts.
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
    # Annotate bars with total and non-redundant counts
    for bar, total, unique in zip(bars, counts, unique_counts, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            total,
            f"{total:.0f}\n({unique:.0f})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    # Set title, labels, and legend.
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
    # Save the plot.
    file_path = Path("plots/annotations/entity_distribution.png")
    fig.savefig(file_path, bbox_inches="tight", dpi=200)
    logger.success(f"Saved entity distribution plot in '{file_path}'.")


def plot_entity_distribution_by_category(
    df: pd.DataFrame, logger: "loguru.Logger" = loguru.logger
) -> None:
    """Plot histograms of entity counts per category from a flat entity DataFrame."""
    # Normalize entity text to lower case and strip whitespace.
    df_norm = df.assign(entity_norm=df["entity"].str.lower().str.strip())
    # Define setup tuples for total and unique entity distributions.
    configs = [
        (
            df.groupby(["category", "json_file"]).size(),
            "Entity distributions by category",
            "plots/annotations/entity_distribution_by_category.png",
            None,
        ),
        (
            df_norm.groupby(["category", "json_file"])["entity_norm"].nunique(),
            "Non-redundant entity distributions by category",
            "plots/annotations/unique_entity_distribution_by_category.png",
            "//",
        ),
    ]
    # Generate and export both histogram figures.
    for counts, suptitle, out_file, hatch_pattern in configs:
        # Initialize grid layout for subplots.
        fig, axes = plt.subplots(3, 2, figsize=(18, 15), constrained_layout=True)
        # Plot distribution per category.
        for axis, (category, data) in zip(
            axes.flat, counts.groupby(level="category"), strict=False
        ):
            # Compute integer bins based on maximum entity count.
            max_val = int(data.max())
            bins = np.arange(0, max_val + 2) - 0.5
            _, _, bars = axis.hist(
                data,
                bins=bins,
                color=COLORS.get(category, "#cccccc"),
                edgecolor="black",
                hatch=hatch_pattern,
            )
            # Add frequency labels on top of bars.
            labels = [int(v) if v > 0 else "" for v in bars.datavalues]
            axis.bar_label(bars, labels=labels, padding=2)
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set(
                title=f"Category {category}\nmin: {data.min()} max: {max_val}",
                xlabel="Number of entities",
                ylabel="Number of files",
                xlim=(-0.5, max_val + 0.5),
                xticks=range(max_val + 1),
            )
        # Configure global figure title.
        fig.suptitle(suptitle, fontsize=16, fontweight="bold")
        # Save figure to disk and close plot instance.
        out_path = Path(out_file)
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        logger.success(f"Saved plot in '{out_path}'.")


def plot_categories_per_text_distribution(
    df: pd.DataFrame, logger: "loguru.Logger" = loguru.logger
) -> None:
    """Plot distribution of unique entity categories per text."""
    # Count the number of unique categories per text.
    cat_per_file = df.groupby("json_file")["category"].nunique()
    n_texts, n_cats = len(cat_per_file), df["category"].nunique()
    # Outlier alerts for texts with low category coverage.
    for fn, count in cat_per_file[cat_per_file <= 1].items():
        logger.warning(f"Text with low category coverage ({count}) in '{fn}'")
    # Plot histogram of unique categories per text.
    fig, ax = plt.subplots(figsize=(10, 5))
    _, _, bars = ax.hist(
        cat_per_file,
        bins=np.arange(1, n_cats + 2) - 0.5,
        rwidth=0.8,
        color="#B873C9",
        edgecolor="black",
    )
    ax.bar_label(bars, padding=2)
    ax.set(
        title=f"Category diversity distribution ({n_texts} texts / "
        f"{n_cats} total categories)",
        xlabel="Unique categories per text",
        ylabel="Total count",
        xticks=range(1, n_cats + 1),
    )
    # Save the plot.
    out = Path("plots/annotations/categories_per_text_distribution.png")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    logger.success(f"Saved categories per text distribution plot in '{out}'.")


def plot_text_length_distribution(
    texts_dict: dict[str, str], logger: "loguru.Logger" = loguru.logger
) -> None:
    """Plot histogram of text word lengths."""
    # Compute word counts for all texts
    counts = pd.Series(
        {
            doc_name: len(text_content.split())
            for doc_name, text_content in texts_dict.items()
        }
    )
    # Log warning for length outliers.
    for doc_name, word_count in counts[(counts < 10) | (counts > 500)].items():
        logger.warning(f"Outlier text length in '{doc_name}': {word_count} words")
    # Initialize and populate histogram plot.
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(counts, bins=20, color="#4C6EF5", edgecolor="black")
    axis.set(
        title=f"Text length distribution in words ({len(counts)} texts / "
        f"{counts.sum():,} words)",
        xlabel="Word count",
        ylabel="Total count",
    )
    # Add summary statistics box.
    stats = f"Median: {int(counts.median())}\nMin: {counts.min()}\nMax: {counts.max()}"
    axis.text(
        0.85,
        0.95,
        stats,
        transform=axis.transAxes,
        va="top",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "whitesmoke",
            "edgecolor": "lightgrey",
            "alpha": 0.8,
        },
    )
    # Save the plot.
    output_path = Path("plots/annotations/text_length_distribution.png")
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    logger.success(f"Saved text length distribution plot in '{output_path}'.")


def plot_text_similarity_distribution(
    texts_dict: dict[str, str], logger: "loguru.Logger" = loguru.logger
) -> None:
    """Compute embeddings via OpenRouter and plot pairwise cosine similarities."""
    # Fetch normalized embeddings via API.
    filenames, text_list = list(texts_dict.keys()), list(texts_dict.values())
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=load_api_key("OPENROUTER_API_KEY"),
    )
    response = client.embeddings.create(
        model="openai/text-embedding-3-large", input=text_list
    )
    raw_embeds = np.array([item.embedding for item in response.data])
    norm_embeds = raw_embeds / np.linalg.norm(raw_embeds, axis=1, keepdims=True)
    # Extract pairwise upper-triangle cosine similarities.
    row_indices, col_indices = np.triu_indices(len(text_list), k=1)
    similarities = (norm_embeds @ norm_embeds.T)[row_indices, col_indices]
    # Identify representative pair indices for extremes and median.
    min_pair_idx = int(np.argmin(similarities))
    max_pair_idx = int(np.argmax(similarities))
    med_pair_idx = int(np.abs(similarities - np.median(similarities)).argmin())
    # Log representative similarity examples.
    pair_reports = (
        ("Most dissimilar", min_pair_idx),
        ("Median similarity", med_pair_idx),
        ("Most similar", max_pair_idx),
    )
    for label, pair_index in pair_reports:
        first_doc = filenames[row_indices[pair_index]]
        second_doc = filenames[col_indices[pair_index]]
        score = similarities[pair_index]
        logger.info(f"{label} pair ({score:.3f}): '{first_doc}' and '{second_doc}'")
    # Log alerts for near-duplicate texts.
    for row_idx, col_idx, sim_score in zip(
        row_indices, col_indices, similarities, strict=False
    ):
        if sim_score >= 0.98:
            logger.warning(
                f"Near-duplicate text pair ({sim_score:.3f}): '{filenames[row_idx]}' "
                f"and '{filenames[col_idx]}'"
            )
    # Initialize and populate histogram.
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(similarities, bins=25, color="#7048E8", edgecolor="black")
    axis.set(
        xlim=(0, 1),
        title=f"Text similarity distribution ({len(text_list)} texts / "
        f"{len(similarities):,} pairs)",
        xlabel="Cosine similarity (0 = dissimilar, 1 = identical)",
        ylabel="Total count",
    )
    # Add summary statistics text box.
    stats = f"Median: {np.median(similarities):.2f}\nMin: {similarities.min():.2f}"
    stats += f"\nMax: {similarities.max():.2f}"
    axis.text(
        0.85,
        0.95,
        stats,
        transform=axis.transAxes,
        va="top",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "whitesmoke",
            "edgecolor": "lightgrey",
            "alpha": 0.8,
        },
    )
    # Save the plot.
    output_path = Path("plots/annotations/text_similarity_distribution.png")
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    logger.success(f"Saved text similarity distribution plot in '{output_path}'.")


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
    """Run the QC entity inventory process."""
    logger = create_logger()
    logger.info("Starting entity inventory.")
    entities, texts_dict = collect_entities(annotations_path, logger=logger)
    # Create the dataframe.
    df_entities = pd.DataFrame(entities)
    # Write to TSV.
    df_entities.to_csv(out_path, sep="\t", index=False)
    logger.success(f"Saved entity inventory in: {out_path}")
    # Generate plots.
    Path("plots/annotations/").mkdir(parents=True, exist_ok=True)
    plot_category_distribution(df_entities, logger=logger)
    plot_entity_distribution_by_category(df_entities, logger=logger)
    plot_categories_per_text_distribution(df_entities, logger=logger)
    plot_text_length_distribution(texts_dict, logger=logger)
    # plot_text_similarity_distribution(texts_dict, logger=logger)
    logger.success("Entity inventory completed successfully!")


if __name__ == "__main__":
    run_cli()
