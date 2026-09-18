"""Module for characterizing and visualizing ground truth entity annotations."""

from pathlib import Path

import loguru
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from openai import OpenAI

from mdner_llm.common import load_api_key
from mdner_llm.visualization.colors import COLORS


def plot_entity_counts_by_category(
    df: pd.DataFrame,
    output_path: Path | None = None,
    logger: "loguru.Logger" = loguru.logger,
) -> plt.Figure:
    """Plot a bar chart showing the total number of entities per category.

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the category distribution plot.
    """
    total_texts = df["json_file"].nunique()
    summary = df["category"].value_counts().sort_values(ascending=False)
    categories = summary.index.tolist()
    counts = summary.to_numpy()
    # Non-redundant counts: unique entity text (case-insensitive) per category.
    unique_per_cat = df.groupby("category")["entity_normalized"].nunique()
    unique_counts = unique_per_cat.reindex(categories, fill_value=0).to_numpy()
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
    # Annotate bars with total and non-redundant counts.
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
        "Entity counts by category "
        f"({total_texts} texts / {total_entities:,} entities)",
        fontsize=15,
    )
    ax.set_ylabel("Entity count", fontsize=13)
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
    if output_path:
        # Save the plot.
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        logger.success(f"Saved entity distribution plot in '{output_path}'.")
    return fig


def plot_per_doc_entity_counts(
    df: pd.DataFrame,
    *,
    non_redundant: bool = False,
    output_path: Path | None = None,
    logger: "loguru.Logger" = loguru.logger,
) -> plt.Figure:
    """Plot per-document entity count distributions by category.

    Returns
    -------
    plt.Figure
        Figure containing one histogram per category.
    """
    # Count entities per document and category.
    counts = (
        (
            df.groupby(["category", "json_file"])["entity_normalized"].nunique()
            if non_redundant
            else df.groupby(["category", "json_file"]).size()
        )
        .rename("count")
        .reset_index()
    )
    # Add missing document/category combinations with zero counts.
    categories = df["category"].unique()
    files = df["json_file"].unique()
    index = pd.MultiIndex.from_product(
        [categories, files],
        names=["category", "json_file"],
    )
    counts = (
        counts.set_index(["category", "json_file"])
        .reindex(index, fill_value=0)
        .reset_index()
    )
    fig, axes = plt.subplots(3, 2, figsize=(18, 15), constrained_layout=True)
    # Plot one integer-valued histogram per category.
    for axis, (category, group) in zip(
        axes.flat, counts.groupby("category"), strict=False
    ):
        values = group["count"].astype(int)
        max_val, min_val = values.max(), values.min()
        bars = axis.hist(
            values,
            bins=np.arange(max_val + 2) - 0.5,
            color=COLORS.get(category, "#cccccc"),
            edgecolor="black",
            hatch="//" if non_redundant else None,
        )[2]
        # Label non-zero bars and use integer y-axis ticks.
        axis.bar_label(
            bars,
            labels=[int(v) if v else "" for v in bars.datavalues],
            padding=2,
        )
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set(
            xlabel="Number of entities",
            ylabel="Number of documents",
            xlim=(-0.5, max_val + 0.5),
            xticks=range(max_val + 1),
            title=f"Category {category}\nmin: {min_val} max: {max_val}",
        )
        axis.title.set_fontweight("bold")
    # Configure overall figure title.
    fig.suptitle(
        "Per-document non-redundant entity counts"
        if non_redundant
        else "Per-document entity counts",
        fontsize=16,
        fontweight="bold",
    )
    # Save the plot.
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight", dpi=300)
        logger.success(f"Saved entity distribution plot in '{out_p}'.")
    return fig


def plot_categories_per_text_distribution(
    df: pd.DataFrame,
    output_path: Path | None = None,
    logger: "loguru.Logger" = loguru.logger,
) -> plt.Figure:
    """Plot distribution of unique entity categories per text.

    Returns
    -------
    plt.Figure
        Figure containing a stacked bar chart of category diversity distribution.
    """
    # Extract unique text-category pairs,
    # to avoid counting duplicate annotations in the same text.
    unique_pairs = df[["json_file", "category"]].drop_duplicates()
    # Compute the number of unique categories per text file.
    cat_per_file = unique_pairs.groupby("json_file")["category"].nunique()
    n_texts = len(cat_per_file)
    all_categories = sorted(df["category"].unique())
    n_cats = len(all_categories)
    # Warn about texts with low category coverage.
    for fn, count in cat_per_file[cat_per_file <= 1].items():
        logger.warning(f"Text with low category coverage ({count}) in '{fn}'")
    # Assign equal fractional weight to each category present in a text.
    unique_pairs["n_cats"] = unique_pairs["json_file"].map(cat_per_file)
    unique_pairs["weight"] = 1.0 / unique_pairs["n_cats"]
    # Build cross-tabulation of category proportions across distinct category counts.
    pivot = (
        pd.crosstab(
            index=unique_pairs["n_cats"],
            columns=unique_pairs["category"],
            values=unique_pairs["weight"],
            aggfunc="sum",
        )
        .reindex(index=range(1, n_cats + 1), columns=all_categories)
        .fillna(0)
    )
    # Render stacked bar chart showing category compositions per text diversity tier.
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[COLORS.get(color, "#cccccc") for color in pivot.columns],
        edgecolor="black",
        width=0.8,
    )
    # Label each bar with total text count at the top.
    totals = cat_per_file.value_counts().reindex(range(1, n_cats + 1), fill_value=0)
    for idx, total in enumerate(totals):
        if total > 0:
            ax.text(idx, total + 0.2, str(total), ha="center", va="bottom")
    # Configure plot labels and layout attributes.
    ax.set(
        title=f"Category diversity per document ({n_texts} texts / "
        f"{n_cats} total categories)",
        xlabel="Number of distinct categories per document",
        ylabel="Number of documents",
    )
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Categories", loc="upper left")
    # Save the plot.
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=500)
        logger.success(
            f"Saved categories per text distribution plot in '{output_path}'."
        )
    return fig


def plot_text_length_distribution(
    texts_dict: dict[str, str],
    output_path: Path | None = None,
    logger: "loguru.Logger" = loguru.logger,
) -> plt.Figure:
    """Plot histogram of text word lengths.

    Returns
    -------
    plt.Figure
        Figure containing a histogram of text lengths in words.
    """
    # Compute word counts for all texts.
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
        ylabel="Number of documents",
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
    if output_path:
        output_path = Path("plots/annotations/text_length_distribution.png")
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        logger.success(f"Saved text length distribution plot in '{output_path}'.")
    return fig


def plot_text_similarity_distribution(
    texts_dict: dict[str, str],
    output_path: Path | None = None,
    logger: "loguru.Logger" = loguru.logger,
) -> plt.Figure:
    """Compute embeddings via OpenRouter and plot pairwise cosine similarities.

    Returns
    -------
    plt.Figure
        Figure containing a histogram of pairwise cosine similarities between texts.
    """
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
        ylabel="Number of document pairs",
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
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        logger.success(f"Saved text similarity distribution plot in '{output_path}'.")
    return fig
