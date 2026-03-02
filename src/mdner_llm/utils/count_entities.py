"""Count the number of entities by class for each annotation.

This script processes all JSON annotation files in a specified directory
and counts the number of entities for each class defined in the annotation file.
The script outputs a TSV file containing the filename, annotated text length
and the number of entities by class.
"""

import json
import math
from pathlib import Path

import click
import loguru
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import watermark
from loguru import logger

from mdner_llm.core.logger import create_logger

CLASSES = ["TEMP", "SOFTNAME", "SOFTVERS", "STIME", "MOL", "FFM"]


def display_watermark(logger) -> None:
    """Display watermark information in the logs."""
    wm = watermark.watermark(
        watermark=True,
        iversions=True,
        python=True,
        globals_=globals(),
    )
    print(type(logger))
    logger.info(f"Environment information:\n{wm}")


def list_json_files(
    directory: Path, logger: "loguru.Logger" = loguru.logger
) -> list[Path]:
    """
    Retrieve all JSON files from a given directory.

    Parameters
    ----------
        directory (Path): The path to the directory containing JSON files.

    Returns
    -------
    files: List[Path]
        A list of JSON file paths.
    """
    files = list(directory.rglob("*.json"))
    logger.success(f"Found {len(files)} JSON annotation files.")
    return files


def collect_entity_counts(
    json_files: list[Path], classes: list[str]
) -> list[dict[str, int]]:
    """
    Collect entity counts per class from a list of JSON annotation files.

    Parameters
    ----------
    json_files : list[Path]
        List of paths to JSON annotation files.
    classes : list[str]
        List of entity classes to count.

    Returns
    -------
    list[dict[str, int]]
        List of dictionaries containing entity counts,
        filename, and text length for each file.
    """
    all_counts = []

    for filepath in json_files:
        json_data = load_json(filepath)
        if not json_data or "raw_text" not in json_data:
            logger.warning(
                f"File {filepath} is missing 'raw_text' or failed to load. Skipping."
            )
            continue

        counts = count_entities_per_class(json_data, classes)
        counts["filename"] = filepath.name
        counts["text_length"] = len(json_data["raw_text"])
        all_counts.append(counts)

    return all_counts


def load_json(json_file_path: Path) -> dict:
    """
    Load a JSON file and return its content as a dictionary.

    Parameters
    ----------
        json_file_path (Path): The full path to the JSON file.

    Returns
    -------
        Dict: Parsed JSON data.
    """
    data = {}
    try:
        with open(json_file_path, encoding="utf-8") as json_file:
            data = json.load(json_file)
    except ValueError as e:
        logger.error(f"Failed to load {json_file_path}: {e}")
    except OSError as e:
        logger.error(f"Cannot open {json_file_path}: {e}")
    return data


def count_entities_per_class(data: dict, classes: list[str]) -> dict:
    """
    Count the number of entities per class in a JSON annotation.

    Parameters
    ----------
        data (Dict): The JSON data loaded from an annotation file.
        classes (List[str]): List of entity classes to count.

    Returns
    -------
        Dict: Updated dictionary with the count of entities per class.
    """
    # Create empty dictionary.
    record = dict.fromkeys(classes, 0)
    # Count entities per class.
    for entity in data["entities"]:
        record[entity["label"]] += 1
    return record


def aggregate(counts_list: list[dict], classes: list[str]) -> pd.DataFrame:
    """
    Aggregate a list of entity count dictionaries into a DataFrame.

    Parameters
    ----------
    counts_list: List[Dict]
        A list of dictionaries with entity counts.
    classes: List[str]
        List of entity classes.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the aggregated counts.
    """
    df = pd.DataFrame(counts_list)
    columns = {}
    for cls in classes:
        columns[cls] = f"{cls}_nb"
    df = df.rename(columns=columns)
    df = df[["filename", "text_length", *list(columns.values())]]
    return df


def display_stats(df: pd.DataFrame, classes: list[str]) -> None:
    """
    Display statistics of entity counts per class.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame containing the entity counts.
    classes: List[str]
        List of entity classes.
    """
    total_entities = df[[f"{cls}_nb" for cls in classes]].sum()
    logger.info(f"Total number of entities: {total_entities.sum()}")
    for cls in classes:
        class_total = df[f"{cls}_nb"].sum()
        logger.info(f"Number of entities for class '{cls}': {class_total}")


def export_to_tsv(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Export the entity counts into a TSV file.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame containing the aggregated entity counts.
    output_dir: Path
        Directory where the TSV file will be saved.
    """
    # Ensure output directory exists.
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    tsv_file_path = output_dir / "entities_count.tsv"
    try:
        df.to_csv(tsv_file_path, sep="\t", index=False)
    except OSError as e:
        logger.error(f"Failed to write TSV file: {e}")
    else:
        logger.success(f"Count results saved in '{tsv_file_path}'")


def plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot a bar chart showing the total number of entities per class across all files.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns ending with ``_nb`` representing
        the counts of each entity class.
    output_dir : Path
        Directory where the plot image will be saved.
    """
    total_entities = len(df)
    cols = [col for col in df.columns if col.endswith("_nb")]
    summary = df[cols].sum().sort_values(ascending=False)

    classes = [col.replace("_nb", "") for col in summary.index]
    counts = summary.to_numpy()

    cmap = mpl.colormaps.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(classes)))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    bars = ax.bar(x, counts, color=colors)
    # Add labels
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
        f"Classe distribution ({total_entities} texts / {np.sum(counts):,} entities)",
        fontsize=15,
    )
    ax.set_xlabel("Entity class", fontsize=13)
    ax.set_ylabel("Total count", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    filename = output_dir / "entity_class_distribution.png"
    fig.savefig(filename, bbox_inches="tight", dpi=200)
    logger.success(f"Class distribution plot saved in '{filename}'")


def plot_entity_distribution_by_class(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot histograms of entity counts per class.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns ending with '_nb' = counts of each entity class.
    """
    cols = [col for col in df.columns if col.endswith("_nb")]
    n_classes = len(cols)

    cmap = mpl.colormaps.get_cmap("Set1")
    colors = cmap(np.linspace(0, 1, n_classes))

    n_cols = 2
    n_rows = math.ceil(n_classes / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(18, 5 * n_rows), constrained_layout=True
    )
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col]
        max_val = int(data.max())
        # Define bins, tics and labels.
        max_bins = 15
        bins = np.arange(0, max_bins + 1)
        labels = [str(label) for label in np.arange(0, max_bins)]
        # Add last boundary depending on data beyond the fixed boundary.
        if max_val <= max_bins:
            bins = np.append(bins, max_bins + 1)
            labels = [*labels, f"{max_bins}"]
        else:
            bins = np.append(bins, max_val)
            labels = [*labels, f"{max_bins}+"]
        ticks = np.arange(0, len(bins) - 1) + 0.5
        # Compute histogram values first.
        heights, _ = np.histogram(data, bins=bins)
        # Plot historam.
        ax.bar(
            x=ticks,
            height=heights,
            color=colors[i],
            edgecolor="black",
        )
        # Add appropriate ticks.
        ax.set_xticks(ticks=ticks, labels=labels)
        title = f"Class {col.replace('_nb', '')}"
        title += f"\nmin: {int(data.min())} max: {int(data.max())}"
        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Number of entities", fontsize=11)
        ax.set_ylabel("Number of files", fontsize=11)
    # Remove empty subplots.
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Entity distributions by class", fontsize=16, fontweight="bold")
    # Save figure.
    filename = output_dir / "entity_distribution_by_class.png"
    fig.savefig(filename, bbox_inches="tight", dpi=200)
    logger.success(f"Entity distribution by class plot saved as '{filename}'")


def compute_entity_counts_df(
    annotations_dir: Path, classes: list, logger: "loguru.Logger" = loguru.logger
) -> pd.DataFrame:
    """
    Compute a DataFrame of entity counts per class from all JSON files in a directory.

    Parameters
    ----------
    annotations_dir : Path
        Path to the directory containing JSON annotation files.
    classes : list
        List of entity classes to count.

    Returns
    -------
    pd.DataFrame
        DataFrame with counts per class, filename, and text length.
    """
    # List JSON files
    json_files = list_json_files(annotations_dir, logger)
    if not json_files:
        logger.warning(f"No JSON files found in {annotations_dir}")
        return pd.DataFrame()
    # Collect entity counts for each file
    all_counts = collect_entity_counts(json_files, classes)
    # Aggregate results into a DataFrame
    counts_df = aggregate(all_counts, classes)
    return counts_df


def main(annotations_dir: Path, results_dir: Path) -> None:
    """
    Run entire workflow to count entities per class in JSON annotation files.

    Parameters
    ----------
    annotations_dir : Path
        Directory containing JSON annotation files to process.
    results_dir : Path
        Directory where results will be saved.
    """
    logger = create_logger()
    display_watermark(logger)
    logger.info("Searching for JSON files...")
    json_files = list_json_files(annotations_dir)
    logger.info("Counting entities...")
    all_counts = collect_entity_counts(json_files, CLASSES)
    # Aggregate results.
    counts_df = aggregate(all_counts, CLASSES)
    # Display statistics.
    display_stats(counts_df, CLASSES)
    # Export to TSV.
    export_to_tsv(counts_df, results_dir)
    # Plot class distribution.
    plot_class_distribution(counts_df, results_dir)
    # Plot entity distribution per text.
    plot_entity_distribution_by_class(counts_df, results_dir)


@click.command()
@click.option(
    "--annotations-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("annotations/v3"),
    show_default=True,
    help="Directory containing JSON annotation files",
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default="results",
    show_default=True,
    help="Directory to save results",
)
def run_from_cli(annotations_dir: Path, results_dir: Path) -> None:
    """Command line interface to count entities per class in JSON annotations.

    Parameters
    ----------
    annotations_dir : Path
        Directory containing JSON annotation files to process.
    results_dir : Path
        Directory where results will be saved.
    """
    main(annotations_dir, results_dir)


if __name__ == "__main__":
    run_from_cli()
