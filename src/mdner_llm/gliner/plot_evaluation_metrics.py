"""
Generate bar plots from a GLiNER evaluation Excel file.

This script reads an Excel file containing evaluation metrics per entity class
and generates:
1. A bar plot for the number of annotations per entity class.
2. One bar plot per metric (precision, recall, f1, fbeta_0.5).

Plots are saved in: plots/gliner/<model_name>/

Usage:
------
python script.py --input-file results.xlsx --model-name "GLiNER2 Base"

Example:
--------
python script.py --input-file metrics.xlsx --model-name "GLiNER2 Base (205M)"
"""

from pathlib import Path

import click
import loguru
import matplotlib.pyplot as plt
import pandas as pd

from mdner_llm.core.logger import create_logger
from mdner_llm.utils.common import sanitize_filename
from mdner_llm.utils.visualize_annotations import COLORS


def load_data(input_file: Path) -> pd.DataFrame:
    """Load the Excel file as a DataFrame.

    Parameters
    ----------
    input_file : Path
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the evaluation metrics.
    """
    return pd.read_excel(input_file)


def get_output_dir(model_name: str) -> Path:
    """Create output directory.

    Parameters
    ----------
    model_name : str
        Model name.

    Returns
    -------
    Path
        Output directory path.
    """
    out_dir = Path("plots") / "gliner" / f"{sanitize_filename(model_name)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_number_of_annotations_per_class(
    df: pd.DataFrame,
    out_dir: Path,
    total_annotations: int,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Plot number of annotations per entity class."""
    df_plot = df[df["label"] != "OVERALL"].copy()
    # Extract labels and values for the plot
    labels: list[str] = df_plot["label"].tolist()
    values: list[int] = df_plot["nb_annotations"].tolist()
    colors = [COLORS.get(label, "#cccccc") for label in labels]
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor="none")
    # Set x-axis limit slightly above the max value for better visualization
    max_val = max(values) if values else 0
    ax.set_xlim(0, max_val * 1.15)
    # Values inside bars
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_width() - (max_val * 0.02),
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
            color="black",
        )
    # Grid (vertical only)
    ax.xaxis.grid(visible=True, linestyle="-", linewidth=1, alpha=0.3)
    ax.yaxis.grid(visible=False)
    # Axes formatting
    ax.set_xlabel(
        "Number of annotations",
        fontsize=12,
        color="#3D3D3D",
        fontweight="bold",
    )
    ax.set_ylabel(
        "Entity Class",
        fontsize=12,
        color="#3D3D3D",
        fontweight="bold",
    )
    # Title
    fig.suptitle(
        f"Number of annotations per entity class (test samples: {total_annotations})",
        fontsize=16,
        fontweight="bold",
        color="#3D3D3D",
    )
    # Clean spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # Save the plot
    output_path = out_dir / "annotations_per_class.png"
    plt.savefig(output_path, dpi=300)
    logger.success(f"Saved number of annotations per class plot to {output_path}")
    plt.close()


def plot_metric(
    df_plot: pd.DataFrame,
    metric: str,
    out_dir: Path,
    model_name: str,
    total_annotations: int,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Plot a metric per entity class."""
    # Reorder the DataFrame to have "OVERALL" at the end for better visualization
    df_plot = df_plot.sort_values(
        by="label",
        key=lambda x: x == "OVERALL",
        ascending=False,
    )
    # Extract labels and values for the plot
    labels = df_plot["label"].tolist()
    values = df_plot[metric].tolist()
    # Get colors for each label
    # using a dark color for "OVERALL" and lighter colors for others
    colors = [COLORS.get(label, "#272727") for label in labels]
    # Format metric name for the title and x-axis label
    metric_name = metric.replace("_score", "").capitalize()
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor="none")
    # Labels inside bars
    for bar, val, label in zip(bars, values, labels, strict=False):
        ax.text(
            val - 0.02,
            bar.get_y() + bar.get_height() / 2,
            round(val, 2),
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
            # Text color: white for overall label (dark bar),
            # black for others (lighter bars)
            color="white" if label == "OVERALL" else "black",
        )
    # Grid (vertical only)
    ax.xaxis.grid(visible=True, linestyle="-", linewidth=1, alpha=0.3)
    ax.yaxis.grid(visible=False)
    # Axes formatting
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric_name, fontsize=12, color="#3D3D3D", fontweight="bold")
    ax.set_ylabel("Entity Class", fontsize=12, color="#3D3D3D", fontweight="bold")
    # Custom x-ticks as percentages
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    # Title with model name and total annotations
    fig.suptitle(
        f"{metric_name} score per entity class (test samples: {total_annotations})",
        fontsize=16,
        fontweight="bold",
        color="#3D3D3D",
    )
    # Subtitle with model name
    ax.set_title(
        f"Model: {model_name}",
        fontsize=12,
        color="gray",
        loc="left",
        pad=10,
    )
    # Clean spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # Save the plot
    plt.savefig(out_dir / f"{metric}.png", dpi=300)
    logger.success(f"Saved {metric} plot to {out_dir / f'{metric}.png'}")
    plt.close()


@click.command()
@click.option(
    "--input-file",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to input Excel file.",
)
@click.option(
    "--model-name",
    type=str,
    required=True,
    help="GLiNER model name (used for plot titles and output directory).",
)
def main(input_file: Path, model_name: str) -> None:
    """Plot metrics from GLiNER evaluation Excel file."""
    logger = create_logger()
    logger.info("Plotting metrics from GLiNER evaluation...")
    # Load dataframe
    df = load_data(input_file)
    # Create output directory
    out_dir = get_output_dir(model_name)
    # Get total number of annotations from the "OVERALL" row
    overall = df[df["label"] == "OVERALL"].iloc[0]
    total_annotations = int(overall["nb_annotations"])
    # Plot and save number of annotations per entity class
    plot_number_of_annotations_per_class(df, out_dir, total_annotations, logger)
    # Plot and save metrics per entity class
    metrics = [col for col in df.columns if col.endswith("_score")]
    for metric in metrics:
        plot_metric(df, metric, out_dir, model_name, total_annotations, logger)


if __name__ == "__main__":
    main()
