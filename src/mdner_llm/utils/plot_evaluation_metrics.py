"""
Generate bar plots from a GLiNER evaluation Excel file.

This script reads an Excel file containing evaluation metrics per entity class
and generates:
1. A bar plot for the number of annotations per entity class.
2. One bar plot per metric (precision, recall, f1, fbeta_0.5).

Plots are saved in: plots/gliner/<model_name>/
"""

import operator
import os
from pathlib import Path

import click
import loguru
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
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
    out_dir = Path("plots/evaluation/gliner") / f"{sanitize_filename(model_name)}"
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
    values: list[int] = df_plot["nb_of_texts_with_label"].tolist()
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
        "Number of texts with this entity label",
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
        f"Number of texts per entity class (test samples: {total_annotations})",
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
    logger.success(
        f"Saved number of texts with this entity label per class plot to {output_path}"
    )
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
    # Reorder the DataFrame to have "OVERALL" at the end
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


def plot_and_save_overall_metrics(
    input_file: Path, model_name: str, logger: "loguru.Logger" = loguru.logger
) -> None:
    """Plot metrics from GLiNER evaluation Excel file."""
    logger.info("Plotting metrics from GLiNER evaluation...")
    # Load dataframe
    df = load_data(input_file)
    # Create output directory
    out_dir = get_output_dir(model_name)
    # Get total number of annotations from the "OVERALL" row
    overall = df[df["label"] == "OVERALL"].iloc[0]
    total_annotations = int(overall["nb_of_texts_with_label"])
    # Plot and save number of annotations per entity class
    plot_number_of_annotations_per_class(df, out_dir, total_annotations, logger)
    # Plot and save metrics per entity class
    metrics = [col for col in df.columns if col.endswith("_score")]
    for metric in metrics:
        plot_metric(df, metric, out_dir, model_name, total_annotations, logger)


def darken_color(color: str, factor: float = 0.3) -> str:
    """
    Darken a matplotlib color.

    Parameters
    ----------
    color:
        Base color (hex or named).
    factor:
        Multiplicative factor (<1 darker, >1 lighter).

    Returns
    -------
        Darkened hex color.
    """
    rgb = mcolors.to_rgb(color)
    darkened = tuple(max(0, c * factor) for c in rgb)
    return mcolors.to_hex(darkened)


def make_plot_compare_scores_models_by_framework(
    df: pd.DataFrame, out_dir: Path = Path("../../plots/evaluation/llm")
) -> None:
    """Generate comparison plots per framework from a flattened DataFrame.

    Expected column format: "<framework>_<metric>" + "model_name", "label".
    Example: "rag_accuracy", "rag_f1", "baseline_accuracy", etc.
    """
    # Extract metric columns (exclude metadata columns)
    metric_columns = [col for col in df.columns if col not in {"model_name", "label"}]
    # Extract frameworks and metrics from flattened names
    frameworks = sorted({col.split("_")[0] for col in metric_columns})
    metrics = sorted({col.split("_", 1)[1] for col in metric_columns})
    # Get labels and models
    labels = df["label"]
    model_names = df["model_name"].unique()
    labels_unique = labels.unique()
    group_spacing = 1.2
    labels_index = np.arange(len(labels_unique)) * group_spacing
    # Get colors for each label (same color across models)
    colors = [COLORS.get(label, "#272727") for label in labels_unique]
    edge_colors = [darken_color(c, 0.6) for c in colors]

    for framework in frameworks:
        # Compute total annotations
        count_col = f"{framework}_Number of Texts with Label"
        if count_col in df.columns:
            total_annotations_series = df[df["label"] == "OVERALL"][count_col]
            total_annotations = (
                int(total_annotations_series.iloc[0])
                if not total_annotations_series.empty
                else 0
            )
        else:
            total_annotations = 0
        for metric in metrics:
            # Check if the column for this framework and metric exists
            col_name = f"{framework}_{metric}"
            if col_name not in df.columns:
                continue
            # Compute ranking ONCE per metric
            top_models_per_label = {}
            for label in labels_unique:
                scores = []
                for m in model_names:
                    df_model = df[df["model_name"] == m]
                    val = df_model[df_model["label"] == label][col_name]
                    score = val.iloc[0] if not val.empty else 0
                    scores.append((m, score))

                scores_sorted = sorted(scores, key=operator.itemgetter(1), reverse=True)
                top_models_per_label[label] = {m for m, _ in scores_sorted[:3]}
            # Create the plot
            fig, ax = plt.subplots(figsize=(20, 15))
            bar_height = (1.2 - 0.2) / len(model_names)
            # For each model, plot its bars for this metric and framework
            for model_idx, model_name in enumerate(model_names):
                # Get values for this model and metric
                df_model = df[df["model_name"] == model_name]
                values = []
                for label in labels_unique:
                    val = df_model[df_model["label"] == label][col_name]
                    values.append(val.iloc[0] if not val.empty else 0)
                shift = bar_height * (len(model_names) - 1) / 2
                positions = labels_index - shift + model_idx * bar_height

                # Plot bars for this model
                ax.barh(
                    positions,
                    values,
                    height=bar_height,
                    edgecolor=edge_colors,
                    color=colors,
                )

                for label, y, x in zip(labels_unique, positions, values, strict=False):
                    is_top = model_name in top_models_per_label[label]
                    weight = "bold" if is_top else "normal"
                    # Model name (outside bar)
                    ax.text(
                        x + 0.01,
                        y,
                        model_name,
                        va="center",
                        ha="left",
                        rotation=0,
                        color="dimgrey",
                        fontweight=weight,
                        fontsize=8,
                    )
                    # Score (inside bar)
                    ax.text(
                        x - 0.02,
                        y,
                        f"{x:.2f}",
                        va="center",
                        ha="right",
                        fontsize=8,
                        color="dimgrey",
                        fontweight=weight,
                    )
            # Formatting
            ax.set_yticks(labels_index, labels_unique, weight="bold", color="#3D3D3D")
            ax.set_xlim(0, 1)
            ax.set_xlabel(metric)
            # Title
            fig.suptitle(
                f"{metric} comparison across LLM models "
                f"(test samples: {total_annotations})",
                fontsize=18,
                fontweight="bold",
                color="#3D3D3D",
                y=0.92,
            )
            ax.set_title(
                f"Framework: {framework}",
                fontsize=12,
                color="gray",
                loc="left",
                pad=2,
            )
            # Clean spines
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            # Saving the plot into png file
            fw_dir = out_dir / framework
            os.makedirs(fw_dir, exist_ok=True)
            plt.savefig(fw_dir / f"{metric}_comparaison_llm_models.png")

            # Show the plot
            plt.show()
            plt.close()


def make_plot_compare_scores_gliner_models(
    df: pd.DataFrame, out_dir: Path = Path("../../plots/evaluation/gliner")
) -> None:
    """
    Plot multiple *_score metrics for multiple models using horizontal bars.

    Each label keeps the same color across models.
    Each model is displayed as a separate bar per label.
    Model names and values are displayed next to bars.
    """
    # Extract score columns (those ending with "_score")
    scores = [col for col in df.columns if col.endswith("_score")]
    # Get unique model names and labels
    model_names = df["model_name"].unique()
    labels = df["label"].unique()
    labels_index = np.arange(len(labels))
    # Get colors for each label (same color across models)
    colors = [COLORS.get(label, "#272727") for label in labels]
    edge_colors = [darken_color(c, 0.6) for c in colors]
    # Get total number of annotations from the "OVERALL"
    # it contains the total number of test sample
    # by counting the number of unique annotation texts
    total_annotations = int(
        df[df["label"] == "OVERALL"].iloc[0]["nb_of_texts_with_label"]
    )
    # Set global font size for better readability
    plt.rcParams.update({"font.size": 14})
    # For each score, create a separate plot comparing all models
    for score in scores:
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_height = (1.0 - 0.15) / len(model_names)
        score_name = score.replace("_score", "").capitalize()
        for model_index, model_name in enumerate(model_names):
            # Calculate bar positions for this model (centered around the label index)
            shift = bar_height * (len(model_names) - 1) / 2
            positions = labels_index - shift + model_index * bar_height
            # Get values for this model and score
            values = np.array(df[df["model_name"] == model_name][score])
            # Plot bars for this model
            ax.barh(
                positions,
                values,
                height=bar_height,
                edgecolor=edge_colors,
                color=colors,
            )
            # Add text annotations
            for y, x in zip(positions, values, strict=False):
                simplified_model_name = model_name.replace("gliner2-", "").replace(
                    "_", " "
                )
                # Model name (outside bar)
                ax.text(
                    x + 0.01,
                    y,
                    simplified_model_name,
                    va="center",
                    ha="left",
                    rotation=0,
                    color="dimgrey",
                    fontsize=10,
                    fontweight="bold",
                )

                # Score (inside bar)
                ax.text(
                    x - 0.03,
                    y,
                    f"{x:.2f}",
                    va="center",
                    ha="right",
                    fontsize=10,
                    color="dimgrey",
                    fontweight="bold",
                )
        # Formatting
        ax.set_yticks(labels_index, labels, weight="bold", color="#3D3D3D")
        ax.set_xlim(0, 1)
        ax.set_xlabel(score_name, weight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="lightgrey")
        # Title
        fig.suptitle(
            f"{score_name} comparison across GLiNER models"
            f" (test samples: {total_annotations})",
            fontsize=18,
            fontweight="bold",
            color="#3D3D3D",
        )
        # Clean spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # Saving the plot into png file
        output_path = out_dir / f"{score}_comparaison_gliner_models.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        # Show the plot
        plt.show()
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
def run_main_from_cli(input_file: Path, model_name: str) -> None:
    """Run the main function from the command line."""
    # Create logger
    logger = create_logger()
    plot_and_save_overall_metrics(input_file, model_name, logger)


if __name__ == "__main__":
    run_main_from_cli()
