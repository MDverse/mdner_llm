"""Module to plot performance metrics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from mdner_llm.annotations.visualize_annotations import COLORS


def plot_score(
    df: pd.DataFrame,
    framework: str,
    metric: str,
    categories: list[str],
    models: list[str],
    top_k: int = 3,
) -> None:
    """Generate comparison plots from a long-format DataFrame."""
    # Filter
    df_filtered = df[
        ((df["framework_name"] == framework) | (df["framework_name"].isna()))
        & (df["category"].isin(categories))
    ].copy()
    if models:
        df_filtered = df_filtered[df_filtered["model_name"].isin(models)]
    else:
        models = df_filtered["model_name"].unique()
    # Total annotations for the test set (overall category)
    total_annotations = int(
        df_filtered.loc[
            df_filtered["category"] == "OVERALL",
            "nb_of_texts_with_label",
        ].iloc[0]
    )
    # Total entities for the test set (overall category)
    total_entities = int(
        df_filtered.loc[
            df_filtered["category"] == "OVERALL",
            "nb_gt_entities",
        ].iloc[0]
    )
    # Compute ranking of models per category
    top_models_per_label = {}
    for cat, group in df_filtered.groupby("category"):
        top_models_per_label[cat] = set(group.nlargest(top_k, metric)["model_name"])
    # Plot
    fig, ax = plt.subplots(figsize=(20, 15))
    bar_height = (1.2 - 0.2) / len(models)
    group_spacing = 1.2
    labels_index = np.arange(len(categories)) * group_spacing
    colors = [COLORS.get(label, "#272727") for label in categories]
    metric_capitalized = metric.replace("_", " ").title()
    # Plot bars for each model
    for model_idx, model_name in enumerate(models):
        df_model = df_filtered[df_filtered["model_name"] == model_name]
        values = []
        # Get metric values for each category for the current model
        for label in categories:
            val = df_model[df_model["category"] == label][metric]
            values.append(val.iloc[0] if not val.empty else 0)
        # Plot horizontal bars
        shift = bar_height * (len(models) - 1) / 2
        positions = labels_index - shift + model_idx * bar_height
        ax.barh(
            positions,
            values,
            height=bar_height,
            edgecolor="dimgrey",
            color=colors,
        )
        # Add legends
        for label, y, x in zip(categories, positions, values, strict=False):
            is_top = model_name in top_models_per_label[label]
            weight = "bold" if is_top else "normal"
            color = (
                "red"
                if model_name.startswith("fastino/gliner2")
                and not all(
                    models[i].startswith("fastino/gliner2") for i in range(len(models))
                )
                else "dimgrey"
            )
            # Model name (next to the bar)
            ax.text(
                x + 0.01 if x > 0 else 0.03,
                y,
                model_name,
                va="center",
                ha="left",
                color=color,
                fontweight=weight,
                fontsize=10,
            )
            # Score value (inside the bar)
            ax.text(
                x - 0.02 if x > 0.05 else x + 0.01,
                y,
                f"{x:.2f}" if x > 0 else "0",
                va="center",
                ha="right",
                fontsize=9,
                color="dimgrey",
                fontweight=weight,
            )
    # Axes
    ax.set_yticks(labels_index, categories, fontsize=12, weight="bold", color="#3D3D3D")
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric_capitalized, fontsize=12, fontweight="bold", color="#3D3D3D")
    # Title
    fig.suptitle(
        f"{metric_capitalized} comparison across models "
        f"({total_annotations} samples / "
        f"{total_entities} entities)",
        fontsize=23,
        fontweight="bold",
        color="#3D3D3D",
        y=0.92,
    )
    # Spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Invert y-axis to have the first category on top
    ax.invert_yaxis()
    # Show
    plt.show()
    plt.close()


def plot_llm_cost_vs_time(
    llm_df: pd.DataFrame,
    framework: str = "instructor",
    category: str = "OVERALL",
    figsize: tuple[int, int] = (11, 7),
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    Plot LLM models comparing inference time, cost, and precision score.

    Parameters
    ----------
    llm_df : pd.DataFrame
        Input dataframe containing evaluation metrics.
    framework : str
        Framework to filter on (default: "instructor").
    category : str
        Category filter (default: "OVERALL").
    figsize : tuple[int, int]
        Figure size.
    vmin : float
        Minimum value for color scale.
    vmax : float
        Maximum value for color scale.
    """
    df_plot = llm_df[
        (llm_df["category"] == category) & (llm_df["framework_name"] == framework)
    ].copy()

    df_plot["model_short"] = df_plot["model_name"].str.split("/").str[-1]

    _fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        df_plot["total_inference_time_sec"],
        df_plot["total_cost_usd"],
        c=df_plot["precision_score"],
        s=150,
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        edgecolors="gray",
        linewidths=0.5,
        zorder=3,
    )

    plt.colorbar(scatter, ax=ax, label="Precision score")

    texts = [
        ax.text(
            row["total_inference_time_sec"],
            row["total_cost_usd"],
            row["model_short"],
            fontsize=10,
        )
        for _, row in df_plot.iterrows()
    ]

    adjust_text(
        texts,
        ax=ax,
        arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.8},
    )

    ax.set_xlabel("Inference time (s)", fontsize=12)
    ax.set_ylabel("Cost (USD)", fontsize=12)
    ax.set_title("Cost vs inference time", fontsize=13)
    ax.grid(visible=True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
