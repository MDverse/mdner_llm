"""Module to plot performance metrics."""

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from adjustText import adjust_text

from mdner_llm.annotations.visualize_annotations import COLORS

OPEN_WEIGHT_KEYWORDS = [
    "mistral",
    "mixtral",
    "llama",
    "qwen",
    "gemma",
    "falcon",
    "phi",
    "deepseek",
    "yi",
    "vicuna",
    "wizardlm",
    "openchat",
    "zephyr",
    "solar",
    "glm",
    "z-ai",
    "gemma",
    "minimax",
]


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
    # Compute ranking of models per category
    top_models_per_label = {}
    for cat, group in df_filtered.groupby("category"):
        top_models_per_label[cat] = set(group.nlargest(top_k, metric)["model_name"])
    # Plot
    fig, ax = plt.subplots(figsize=(20, 15))
    bar_height = (1.2 - 0.2) / len(models)
    group_spacing = 1.2
    labels_index = np.arange(len(categories)) * group_spacing
    colors = [COLORS.get(label, "#BAB7BA") for label in categories]
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
                fontsize=24,
            )
            # Score value (inside the bar)
            ax.text(
                x - 0.02 if x > 0.05 else x + 0.01,
                y,
                f"{x:.2f}" if x > 0 else "0",
                va="center",
                ha="right",
                fontsize=24,
                color="dimgrey",
                fontweight=weight,
            )
    # Axes
    ax.set_yticks(labels_index, categories, fontsize=18, weight="bold", color="#3D3D3D")
    ax.set_xlim(0, 1)
    ax.tick_params(axis="x", labelsize=18, labelcolor="#3D3D3D")
    ax.set_xlabel(metric_capitalized, fontsize=20, fontweight="bold", color="#3D3D3D")
    # Title
    fig.suptitle(
        f"{metric_capitalized} comparison across models ",
        fontsize=25,
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
    """Plot LLM models comparing inference time, cost, and precision score."""
    # Filter the DataFrame for the specified framework and category
    df_plot = llm_df[
        (llm_df["category"] == category) & (llm_df["framework_name"] == framework)
    ].copy()
    # Extract short model names for labeling
    df_plot["model_short"] = df_plot["model_name"].str.split("/").str[-1]
    # Determine if models are open-weight based on keywords
    pattern = "|".join(OPEN_WEIGHT_KEYWORDS)
    df_plot["is_open"] = (
        df_plot["model_short"].str.lower().str.contains(pattern, regex=True)
    )
    # Plotting
    _fig, ax = plt.subplots(figsize=figsize)
    scatter_ref = None
    # Define marker styles for open-weight vs frontier models
    # Squares for frontier, circles for open-weight
    marker_cfg = [
        (True, "o", "Open-weight"),
        (False, "^", "Frontier"),
    ]
    for is_open, marker, _label in marker_cfg:
        mask = df_plot["is_open"] == is_open
        sc = ax.scatter(
            df_plot.loc[mask, "total_inference_time_sec"],
            df_plot.loc[mask, "total_cost_usd"],
            c=df_plot.loc[mask, "precision_score"],
            s=150,
            marker=marker,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            edgecolors="gray",
            linewidths=0.5,
            zorder=3,
        )
        if scatter_ref is None:
            scatter_ref = sc
    plt.colorbar(scatter_ref, ax=ax, label="Precision score")
    # Add legend for model types
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor="#aaaaaa",
            markeredgecolor="gray",
            markeredgewidth=0.8,
            markersize=9,
            label="Open-weight",
        ),
        mlines.Line2D(
            [],
            [],
            marker="^",
            color="none",
            markerfacecolor="#aaaaaa",
            markeredgecolor="gray",
            markeredgewidth=0.8,
            markersize=9,
            label="Frontier",
        ),
    ]
    ax.legend(
        handles=legend_handles, title="Model type", loc="upper left", framealpha=0.8
    )

    # Labels of models
    texts = [
        ax.text(
            row["total_inference_time_sec"],
            row["total_cost_usd"],
            row["model_short"],
            fontsize=10,
            fontweight="bold",
            color="dimgrey",
        )
        for _, row in df_plot.iterrows()
    ]
    adjust_text(
        texts,
        ax=ax,
        expand=(1.4, 1.8),
    )
    # Axes and title
    ax.set_xlabel("Inference time (seconds, log10 scale)", fontsize=12)
    ax.set_xscale("log")
    ax.set_ylabel("Cost ($)", fontsize=12)
    ax.set_title("Cost vs inference time", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_mean_model_performance(csv_path, score_column) -> None:
    """Plot score comparisons across models, showing overall mean and std."""
    # Load the data
    df = pd.read_csv(csv_path)
    # Filter rows where category is exactly 'OVERALL'
    df = df[df["category"] == "OVERALL"]
    # Sort models by run number to keep the Y-axis clean
    df["sort_key"] = df["model_name"].str.extract(r"run(\d+)").astype(int)
    df = df.sort_values("sort_key")
    # Compute statistics
    mean_val = df[score_column].mean()
    std_val = df[score_column].std()
    # Reformatting
    clean_score_name = score_column.replace("_", " ").title()
    df["bar_label"] = df.apply(
        lambda row: f"{row[score_column]:.2f}     {row['model_name']}", axis=1
    )
    # Plot
    fig = px.bar(
        df,
        x=score_column,
        y="model_name",
        title=(
            f"Comparison of {clean_score_name}"
            f"<br><sup>Mean: {mean_val:.2f}  |  Std Dev: {std_val:.2f}</sup>"
        ),
        text="bar_label",
        labels={score_column: clean_score_name},
        hover_data={
            "model_name": True,
            score_column: ":.2f",
            "nb_texts_with_category": True,
            "pct_correct_format": ":.2f",
            "total_cost_usd": ":$.4f",
            "total_inference_time_sec": ":.2f",
            "sort_key": False,
        },
        template="plotly_white",
        orientation="h",
    )
    # Add vertical line for Mean
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=1.5)
    # Add shaded area for Standard Deviation
    fig.add_vrect(
        x0=mean_val - std_val,
        x1=mean_val + std_val,
        fillcolor="rgba(255, 0, 0, 0.08)",
        line_width=0,
    )
    # Configure axes
    fig.update_layout(
        xaxis={"tickmode": "linear", "tick0": 0, "dtick": 0.2, "range": [0, 1.0]},
        yaxis={
            "autorange": "reversed",
            "showticklabels": False,
            "title": "Model Names",
        },
        margin={"r": 250},  # Allocates white space on the right for model labels
        title_font_size=16,
        height=600,
    )
    # Style bars and text
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont_size=12,
        marker_color="#262626",
    )
    # Display the chart
    fig.show()
