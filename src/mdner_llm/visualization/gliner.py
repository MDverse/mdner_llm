"""Module for visualizing GLINER performance."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mdner_llm.visualization.theme import apply_journal_theme, generate_palette


def get_base_family(name: str) -> str:
    """Normalize checkpoint name by extracting the base model family.

    Returns
    -------
    str
        The base architecture family name, with any fine-tuning suffix removed.
    """
    clean_name = name.split("/")[-1].lower()
    return clean_name.replace("-finetuned", "").replace("_finetuned", "")


def get_generation_group(family_name: str) -> str:
    """Map architecture family name to its broad GLiNER generation group.

    Returns
    -------
       str
           One of 'GLiNER-BioMed', 'GLiNER2', or 'GLiNER2.5'.
    """
    if "biomed" in family_name:
        return "GLiNER-BioMed"
    return "GLiNER2.5" if "gliner2.5" in family_name else "GLiNER2"


def aggregate_cv_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GLiNER metrics by model and category across CV folds.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame with mean and standard deviation of metrics across folds,
        including information about the base model and F1 improvement.
    """
    # Aggregate metrics within each fold
    fold_metrics = df.groupby(["model", "category", "fold"], as_index=False).agg(
        precision=("precision_no_hallucination", "mean"),
        f1=("f1_no_hallucination", "mean"),
        recall=("recall", "mean"),
        total_inference_time_sec=("total_inference_time_sec", "sum"),
        nb_predicted_entities=("nb_predicted_entities", "sum"),
    )
    # Compute latency per predicted entity for each fold
    fold_metrics["latency_by_entity_sec"] = fold_metrics[
        "total_inference_time_sec"
    ].div(fold_metrics["nb_predicted_entities"].replace(0, np.nan))
    # Compute mean and standard deviation across folds
    summary = fold_metrics.groupby(["model", "category"], as_index=False).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        latency_by_entity_mean=("latency_by_entity_sec", "mean"),
        latency_by_entity_std=("latency_by_entity_sec", "std"),
        n_folds=("fold", "nunique"),
    )
    # Associate each fine-tuned model with its base model
    summary["base_model"] = summary["model"].str.replace("-finetuned", "", regex=False)
    is_finetuned = summary["model"].str.contains("finetuned", case=False, na=False)
    # Extract base-model scores for every category
    base_scores = summary.loc[
        ~is_finetuned,
        ["model", "category", "f1_mean"],
    ].rename(
        columns={
            "model": "base_model",
            "f1_mean": "base_f1_mean",
        }
    )
    # Match fine-tuned models with their base model within the same category
    summary = summary.merge(
        base_scores,
        on=["base_model", "category"],
        how="left",
        validate="many_to_one",
    )
    # Compute the F1 improvement over the corresponding base model
    summary["delta_f1"] = summary["f1_mean"] - summary["base_f1_mean"]
    summary.loc[~is_finetuned, "delta_f1"] = pd.NA
    # Remove temporary columns and sort models by category and F1 score
    return (
        summary.drop(columns=["base_model", "base_f1_mean"])
        .sort_values("f1_mean", ascending=[True])
        .reset_index(drop=True)
    )


def plot_benchmark_gliner_models(
    df: pd.DataFrame,
    human_iaa: float | None,
    metric: str = "f1",
    category: str = "OVERALL_MICRO",
) -> go.Figure:
    """Plot benchmark bar chart comparing base vs fine-tuned models.

    Returns
    -------
    go.Figure
        A Plotly figure object representing the benchmark comparison of GLiNER models
    """
    # Resolve mean and standard deviation column names
    metric_col = (
        metric.lower() if metric.lower().endswith("_mean") else f"{metric.lower()}_mean"
    )
    std_col = metric_col.replace("_mean", "_std")
    metric_title = metric.replace("_mean", "").capitalize()
    # Filter category and extract model attributes
    data = df[df["category"] == category].copy()
    data["base_family"] = data["model"].apply(get_base_family)
    data["generation_group"] = data["base_family"].apply(get_generation_group)
    data["is_finetuned"] = data["model"].str.contains("finetuned", case=False, na=False)
    data["x_axis_label"] = data["model"].apply(lambda s: s.split("/")[-1])
    data[std_col] = data[std_col].fillna(0.0)
    # Define explicit order: BioMed first, GLiNER2 second, GLiNER2.5 last
    family_order = [
        "gliner-biomed-base-v1.0",
        "gliner-biomed-large-v1.0",
        "gliner2-base-v1",
        "gliner2-large-v1",
        "gliner2.5-base-v1",
        "gliner2.5-large-v1",
    ]
    family_rank = {fam: i for i, fam in enumerate(family_order)}
    data = (
        data.assign(order=data["base_family"].map(lambda f: family_rank.get(f, 99)))
        .sort_values(["order", "is_finetuned"])
        .reset_index(drop=True)
    )
    # Map colors per architecture family
    family_palette = {
        "gliner-biomed-base-v1.0": "#b0bec5",
        "gliner-biomed-large-v1.0": "#78909c",
        "gliner2-base-v1": "#56c596",
        "gliner2-large-v1": "#2a9d8f",
        "gliner2.5-base-v1": "#2c5082",
        "gliner2.5-large-v1": "#153578",
    }
    unique_families = list(dict.fromkeys(data["base_family"]))
    default_colors = generate_palette(len(unique_families))
    family_to_color = {
        fam: family_palette.get(fam, default_colors[i])
        for i, fam in enumerate(unique_families)
    }
    figure = go.Figure()
    # Add metric bars with visible error bars and refined pattern solidity
    for _, row in data.iterrows():
        val, err = row[metric_col], row[std_col]
        figure.add_bar(
            x=[row["x_axis_label"]],
            y=[val],
            error_y={
                "type": "data",
                "array": [err],
                "visible": True,
                "thickness": 1.5,
                "width": 5,
                "color": "#222222",
            },
            name=row["model"],
            marker={
                "color": family_to_color[row["base_family"]],
                "line": {"color": "white", "width": 1.2},
                "pattern": {
                    "shape": "/" if row["is_finetuned"] else "",
                    "solidity": 0.22,
                    "fgcolor": "white",
                },
            },
            showlegend=False,
        )
        # Position score cleanly just above the error bar cap
        figure.add_annotation(
            x=row["x_axis_label"],
            y=val + err,
            text=f"{val:.2f}",
            showarrow=False,
            font={"size": 11, "color": "#111111"},
            yanchor="bottom",
            yshift=2,
        )
    # Add dummy proxy scatter traces for legend groups in logical order
    first_x = data["x_axis_label"].iloc[0]
    for group in ["GLiNER-BioMed", "GLiNER2", "GLiNER2.5"]:
        matches = data[data["generation_group"] == group]
        if not matches.empty:
            figure.add_trace(
                go.Scatter(
                    x=[first_x],
                    y=[None],
                    mode="markers",
                    name=group,
                    marker={
                        "size": 13,
                        "symbol": "square",
                        "color": family_to_color[matches.iloc[0]["base_family"]],
                    },
                    showlegend=True,
                )
            )
    # Draw full bridge lines touching bar tops and delta annotations
    x_positions = {label: i for i, label in enumerate(data["x_axis_label"])}
    for family in unique_families:
        subset = data[data["base_family"] == family].set_index("is_finetuned")
        if False in subset.index and True in subset.index:
            base, ft = subset.loc[False], subset.loc[True]
            y_base, y_ft = base[metric_col], ft[metric_col]
            delta = y_ft - y_base
            top_base = y_base + base[std_col]
            top_ft = y_ft + ft[std_col]
            h_bridge = max(top_base, top_ft) + 0.10
            figure.add_shape(
                type="path",
                path=(
                    f"M {base['x_axis_label']},{y_base} "
                    f"L {base['x_axis_label']},{h_bridge} "
                    f"L {ft['x_axis_label']},{h_bridge} "
                    f"L {ft['x_axis_label']},{y_ft}"
                ),
                fillcolor="rgba(0,0,0,0)",
                line={"color": "#666666", "width": 1.1},
            )
            mid_x = (
                x_positions[base["x_axis_label"]] + x_positions[ft["x_axis_label"]]
            ) / 2
            figure.add_annotation(
                x=mid_x,
                y=h_bridge,
                text=f"<b>{'+' if delta >= 0 else ''}{delta:.2f}</b>",
                showarrow=False,
                font={"size": 11, "color": "#111111"},
                align="center",
                yanchor="bottom",
                yshift=2,
            )
    # Add human inter-annotator agreement baseline line with non-overlapping label
    if human_iaa is not None:
        figure.add_hline(
            y=human_iaa,
            line={"color": "#666666", "dash": "dash", "width": 1.4},
            annotation_text="  Human IAA",
            annotation_position="top left",
            annotation_font={"size": 13, "color": "#555555"},
        )
    # Configure publication-ready theme and layout dimensions
    apply_journal_theme(figure, metric)
    figure.update_layout(
        hovermode=False,
        margin={"l": 80, "r": 140, "t": 90, "b": 150},
        xaxis={"type": "category", "tickangle": -25},
        yaxis={"range": [0, 1.16], "title": {"text": metric_title}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 14},
        },
    )
    return figure


def format_mean_std(mean: pd.Series, std: pd.Series) -> pd.Series:
    """Format mean and standard deviation as mean ± standard deviation.

    Returns
    -------
    pd.Series
        A series of strings formatted as "mean ± std" with two decimal places.
    """
    return mean.map("{:.2f}".format) + " ± " + std.map("{:.1f}".format)


def plot_confidence_retention_curves(
    df: pd.DataFrame,
    category: str | None = None,
    model_name: str | None = None,
    suggested_threshold: float | None = None,
) -> go.Figure:
    """Build Plotly retention (survival) comparison curves for TP and FP."""
    filtered_data = df.copy()
    if category is not None:
        filtered_data = filtered_data[filtered_data["category"] == category]
    if model_name is not None:
        filtered_data = filtered_data[
            filtered_data["model_name"].str.contains(model_name, case=False, na=False)
        ]
    threshold_grid = np.linspace(0.5, 1.0, 300)
    figure = go.Figure()
    distributions = [
        ("fp_scores", "False Positives", "#e76f51"),
        ("tp_scores", "True Positives", "#2a9d8f"),
    ]
    retention_at_threshold = {}
    for column_name, label, color in distributions:
        scores = np.array(
            [
                score
                for group in filtered_data[column_name].dropna()
                for score in group
                if score is not None and not np.isnan(score)
            ]
        )
        rates = np.array([(scores >= t).mean() * 100 for t in threshold_grid])
        if suggested_threshold is not None:
            retention_at_threshold[label] = (scores >= suggested_threshold).mean() * 100
        figure.add_scatter(
            x=threshold_grid,
            y=rates,
            mode="lines",
            name=f"{label} (N={len(scores):,})",
            line={"color": color, "width": 2.5},
        )
    if suggested_threshold is not None:
        figure.add_vline(
            x=suggested_threshold,
            line={"color": "#333333", "dash": "dash", "width": 1.5},
        )
        figure.add_annotation(
            x=suggested_threshold,
            y=102,
            text=f"<b>Threshold: {suggested_threshold:.2f}</b>",
            showarrow=False,
            font={"size": 13, "color": "#111111"},
            xanchor="center",
            yanchor="bottom",
        )
        for label, color in [
            ("True Positives", "#2a9d8f"),
            ("False Positives", "#e76f51"),
        ]:
            rate_val = retention_at_threshold.get(label, 0.0)
            figure.add_annotation(
                x=suggested_threshold,
                y=rate_val,
                text=f"<b>{rate_val:.0f}%</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.2,
                arrowcolor=color,
                ax=-40,
                ay=0,
                font={"size": 12, "color": color},
                bgcolor="white",
                bordercolor=color,
                borderwidth=1,
            )
    figure.update_layout(
        hovermode=False,
        xaxis={
            "title": {"text": "Confidence Threshold", "font": {"size": 14}},
            "range": [0.5, 1.0],
            "dtick": 0.1,
            "showgrid": True,
            "gridcolor": "#f0f0f0",
            "showline": True,
            "linecolor": "#333333",
            "linewidth": 1.2,
            "ticks": "outside",
        },
        yaxis={
            "title": {"text": "Retained Predictions (%)", "font": {"size": 14}},
            "range": [0, 108],
            "dtick": 20,
            "showgrid": True,
            "gridcolor": "#f0f0f0",
            "showline": True,
            "linecolor": "#333333",
            "linewidth": 1.2,
            "ticks": "outside",
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 70, "r": 40, "t": 60, "b": 60},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.08,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13},
        },
    )
    return figure


def plot_confidence_score_distributions(
    df: pd.DataFrame,
    category: str | None = None,
    model_name: str | None = None,
) -> go.Figure:
    """Build modern publication-ready histogram distributions for TP and FP."""
    filtered_data = df.copy()
    if category is not None:
        filtered_data = filtered_data[filtered_data["category"] == category]
    if model_name is not None:
        filtered_data = filtered_data[
            filtered_data["model_name"].str.contains(model_name, case=False, na=False)
        ]
    figure = go.Figure()
    distributions = [
        ("fp_scores", "False Positives", "#e76f51"),
        ("tp_scores", "True Positives", "#2a9d8f"),
    ]
    for column_name, label, color in distributions:
        scores = [
            score
            for group in filtered_data[column_name].dropna()
            for score in group
            if score is not None and not np.isnan(score)
        ]
        figure.add_trace(
            go.Histogram(
                x=scores,
                name=f"{label} (N={len(scores):,})",
                histnorm="probability density",
                opacity=0.65,
                marker={
                    "color": color,
                    "line": {"color": "#1f2937", "width": 1.2},
                },
                xbins={"start": 0.0, "end": 1.0, "size": 0.035},
            )
        )

    figure.update_layout(
        barmode="overlay",
        hovermode=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
        xaxis={
            "title": {
                "text": "Confidence Score",
                "font": {"size": 13, "color": "#111111"},
            },
            "range": [0.0, 1.02],
            "dtick": 0.1,
            "tick0": 0.0,
            "showgrid": False,
            "showline": True,
            "mirror": True,
            "linecolor": "#111111",
            "linewidth": 1.3,
            "ticks": "inside",
            "tickfont": {"size": 11, "color": "#111111"},
        },
        yaxis={
            "title": {"text": "Density", "font": {"size": 13, "color": "#111111"}},
            "showgrid": False,
            "showline": True,
            "mirror": True,
            "linecolor": "#111111",
            "linewidth": 1.3,
            "ticks": "inside",
            "tickfont": {"size": 11, "color": "#111111"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 12, "color": "#1f2937"},
        },
    )
    return figure
