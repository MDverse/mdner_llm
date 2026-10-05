"""Module for visualizing GLINER performance."""

import ast
import operator
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gliner2.training.trainer import TrainingConfig
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from mdner_llm.visualization.theme import apply_journal_theme, generate_palette

# Map colors per architecture family
FAMILY_PALETTE = {
    "gliner-biomed-base-v1.0": "#b0bec5",
    "gliner-biomed-large-v1.0": "#78909c",
    "gliner2-base-v1": "#56c596",
    "gliner2-large-v1": "#2a9d8f",
    "gliner2.5-base-v1": "#2c5082",
    "gliner2.5-large-v1": "#153578",
}


def generate_gradient_colors(
    base_color: str,
    n_colors: int,
    min_alpha: float = 0.35,
    max_alpha: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """Generate RGBA colors with opacity gradient.

    Returns
    -------
    list of RGBA tuples
    """
    red, green, blue, _ = to_rgba(base_color)
    alphas = np.linspace(min_alpha, max_alpha, max(n_colors, 1))
    return [(red, green, blue, a) for a in alphas]


def plot_loss_evolution(
    axis: plt.Axes, results_list: list[dict], cfg: "TrainingConfig"
) -> None:
    """Plot training and validation loss curves across folds."""
    # Generate gradient colors for training and evaluation curves.
    number_of_folds = len(results_list)
    train_colors = generate_gradient_colors("#0055FF", number_of_folds)
    eval_colors = generate_gradient_colors("#FFAA00", number_of_folds)
    # Configure axes layout and appearance.
    axis.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.4, color="#94A3B8")
    for spine_name in ("top", "right"):
        axis.spines[spine_name].set_visible(False)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_ylim(bottom=0, top=1000)
    axis.set_xlim(left=0, right=cfg.training.num_epochs)
    global_min_loss = {"loss": float("inf"), "epoch": None, "fold": None, "color": None}
    # Plot loss trajectory per fold and track best evaluation point.
    for fold_index, results in enumerate(results_list):
        fold_train_color = train_colors[fold_index]
        fold_eval_color = eval_colors[fold_index]
        for key, color, is_eval in (
            ("train_metrics_history", fold_train_color, False),
            ("eval_metrics_history", fold_eval_color, True),
        ):
            metric_key = "eval_loss" if is_eval else "loss"
            loss_by_epoch = {}
            for entry in results.get(key, []):
                epoch = int(entry["epoch"])
                loss_value = float(entry[metric_key])
                loss_by_epoch[epoch] = min(
                    loss_by_epoch.get(epoch, float("inf")), loss_value
                )
            epochs = sorted(loss_by_epoch)
            losses = [loss_by_epoch[epoch] for epoch in epochs]
            axis.plot(
                epochs, losses, color=color, linewidth=1.6, marker="o", markersize=3
            )
            if is_eval:
                for epoch, loss_val in zip(epochs, losses, strict=False):
                    if loss_val < global_min_loss["loss"]:
                        global_min_loss = {
                            "loss": loss_val,
                            "epoch": epoch,
                            "fold": fold_index + 1,
                            "color": color,
                        }
    # Annotate minimum evaluation loss point.
    if global_min_loss["epoch"] is not None:
        axis.scatter(
            global_min_loss["epoch"],
            global_min_loss["loss"],
            s=80,
            color=global_min_loss["color"],
            edgecolors="#1E293B",
            linewidths=1.2,
            zorder=10,
        )
        axis.annotate(
            f"Best eval loss\nFold {global_min_loss['fold']} "
            f"(Ep. {global_min_loss['epoch']})\nLoss: {global_min_loss['loss']:.0f}",
            xy=(global_min_loss["epoch"], global_min_loss["loss"]),
            xytext=(15, -18),
            textcoords="offset points",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": global_min_loss["color"],
                "alpha": 0.95,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": global_min_loss["color"],
                "linewidth": 1.2,
            },
        )
    # Render training and validation fold legends.
    for title, anchor_x, color_theme, colors in (
        ("Train", 1.00, "#0055FF", train_colors),
        ("Validation", 0.87, "#FFAA00", eval_colors),
    ):
        handles = [
            Line2D([0], [0], color=colors[fold_idx], lw=2, label=f"Fold {fold_idx + 1}")
            for fold_idx in range(number_of_folds)
        ]
        legend = axis.legend(
            handles=handles,
            title=title,
            loc="upper right",
            bbox_to_anchor=(anchor_x, 1.00),
            frameon=True,
            facecolor="white",
            edgecolor="#E2E8F0",
            fontsize=7.5,
            title_fontsize=8,
        )
        legend.get_title().set_color(color_theme)
        legend.get_title().set_weight("bold")
        axis.add_artist(legend)
    # Set axis titles and labels.
    axis.set_xlabel("Epoch", fontsize=10.5)
    axis.set_ylabel("Loss", fontsize=10.5)
    axis.set_title("Loss Evolution", fontsize=11.5, pad=8, fontweight="medium")


def collect_validation_metric_records(
    results_list: list[dict],
    target_metric_keys: tuple[str, ...],
) -> tuple[dict[str, dict[int, list[float]]], dict[str, float | int | None]]:
    """Extract validation scores per epoch and identify global peak F1 performance.

    Returns
    -------
    tuple[dict[str, dict[int, list[float]]], dict[str, float | int | None]]
        Tuple containing scores grouped by metric/epoch and global best F1 record.

    Examples
    --------
    >>> # Input fold results from cross-validation:
    >>> # results = [
    >>> #     {"eval_metrics_history":
    >>> #      [{"epoch": 1, "eval_f1": 0.82, "eval_precision": 0.85}]},
    >>> #     {"eval_metrics_history":
    >>> #      [{"epoch": 1, "eval_f1": 0.88, "eval_precision": 0.90}]},
    >>> # ]
    >>> # Output scores_by_metric:
    >>> # {"eval_f1": {1: [0.82, 0.88]}, "eval_precision": {1: [0.85, 0.90]}}
    >>> # Output peak_f1_record:
    >>> # {"score": 0.88, "epoch": 1, "fold": 2}
    """
    scores_by_metric = {
        metric_key: defaultdict(list) for metric_key in target_metric_keys
    }
    peak_f1_record = {"score": float("-inf"), "epoch": None, "fold": None}
    # Traverse each fold evaluation history.
    # Example fold entry: {"epoch": 2, "eval_f1": 0.85, "eval_precision": 0.80}.
    for fold_index, results in enumerate(results_list):
        current_fold_number = fold_index + 1
        for entry in results.get("eval_metrics_history", []):
            epoch_number = int(entry["epoch"])
            for metric_key in target_metric_keys:
                raw_metric_value = entry.get(metric_key)
                if raw_metric_value is None:
                    continue

                metric_score = float(raw_metric_value)
                scores_by_metric[metric_key][epoch_number].append(metric_score)

                # Keep track of global maximum validation F1 score across all folds.
                if metric_key == "eval_f1" and metric_score > peak_f1_record["score"]:
                    peak_f1_record["score"] = metric_score
                    peak_f1_record["epoch"] = epoch_number
                    peak_f1_record["fold"] = current_fold_number

    return scores_by_metric, peak_f1_record


def plot_validation_metrics_evolution(
    axis: plt.Axes,
    results_list: list[dict],
    cfg: "TrainingConfig",
) -> None:
    """Plot aggregated mean validation metrics across folds."""
    metric_configs = {
        "eval_precision": {
            "label": "Precision",
            "color": "#2563EB",
            "style": "--",
            "marker": "^",
            "width": 1.8,
        },
        "eval_f1": {
            "label": "F1",
            "color": "#7C3AED",
            "style": "-",
            "marker": "o",
            "width": 2.2,
        },
        "eval_recall": {
            "label": "Recall",
            "color": "#DC2626",
            "style": ":",
            "marker": "s",
            "width": 1.8,
        },
    }
    axis.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.4, color="#94A3B8")
    for spine_name in ("top", "right"):
        axis.spines[spine_name].set_visible(False)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_ylim(bottom=0, top=1.0)
    axis.set_xlim(left=0, right=cfg.training.num_epochs)
    scores_by_metric, peak_f1 = collect_validation_metric_records(
        results_list, tuple(metric_configs.keys())
    )
    f1_color = metric_configs["eval_f1"]["color"]
    for epoch_number, f1_scores in scores_by_metric["eval_f1"].items():
        for single_f1_score in f1_scores:
            axis.plot(
                epoch_number,
                single_f1_score,
                marker="o",
                markersize=2.5,
                color=f1_color,
                alpha=0.25,
            )
    legend_handles = []
    for metric_key, config in metric_configs.items():
        epoch_records = scores_by_metric[metric_key]
        if not epoch_records:
            continue
        sorted_epochs = sorted(epoch_records)
        epoch_values = [epoch_records[epoch] for epoch in sorted_epochs]
        metric_means = np.array([np.mean(values) for values in epoch_values])
        axis.plot(
            sorted_epochs,
            metric_means,
            color=config["color"],
            linestyle=config["style"],
            marker=config["marker"],
            linewidth=config["width"],
            markersize=4.5,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=config["color"],
                lw=config["width"],
                linestyle=config["style"],
                marker=config["marker"],
                markersize=4,
                label=config["label"],
            )
        )
    if peak_f1["epoch"] is not None:
        axis.scatter(
            peak_f1["epoch"],
            peak_f1["score"],
            s=85,
            color=f1_color,
            edgecolors="#1E293B",
            linewidths=1.2,
            zorder=10,
        )
        axis.annotate(
            f"Best eval F1\nFold {peak_f1['fold']} "
            f"(Ep. {peak_f1['epoch']})\nF1: {peak_f1['score']:.2f}",
            xy=(peak_f1["epoch"], peak_f1["score"]),
            xytext=(15, -18),
            textcoords="offset points",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": f1_color,
                "alpha": 0.95,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": f1_color,
                "linewidth": 1.2,
            },
        )
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#E2E8F0",
        fontsize=8.5,
    )
    axis.set_xlabel("Epoch", fontsize=10.5)
    axis.set_ylabel("Validation metric", fontsize=10.5)
    axis.set_title(
        "Cross-Validation Metrics (Mean)",
        fontsize=11.5,
        pad=8,
        fontweight="medium",
    )


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
    unique_families = list(dict.fromkeys(data["base_family"]))
    default_colors = generate_palette(len(unique_families))
    family_to_color = {
        fam: FAMILY_PALETTE.get(fam, default_colors[i])
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
    # Format mean to two decimal places and std to one decimal place
    # if std is NaN, remove the ± part and just show the mean
    if std.isna().all():
        return mean.map("{:.2f}".format)
    return mean.map("{:.2f}".format) + " ± " + std.map("{:.1f}".format)


def plot_confidence_retention_curves(
    df: pd.DataFrame,
    category: str | None = None,
    model_name: str | None = None,
    suggested_threshold: float | None = None,
) -> go.Figure:
    """Build Plotly retention comparison curves for TP and FP.

    Returns
    -------
    go.Figure
        A Plotly figure object representing the retention curves for true positives
        and false positives
    """
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
    """Build histogram distributions for TP and FP.

    Returns
    -------
    go.Figure
        A Plotly figure object representing the histogram distributions
        for true positives and false positives.
    """
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


def evaluate_confidence_separation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation and separation metrics for confidence scores.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame containing point-biserial correlation, p-value, and ROC-AUC
        for each model, along with counts of true positives and false positives.
    """
    # 1. Flatten TP and FP score lists into prediction records
    records = []
    for _, row in df.iterrows():
        # Parse serialized string representations of lists if necessary
        tp_raw = (
            ast.literal_eval(row["tp_scores"])
            if isinstance(row["tp_scores"], str)
            else row["tp_scores"]
        )
        fp_raw = (
            ast.literal_eval(row["fp_scores"])
            if isinstance(row["fp_scores"], str)
            else row["fp_scores"]
        )
        model_name = row.get("model", row.get("model_name"))
        # Append true positives (label = 1)
        if tp_raw is not None and len(tp_raw) > 0:
            records.extend(
                [
                    {
                        "model": model_name,
                        "confidence": float(score),
                        "is_tp": 1,
                    }
                    for score in tp_raw
                ]
            )
        # Append false positives (label = 0)
        if fp_raw is not None and len(fp_raw) > 0:
            records.extend(
                [
                    {
                        "model": model_name,
                        "confidence": float(score),
                        "is_tp": 0,
                    }
                    for score in fp_raw
                ]
            )
    preds_df = pd.DataFrame(records)
    # 2. Compute correlation and separation metrics per model
    table_rows = []
    for model_name, group in preds_df.groupby("model"):
        y_true = group["is_tp"].to_numpy()
        conf_scores = group["confidence"].to_numpy()
        # Point-biserial measures linear relationship
        # between continuous score and binary label
        r_pb, p_pb = stats.pointbiserialr(y_true, conf_scores)
        # ROC-AUC measures probability that a random TP
        # has a higher score than a random FP
        auc = roc_auc_score(y_true, conf_scores)
        table_rows.append(
            {
                "Model": model_name,
                "Number of True Positives": int(np.sum(y_true == 1)),
                "Number of False Positives": int(np.sum(y_true == 0)),
                "Point-Biserial Correlation": r_pb,
                "P-Value": f"{p_pb:.1e}",
                "ROC-AUC": auc,
            }
        )
    return (
        pd.DataFrame(table_rows)
        .sort_values(by="ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )


def plot_paper_roc_curves(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> go.Figure:
    """Plot ROC curves for multiple models with AUC metrics and reference lines.

    Returns
    -------
    go.Figure: Plotly figure object containing the ROC curves and reference lines.
    """
    # 1. Flatten prediction scores into individual records
    records = []
    for _, row in df.iterrows():
        tp_raw = (
            ast.literal_eval(row["tp_scores"])
            if isinstance(row["tp_scores"], str)
            else row["tp_scores"]
        )
        fp_raw = (
            ast.literal_eval(row["fp_scores"])
            if isinstance(row["fp_scores"], str)
            else row["fp_scores"]
        )
        model_name = row.get("model", row.get("model_name"))
        if tp_raw is not None and len(tp_raw) > 0:
            records.extend(
                [
                    {
                        "model": model_name,
                        "confidence": float(score),
                        "is_tp": 1,
                    }
                    for score in tp_raw
                ]
            )
        if fp_raw is not None and len(fp_raw) > 0:
            records.extend(
                [
                    {
                        "model": model_name,
                        "confidence": float(score),
                        "is_tp": 0,
                    }
                    for score in fp_raw
                ]
            )
    preds_df = pd.DataFrame(records)

    # 2. Compute ROC coordinates and AUC metrics per model
    model_curves = []
    for model_name, group in preds_df.groupby("model"):
        y_true = group["is_tp"].to_numpy()
        conf_scores = group["confidence"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, conf_scores)
        auc = roc_auc_score(y_true, conf_scores)
        model_curves.append({"model": model_name, "fpr": fpr, "tpr": tpr, "auc": auc})
    model_curves.sort(key=operator.itemgetter("auc"), reverse=True)

    # 3. Build figure for publication
    fig = go.Figure()

    # Reference: Perfect Classifier
    fig.add_trace(
        go.Scatter(
            x=[0, 0, 1],
            y=[0, 1, 1],
            mode="lines",
            line={"color": "#7E57C2", "width": 2.5},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Reference: Random Classifier
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "#E53935", "width": 2, "dash": "dash"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Empirical curves styled using family palette
    for item in model_curves:
        color = FAMILY_PALETTE.get(item["model"], "#424242")
        fig.add_trace(
            go.Scatter(
                x=item["fpr"],
                y=item["tpr"],
                mode="lines",
                line={"color": color, "width": 2.5},
                name=f"{item['model']} (AUC = {item['auc']:.2f})",
            )
        )
    # In-graph reference curve annotations
    fig.add_annotation(
        x=0.02,
        y=0.97,
        text="<b>PERFECT CLASSIFIER</b>",
        font={"color": "#7E57C2", "size": 10, "family": "Arial"},
        showarrow=False,
        xanchor="left",
    )
    fig.add_annotation(
        x=0.52,
        y=0.48,
        text="<b>RANDOM CLASSIFIER</b>",
        textangle=-36,
        font={"color": "#E53935", "size": 10, "family": "Arial"},
        showarrow=False,
    )

    # Publication layout without title and minimal margins
    fig.update_layout(
        hovermode=False,
        title=None,
        margin={"l": 55, "r": 20, "t": 20, "b": 55},
        xaxis={
            "title": "<b>False Positive Rate</b> (1 - Specificity)",
            "range": [-0.01, 1.01],
            "showgrid": True,
            "gridcolor": "#ECEFF1",
            "linecolor": "#37474F",
            "linewidth": 1,
            "zeroline": False,
        },
        yaxis={
            "title": "<b>True Positive Rate</b> (Sensitivity)",
            "range": [-0.01, 1.02],
            "showgrid": True,
            "gridcolor": "#ECEFF1",
            "linecolor": "#37474F",
            "linewidth": 1,
            "zeroline": False,
        },
        plot_bgcolor="#FFFFFF",
        width=700,
        height=560,
        legend={
            "x": 0.63,
            "y": 0.05,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "#CFD8DC",
            "borderwidth": 1,
            "font": {"size": 10, "family": "Arial"},
        },
    )

    if save_path:
        fig.write_image(save_path) if save_path.endswith(
            (".svg", ".pdf", ".png")
        ) else fig.write_html(save_path)

    return fig
