"""Module to benchmark the effect of confidence score thresholding across models."""

from pathlib import Path

import pandas as pd
from plotly import graph_objects as go

from mdner_llm.core.evaluate_entities_extraction import (
    add_quality_columns,
    build_category_level_dataframe,
    compute_confusion_metrics_by_row,
    compute_grouped_stats,
    load_json_annotations_as_dataframe,
)


def evaluate_inferences_with_threshold(
    inferences_directory: Path | str,
    confidence_threshold: float | None = None,
) -> pd.DataFrame:
    """Evaluate the inferences in the specified directory with a confidence threshold.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the aggregated evaluation statistics
        for each model and category.

    Raises
    ------
    ValueError
        If no valid JSON annotations are found in the specified directory.
    """
    # Load the JSON annotations as a DataFrame.
    annotations_dataframe = load_json_annotations_as_dataframe(
        Path(inferences_directory)
    )
    # Check if the DataFrame is empty and raise an error if so.
    if annotations_dataframe.empty:
        msg = f"No valid JSON annotations found in {inferences_directory}."
        raise ValueError(msg)
    # Add quality columns based on the confidence threshold.
    quality_dataframe = add_quality_columns(
        annotations_dataframe, confidence_threshold=confidence_threshold
    )
    # Build a category-level DataFrame to compute confusion metrics.
    category_dataframe = build_category_level_dataframe(
        quality_dataframe, confidence_threshold=confidence_threshold
    )
    # Compute confusion metrics for each category.
    confusion_metrics = category_dataframe.apply(
        compute_confusion_metrics_by_row, axis=1
    )
    # Combine the category-level DataFrame with the confusion metrics.
    annotated_categories_dataframe = pd.concat(
        [category_dataframe, confusion_metrics], axis=1
    )
    # Compute grouped statistics for each model and category.
    grouped_statistics = compute_grouped_stats(
        quality_dataframe, annotated_categories_dataframe
    )
    # Add the confidence threshold to the grouped statistics DataFrame.
    grouped_statistics["threshold"] = confidence_threshold
    return grouped_statistics


def benchmark_confidence_threshold_impact(
    inferences_dir: Path | str,
    baseline_metrics: pd.DataFrame | Path | str,
    confidence_threshold: float = 0.8,
) -> pd.DataFrame:
    """Compare model performance before and after applying a confidence threshold.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the comparison of metrics before and after thresholding,
        including precision, recall, and F1 score for each model and category.
    """
    inferences_directory = Path(inferences_dir)
    baseline_dataframe = (
        pd.read_parquet(baseline_metrics)
        if isinstance(baseline_metrics, (Path, str))
        else baseline_metrics
    )
    # Filter out any baseline entries that are irrelevant to the models being evaluated.
    comparison_records = []
    model_directories = []
    for path in sorted(inferences_directory.iterdir()):
        if not path.is_dir():
            # Skip non-directory files in the inferences directory.
            continue
        if path.name == "folds":
            # If the directory contains "folds",
            # we assume it contains subdirectories for each fold.
            model_directories.extend([p for p in sorted(path.iterdir()) if p.is_dir()])
        else:
            # Otherwise, we treat it as a single model directory.
            model_directories.append(path)

    # Evaluate each model directory and compare metrics before and after thresholding.
    for model_directory in model_directories:
        if not any(model_directory.rglob("*.json")):
            continue
        filtered_statistics = evaluate_inferences_with_threshold(
            inferences_directory=model_directory,
            confidence_threshold=confidence_threshold,
        )
        is_fold = model_directory.parent.name == "folds"
        display_model_name = (
            f"folds/{model_directory.name}" if is_fold else model_directory.name
        )
        # Match baseline on the specific directory name rather than a broad substring.
        target_name = model_directory.name
        baseline_matches = baseline_dataframe[
            (
                baseline_dataframe["model_name"].str.contains(
                    target_name, case=False, na=False
                )
            )
        ]
        # Aggregate across folds for each category.
        for current_category, group in filtered_statistics.groupby("category"):
            cat_baseline = baseline_matches[
                baseline_matches["category"] == current_category
            ]
            precision_before = cat_baseline["precision"].mean()
            recall_before = cat_baseline["recall"].mean()
            f1_before = cat_baseline["f1"].mean()
            # Mean post-threshold metrics across all folds of this model.
            precision_after = group["precision"].mean()
            recall_after = group["recall"].mean()
            f1_after = group["f1"].mean()
            comparison_records.append(
                {
                    "model": display_model_name,
                    "category": current_category,
                    "threshold": confidence_threshold,
                    "precision_before": precision_before,
                    "precision_after": precision_after,
                    "delta_precision": precision_after - precision_before,
                    "recall_before": recall_before,
                    "recall_after": recall_after,
                    "delta_recall": recall_after - recall_before,
                    "f1_before": f1_before,
                    "f1_after": f1_after,
                    "delta_f1": f1_after - f1_before,
                }
            )

    return (
        pd.DataFrame(comparison_records)
        .sort_values(by=["category", "f1_after"], ascending=[True, False])
        .reset_index(drop=True)
    )


def plot_benchmark_models_with_threshold(
    comparison_dataframe: pd.DataFrame, metric: str, category: str
) -> go.Figure:
    """Plot paired horizontal bar comparison for a category.

    Returns
    -------
    go.Figure
        A Plotly figure object visualizing the before and after metrics for each model.
    """
    # Filter dataset for target evaluation category
    plot_data = comparison_dataframe[
        comparison_dataframe["category"] == category
    ].copy()
    metric_before = f"{metric}_before"
    metric_after = f"{metric}_after"
    delta_metric = f"delta_{metric}"
    # Sort models ascending by post-threshold score
    plot_data = plot_data.sort_values(by=metric_after, ascending=True).reset_index(
        drop=True
    )
    threshold_value = (
        plot_data["threshold"].iloc[0] if "threshold" in plot_data else 0.0
    )
    figure = go.Figure()
    # Filtered score (turquoise green)
    figure.add_trace(
        go.Bar(
            y=plot_data["model"],
            x=plot_data[metric_after],
            orientation="h",
            name=f"After Filtering (Score ≥ {threshold_value:.2f})",
            marker={
                "color": "#62c5a5",
                "line": {"color": "#222222", "width": 1.2},
            },
            hoverinfo="skip",
        )
    )
    # Baseline score (salmon red)
    figure.add_trace(
        go.Bar(
            y=plot_data["model"],
            x=plot_data[metric_before],
            orientation="h",
            name="Before Filtering",
            marker={
                "color": "#fa796b",
                "line": {"color": "#222222", "width": 1.2},
            },
            hoverinfo="skip",
        )
    )
    # Numeric delta annotation placed at the end of the longer bar
    for _, row in plot_data.iterrows():
        max_x = max(row[metric_before], row[metric_after])
        delta = row[delta_metric]
        symbol = "+" if delta > 0 else ""
        figure.add_annotation(
            x=max_x + 0.015,
            y=row["model"],
            text=f"{symbol}{delta:.2f}",
            showarrow=False,
            font={
                "size": 11,
                "color": "#1ca347" if delta >= 0 else "#d90429",
            },
            xanchor="left",
            yanchor="middle",
        )
    max_limit = (
        max(plot_data[metric_before].max(), plot_data[metric_after].max())
        if not plot_data.empty
        else 1.0
    )
    # Styling and layout adjustments
    figure.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        hovermode=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 200, "r": 60, "t": 60, "b": 60},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 12},
        },
        xaxis={
            "title": {
                "text": metric.capitalize(),
                "font": {"size": 13, "color": "#111111"},
            },
            "range": [0, min(1.05, max_limit + 0.08)],
            "dtick": 0.1,
            "showgrid": False,
            "showline": True,
            "mirror": True,
            "linecolor": "#111111",
            "linewidth": 1.2,
            "ticks": "inside",
            "tickfont": {"size": 11, "color": "#111111"},
        },
        yaxis={
            "showgrid": False,
            "showline": True,
            "mirror": True,
            "linecolor": "#111111",
            "linewidth": 1.2,
            "ticks": "inside",
            "tickfont": {"size": 11, "color": "#111111"},
        },
    )
    return figure
