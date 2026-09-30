"""Module for visualizing LLM performance."""

import pandas as pd
import plotly.graph_objects as go
from pandas.io.formats.style import Styler

from mdner_llm.visualization.theme import apply_journal_theme, generate_palette


def display_strategies_table(file_path_or_df: str | pd.DataFrame) -> Styler:
    """Build and format a centered ablation summary table for LLM strategies.

    Returns
    -------
    Styler
        A pandas Styler object representing the formatted ablation table.
    """
    raw_df = (
        pd.read_csv(file_path_or_df)
        if isinstance(file_path_or_df, str)
        else file_path_or_df.copy()
    )
    # Filter micro and macro evaluation summaries.
    eval_df = raw_df[raw_df["category"].isin(["OVERALL_MICRO", "OVERALL_MACRO"])].copy()
    # Extract per-run metrics from micro rows and map macro F1 scores.
    micro_df = eval_df[eval_df["category"] == "OVERALL_MICRO"].copy()
    macro_series = eval_df[eval_df["category"] == "OVERALL_MACRO"].set_index(
        ["Name", "Framework", "With_Guideline"]
    )["F1"]
    micro_df["Macro F1"] = micro_df.set_index(
        ["Name", "Framework", "With_Guideline"]
    ).index.map(macro_series)
    # Standardize boolean flags for Instructor usage and Guideline inclusion.
    micro_df["Instructor"] = (
        micro_df["Framework"]
        .str.lower()
        .str.contains("instructor")
        .map({True: "✓", False: "✗"})
    )
    micro_df["Guidelines"] = (
        micro_df["With_Guideline"].astype(bool).map({True: "✓", False: "✗"})
    )
    # Assign publication column names and sort by model and strategy hierarchy.
    ablation_table = (
        micro_df.assign(
            **{
                "Model": micro_df["Name"],
                "Valid Format (%)": micro_df["Correct_format_(%)"],
                "Hallucinations (%)": micro_df["Hallucinations_(%)"],
                "Precision": micro_df["Precision_with_no_hallucination"],
                "Micro F1": micro_df["F1_with_no_hallucination"],
                "Latency by entity (s)": micro_df["Inference_time_by_entity_(s)"],
                "Cost by entity ($)": micro_df["Cost_by_entity_($)"],
            }
        )[
            [
                "Model",
                "Instructor",
                "Guidelines",
                "Valid Format (%)",
                "Hallucinations (%)",
                "Precision",
                "Micro F1",
                "Macro F1",
                "Latency by entity (s)",
                "Cost by entity ($)",
            ]
        ]
        .sort_values(
            ["Model", "Instructor", "Guidelines"], ascending=[True, True, True]
        )
        .reset_index(drop=True)
    )
    # Render centered publication-ready table with numeric formatting.
    return (
        ablation_table.style.format(
            {
                "Valid Format (%)": "{:.0f}",
                "Hallucinations (%)": "{:.0f}",
                "Precision": "{:.2f}",
                "Micro F1": "{:.2f}",
                "Macro F1": "{:.2f}",
                "Latency by entity (s)": "{:.2f}",
                "Cost by entity ($)": "{:.2f}",
            },
            na_rep="-",
        )
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    )


def plot_benchmark_llm_strategy(
    df: pd.DataFrame,
    human_iaa: float | None,
    metric: str,
    category: str,
    models_to_keep: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot the performance of different LLM strategies for a given metric.

    Returns
    -------
    go.Figure
        A Plotly figure object representing the benchmark performance of LLM strategies.
    """
    # Define mapping dictionary for strategy display labels.
    combo_labels = {
        "no_instructor_no_guidelines": "No instructor<br>no guideline",
        "no_instructor_with_guidelines": "No instructor<br>with guideline",
        "with_instructor_no_guidelines": "With instructor<br>no guideline",
        "with_instructor_with_guidelines": "With instructor<br>with guideline",
    }
    # Filter dataset by selected category.
    data = df[df["category"] == category].copy()
    # Filter by specific models if provided, otherwise keep all models.
    if models_to_keep is not None and len(models_to_keep) > 0:
        data = data[data["Name"].isin(models_to_keep)]
    # Construct strategy key matching combo_labels dictionary keys.
    inst_part = (
        data["Framework"]
        .astype(str)
        .str.lower()
        .apply(lambda x: "with_instructor" if "instructor" in x else "no_instructor")
    )
    guide_part = data["With_Guideline"].map(
        {True: "with_guidelines", False: "no_guidelines"}
    )
    data["combo_key"] = inst_part + "_" + guide_part
    # Map raw strategy keys into formatted multi-line strings.
    data["strategy"] = data["combo_key"].map(combo_labels)
    # Filter strategies maintaining predefined order.
    strategies = [lbl for lbl in combo_labels.values() if lbl in set(data["strategy"])]
    # Directly assign input metric column.
    metric_col = metric
    # Aggregate duplicates by model and strategy to allow reindexing.
    data = data.groupby(["Name", "strategy"], as_index=False)[metric_col].mean()
    # Compute maximum value per strategy for bold highlighting.
    max_per_strat = data.groupby("strategy")[metric_col].apply(
        lambda s: s.round(2).max()
    )
    models = data["Name"].unique()
    # Generate distinct colors for each unique model.
    palette = generate_palette(len(models))
    fig = go.Figure()
    # Add grouped bars with journal styling and highlighted values.
    for model, color in zip(models, palette, strict=False):
        sub = data[data["Name"] == model].set_index("strategy").reindex(strategies)
        text_labels = [
            f"<b>{value:.2f}</b>"
            if round(value, 2) == max_per_strat.get(strat)
            else f"{value:.2f}"
            for strat, value in sub[metric_col].items()
        ]
        fig.add_bar(
            x=strategies,
            y=sub[metric_col],
            name=model,
            marker={"color": color, "line": {"color": "white", "width": 1}},
            text=text_labels,
            textposition="outside",
            textfont={"size": 13},
        )
    # Add baseline human IAA threshold line if provided.
    if human_iaa is not None:
        fig.add_hline(
            y=human_iaa,
            line={"color": "#555555", "dash": "dash", "width": 1.5},
            annotation_text="Human IAA",
            annotation_position="right",
            annotation_font={"size": 14, "color": "#555555"},
        )
    # Format y-axis title by replacing underscores with spaces.
    clean_y_title = metric_col.split("_")[0].capitalize()
    apply_journal_theme(fig, clean_y_title)
    # Disable hover interaction and place title properly above legend.
    fig.update_layout(
        hovermode=False,
        title={
            "text": title,
            "x": 0.5,
            "y": 0.98,
            "xanchor": "center",
            "yanchor": "top",
        }
        if title
        else None,
        bargroupgap=0.08,
        legend={
            "orientation": "h",
            "y": 1.10,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "bottom",
            "font": {"size": 15},
        },
        margin={"r": 100, "t": 140 if title else 80},
    )
    return fig


def display_benchmark_table(
    file_path_or_df: str | pd.DataFrame,
    category: str = "OVERALL_MICRO",
    sort_by: str = "Micro F1",
    *,
    is_ascending: bool = False,
) -> Styler:
    """Filter, format, and display evaluation metrics sorted by a given column.

    Returns
    -------
    Styler
        A pandas Styler object representing the formatted benchmark table.
    """
    raw_df = (
        pd.read_csv(file_path_or_df)
        if isinstance(file_path_or_df, str)
        else file_path_or_df.copy()
    )
    # Filter rows by the requested category and keep the latest run per model.
    cat_df = (
        raw_df[raw_df["category"] == category]
        .drop_duplicates(subset=["Name"], keep="last")
        .copy()
    )
    # Extract Macro F1 values per model to support side-by-side reporting.
    macro_series = (
        raw_df[raw_df["category"] == "OVERALL_MACRO"]
        .drop_duplicates(subset=["Name"], keep="last")
        .set_index("Name")["F1"]
    )
    # Construct unified reporting table with renamed columns.
    table_df = pd.DataFrame(
        {
            "Model": cat_df["Name"],
            "Valid Format (%)": cat_df["Correct_format_(%)"],
            "Hallucinations (%)": cat_df["Hallucinations_(%)"],
            "Precision": cat_df["Precision"],
            "Recall": cat_df["Recall"],
            "Micro F1": cat_df["F1"],
            "Macro F1": cat_df["Name"].map(macro_series),
            "Latency (hh:mm:ss)": cat_df["Inference_time_total_(hh:mm:ss)"],
            "Cost ($)": cat_df["Cost_total_($)"],
        }
    )
    # Sort table by specified column key and apply publication centering styles.
    return (
        table_df.sort_values(by=sort_by, ascending=is_ascending)
        .reset_index(drop=True)
        .style.format(
            {
                "Valid Format (%)": "{:.0f}",
                "Hallucinations (%)": "{:.0f}",
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "Micro F1": "{:.2f}",
                "Macro F1": "{:.2f}",
                "Latency (hh:mm:ss)": "{:s}",
                "Cost ($)": "{:.1f}",
            },
            na_rep="-",
        )
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    )


def plot_benchmark_consensus_strategy(
    df: pd.DataFrame,
    human_iaa: float | None = 0.76,
    metric: str = "F1_with_no_hallucination",
    category: str = "OVERALL_MICRO",
    models_to_keep: list[str] | None = None,
    consensus_temperatures: list[list[float]] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot benchmark comparison between selected solo models and consensus temperature settings."""
    data = df[df["category"] == category].copy()
    # Build clean mapping dictionary for targeted models and consensus runs.
    target_mapping = {}
    if models_to_keep:
        for m in models_to_keep:
            target_mapping[m] = m.split("/")[-1]
    if consensus_temperatures:
        for t_group in consensus_temperatures:
            t_key_suffix = "_t_" + "_".join(str(float(t)) for t in t_group)
            t_label = "Consensus<br>T=" + ", ".join(str(t) for t in t_group)
            # Find matching full consensus run name in dataframe.
            matches = [
                name for name in data["Name"].unique() if name.endswith(t_key_suffix)
            ]
            if matches:
                target_mapping[matches[0]] = t_label
    # Filter dataset strictly to resolved runs while preserving input list order.
    data = data[data["Name"].isin(target_mapping)].copy()
    data["display_name"] = data["Name"].map(target_mapping)
    ordered_labels = [
        target_mapping[k] for k in target_mapping if k in set(data["Name"])
    ]
    # Aggregate duplicate runs and align ordered categories.
    stats = (
        data.groupby("display_name", as_index=False)[metric]
        .mean()
        .set_index("display_name")
        .reindex(ordered_labels)
        .reset_index()
    )
    # Generate distinct palette colors for each column.
    palette = generate_palette(len(stats))
    max_val = stats[metric].round(2).max()
    fig = go.Figure()
    # Add styled bars without showing legend items.
    for (_, row), color in zip(stats.iterrows(), palette, strict=False):
        val = row[metric]
        txt = f"<b>{val:.2f}</b>" if round(val, 2) == max_val else f"{val:.2f}"
        fig.add_bar(
            x=[row["display_name"]],
            y=[val],
            marker={"color": color, "line": {"color": "white", "width": 1}},
            text=[txt],
            textposition="outside",
            textfont={"size": 13},
            showlegend=False,
        )
    # Add human agreement baseline line.
    if human_iaa is not None:
        fig.add_hline(
            y=human_iaa,
            line={"color": "#555555", "dash": "dash", "width": 1.5},
            annotation_text="Human IAA",
            annotation_position="right",
            annotation_font={"size": 14, "color": "#555555"},
        )
    # Apply publication journal theme and layout without legend.
    apply_journal_theme(fig, metric.split("_")[0].capitalize())
    fig.update_layout(
        hovermode=False,
        showlegend=False,
        title={
            "text": title,
            "x": 0.5,
            "y": 0.98,
            "xanchor": "center",
            "yanchor": "top",
        }
        if title
        else None,
        bargroupgap=0.08,
        margin={"r": 100, "t": 80 if title else 40},
    )
    return fig
