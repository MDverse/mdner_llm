"""Module for visualizing LLM performance."""

from itertools import combinations

import pandas as pd
import plotly.graph_objects as go
from pandas.io.formats.style import Styler
from plotly.subplots import make_subplots

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
    without_consensus: bool = True,
) -> Styler:
    """Build and format a centered benchmark summary table for LLMs.

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
    if without_consensus:
        raw_df = raw_df[~raw_df["Name"].str.startswith("consensus_", na=False)]

    cat_df = (
        raw_df[raw_df["category"] == category]
        .drop_duplicates(subset=["Name"], keep="last")
        .copy()
    )
    macro_series = (
        raw_df[raw_df["category"] == "OVERALL_MACRO"]
        .drop_duplicates(subset=["Name"], keep="last")
        .set_index("Name")["F1"]
    )
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


def build_families(
    small_models: list[str],
    frontier_models: list[str],
    consensus_group: str,
    consensus_temperatures: list[float | int],
    consensus_threshold: float,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Build hierarchy of model families.

    Returns
    -------
    list[tuple[str, list[tuple[str, str]]]]
        A list of tuples where each tuple contains a family name and a list of runs
        with their corresponding labels.
    """
    # Temperature labels without trailing zeros (1.0 -> "1")
    temperature_labels = [
        f"{float(temperature):g}" for temperature in consensus_temperatures
    ]
    # Build the small LLM runs with temperature labels for each model
    small_runs = [
        (
            f"{model} (T = {temp})",
            f"{model.split('/')[-1]} (T = {temp})",
        )
        for temp in temperature_labels
        for model in small_models
    ]
    consensus_runs = [
        (
            f"consensus_{consensus_group}_temp_{'_and_'.join(subset)}_th_{consensus_threshold:g}",
            f"Consensus (T = {', '.join(subset)})",
        )
        for size in range(1, len(temperature_labels) + 1)
        for subset in combinations(temperature_labels, size)
    ]
    return [
        ("Small LLM (single run)", small_runs),
        ("Consensus of small LLMs", consensus_runs),
        ("Frontier LLM", [(model, model.split("/")[-1]) for model in frontier_models]),
    ]


def plot_small_llm_consensus_vs_frontier(
    df: pd.DataFrame,
    small_models: list[str],
    frontier_models: list[str],
    consensus_group: str,
    consensus_temperatures: list[float | int | str],
    consensus_threshold: float = 0.5,
    metrics: tuple[str, ...] = (
        "Precision_with_no_hallucination",
        "Recall",
        "F1_with_no_hallucination",
        "Correct_format_(%)",
        "Hallucinations_(%)",
    ),
    category: str = "OVERALL_MICRO",
) -> go.Figure:
    """Plot Small LLMs, Consensus and Frontier LLMs.

    Returns
    -------
    go.Figure
        Formatted Plotly multi-column figure with top legend.
    """
    data = df[df["category"] == category]
    mean_by_name = data.groupby("Name")[list(metrics)].mean()
    # Build the 3 blocks in order: Small -> Consensus -> Frontier
    families = build_families(
        small_models=small_models,
        frontier_models=frontier_models,
        consensus_group=consensus_group,
        consensus_temperatures=consensus_temperatures,
        consensus_threshold=consensus_threshold,
    )
    family_colors = {
        "Small LLM (single run)": "#BDD7EE",
        "Consensus of small LLMs": "#1F77B4",
        "Frontier LLM": "#595959",
    }
    layout_entries, ticktext = [], []
    position = 0.0
    num_small = len(small_models)

    for family, runs in families:
        kept_runs = [
            (name, label) for name, label in runs if name in mean_by_name.index
        ]
        for idx, (name, label) in enumerate(kept_runs):
            layout_entries.append((family, name, position))
            ticktext.append(label)
            position += 1.0
            if (
                family == "Small LLM (single run)"
                and (idx + 1) % num_small == 0
                and (idx + 1) < len(kept_runs)
            ):
                position += 0.5

        position += 0.5  # Gap standard entre les grandes familles
    positions = [entry_position for _, _, entry_position in layout_entries]
    names = [name for _, name, _ in layout_entries]

    # Percent columns only display text, so they are narrower than bar columns
    fig = make_subplots(
        rows=1,
        cols=len(metrics),
        shared_yaxes=True,
        horizontal_spacing=0.04,
        column_widths=[0.45 if "%" in metric else 1.0 for metric in metrics],
        # Two-line titles for percent metrics to avoid overlaps
        subplot_titles=[
            metric.replace("_with_no_hallucination", "")
            .replace("Correct_format_(%)", "Correct format<br>(%)")
            .replace("Hallucinations_(%)", "Hallucinations<br>(%)")
            .replace("_", " ")
            for metric in metrics
        ],
    )
    for col, metric in enumerate(metrics, start=1):
        values = mean_by_name[metric]
        if "%" in metric:
            # Ideal value is 0 for hallucinations and 100 for correct format,
            # anything else is red and bold
            ideal_value = 0 if metric.startswith("Hallucinations") else 100
            texts = [
                f"<span style='color:red'><b>{round(values[name]):.0f}</b></span>"
                if round(values[name]) != ideal_value
                else f"{round(values[name]):.0f}"
                for name in names
            ]
            fig.add_scatter(
                x=[0] * len(positions),
                y=positions,
                mode="text",
                text=texts,
                textfont={"size": 13},
                showlegend=False,
                row=1,
                col=col,
            )
            # No axis line, same row order as the bars
            fig.update_xaxes(range=[-1, 1], visible=False, row=1, col=col)
            fig.update_yaxes(visible=False, autorange="reversed", row=1, col=col)
            continue
        best_value = round(values[names].max(), 2)
        for family, _ in families:
            family_rows = [
                (name, pos) for fam, name, pos in layout_entries if fam == family
            ]
            if not family_rows:
                continue
            family_values = [values[name] for name, _ in family_rows]
            fig.add_bar(
                x=family_values,
                y=[pos for _, pos in family_rows],
                orientation="h",
                marker={"color": family_colors[family]},
                text=[
                    f"<b>{value:.2f}</b>"
                    if round(value, 2) == best_value
                    else f"{value:.2f}"
                    for value in family_values
                ],
                textposition="outside",
                textfont={"size": 13},
                cliponaxis=False,
                width=0.75,
                name=family,
                legendgroup=family,
                showlegend=(col == 1),
                row=1,
                col=col,
            )
        fig.update_xaxes(
            range=[0, 1.15],
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            showline=True,
            linewidth=1,
            linecolor="black",
            showgrid=False,
            row=1,
            col=col,
        )
        # Y axis only on bar columns, labels only on the first one, Small LLMs on top
        fig.update_yaxes(
            tickvals=positions if col == 1 else [],
            ticktext=ticktext if col == 1 else [],
            showticklabels=(col == 1),
            autorange="reversed",
            tickfont={"size": 12},
            showline=True,
            linewidth=1,
            linecolor="black",
            showgrid=False,
            row=1,
            col=col,
        )

    # Uniformisation de la taille et de l'alignement vertical par le bas
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = {"size": 12}
        annotation["y"] = 1.01
        annotation["yanchor"] = "bottom"

    fig.update_layout(
        template="simple_white",
        hovermode=False,
        width=1250,
        height=max(450, int(position * 32 + 150)),
        margin={"l": 230, "r": 30, "t": 105, "b": 40},
        legend={
            "orientation": "h",
            "y": 1.22,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 13},
        },
    )
    return fig


def format_time_label(seconds: float) -> str:
    """Format duration in seconds into human-readable hours and minutes.

    Returns
    -------
    str
        Formatted string representing the duration in hours and minutes.
    """
    hrs = int(seconds // 3600)
    mins = round((seconds % 3600) / 60)
    if hrs == 0:
        return f"{mins}m"
    if mins == 0:
        return f"{hrs}h"
    return f"{hrs}h {mins}m"


def plot_small_llm_consensus_cost_and_time(
    df: pd.DataFrame,
    small_models: list[str],
    frontier_models: list[str],
    consensus_temperatures: list[float | int | str],
    consensus_threshold: float = 0.5,
    consensus_group: str | None = None,
    category: str = "OVERALL_MICRO",
) -> go.Figure:
    """Plot multi-panel horizontal bars comparing total cost and inference time.

    Returns
    -------
    go.Figure
        Formatted Plotly two-column horizontal bar figure with bottom legend.
    """
    data = df[df["category"] == category].copy()
    data = data.sort_values(
        "Inference_time_total_(s)", ascending=False
    ).drop_duplicates(subset=["Name"])

    # Build the 3 blocks in order: Small -> Consensus -> Frontier
    families = build_families(
        small_models=small_models,
        frontier_models=frontier_models,
        consensus_group=consensus_group,
        consensus_temperatures=consensus_temperatures,
        consensus_threshold=consensus_threshold,
    )

    # Palette matching the publication figure
    family_colors = {
        "Small LLM (single run)": "#BDD7EE",  # Light blue
        "Consensus of small LLMs": "#1F77B4",  # Strong blue
        "Frontier LLM": "#595959",  # Dark grey
    }

    layout_entries, tickvals, ticktext = [], [], []
    pos = 0.0
    for family, runs in families:
        kept = [(n, lab) for n, lab in runs if (data["Name"] == n).any()]
        for name, label in kept:
            layout_entries.append((family, name, pos))
            tickvals.append(pos)
            ticktext.append(label)
            pos += 1.0
        pos += 0.5  # Gap between families

    all_names = [name for _, name, _ in layout_entries]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=["Total Cost", "Total Time"],
    )

    # 1. Cost Panel
    cost_values = {
        n: float(data.loc[data["Name"] == n, "Cost_total_($)"].mean())
        for n in all_names
        if not data.loc[data["Name"] == n, "Cost_total_($)"].empty
    }
    max_cost = max(cost_values.values()) if cost_values else 10.0
    for family, _ in families:
        family_rows = [(n, p) for fam, n, p in layout_entries if fam == family]
        if not family_rows:
            continue
        xs = [cost_values.get(n, 0.0) for n, _ in family_rows]
        ys = [p for _, p in family_rows]
        fig.add_bar(
            x=xs,
            y=ys,
            orientation="h",
            marker={"color": family_colors.get(family, "#1F77B4")},
            text=[f"${x:.2f}" for x in xs],
            textposition="outside",
            textfont={"size": 12},
            cliponaxis=False,
            width=0.75,
            name=family,
            legendgroup=family,
            showlegend=True,
            row=1,
            col=1,
        )

    # 2. Time Panel (Values converted to hours)
    time_values_sec = {
        n: float(data.loc[data["Name"] == n, "Inference_time_total_(s)"].mean())
        for n in all_names
        if not data.loc[data["Name"] == n, "Inference_time_total_(s)"].empty
    }
    time_values_hrs = {n: s / 3600.0 for n, s in time_values_sec.items()}
    max_hours = max(time_values_hrs.values()) if time_values_hrs else 10.0
    for family, _ in families:
        family_rows = [(n, p) for fam, n, p in layout_entries if fam == family]
        if not family_rows:
            continue
        xs_hrs = [time_values_hrs.get(n, 0.0) for n, _ in family_rows]
        xs_sec = [time_values_sec.get(n, 0.0) for n, _ in family_rows]
        ys = [p for _, p in family_rows]
        fig.add_bar(
            x=xs_hrs,
            y=ys,
            orientation="h",
            marker={"color": family_colors.get(family, "#1F77B4")},
            text=[format_time_label(s) for s in xs_sec],
            textposition="outside",
            textfont={"size": 12},
            cliponaxis=False,
            width=0.75,
            name=family,
            legendgroup=family,
            showlegend=False,
            row=1,
            col=2,
        )

    # Axes styling
    fig.update_xaxes(
        range=[0, max_cost * 1.3],
        showline=True,
        linewidth=1,
        linecolor="black",
        showgrid=False,
        row=1,
        col=1,
    )
    hour_step = 10 if max_hours > 20 else 2
    hour_ticks = list(range(0, int(max_hours) + hour_step, hour_step))
    fig.update_xaxes(
        range=[0, max_hours * 1.35],
        tickvals=hour_ticks,
        ticktext=[f"{h}h" for h in hour_ticks],
        showline=True,
        linewidth=1,
        linecolor="black",
        showgrid=False,
        row=1,
        col=2,
    )
    for col in (1, 2):
        fig.update_yaxes(
            tickvals=tickvals,
            ticktext=ticktext if col == 1 else [""] * len(tickvals),
            autorange="reversed",
            tickfont={"size": 13},
            showline=True,
            linewidth=1,
            linecolor="black",
            showgrid=False,
            row=1,
            col=col,
        )

    fig.update_layout(
        template="simple_white",
        hovermode=False,
        width=950,
        height=max(400, int(len(tickvals) * 38 + 140)),
        margin={"l": 250, "r": 90, "t": 60, "b": 90},
        legend={
            "orientation": "h",
            "y": -0.15,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 13},
        },
    )
    return fig
