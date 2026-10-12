"""Agreement between LLM annotators, computed from the consensus details CSV."""

import json
from itertools import starmap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mdner_llm.visualization.colors import COLORS

SET_COLUMNS = [
    "groundtruth_by_category",
    "pred_gliner",
    "pred_llm",
    "tp_gliner",
    "tp_llm",
    "fp_gliner",
    "fp_llm",
    "fn_gliner",
    "fn_llm",
]


def normalize_text(text: object) -> str:
    """Normalize an entity text to ignore tiny formatting differences.

    Returns
    -------
    str
        Text with collapsed whitespace and case folded.
    """
    # Example: "  Water   Box " -> "water box"
    return " ".join(str(text).split()).casefold()


def load_ground_truth_keys(path: Path) -> set[tuple[str, str]]:
    """Read a ground truth JSON file.

    Returns
    -------
    set[tuple[str, str]]
        Set of (normalized text, category) pairs.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entities = payload.get("entities")
    if entities is None:
        entities = payload.get("formatted_response", {}).get("entities", [])
    return {(normalize_text(entity["text"]), entity["category"]) for entity in entities}


def filter_by_values(details: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
    """Keep the rows whose column value is in the given values.

    Returns
    -------
    pd.DataFrame
        Filtered details.

    Raises
    ------
    ValueError
        If any value is not present in the column.
    """
    available = set(details[column].unique())
    unknown = set(values) - available
    if unknown:
        msg = f"Unknown {column} {sorted(unknown)}. Available: {sorted(available)}."
        raise ValueError(msg)
    return details[details[column].isin(values)]


def load_filtered_details(
    details_csv_path: str | Path,
    models: list[str] | None,
    temperatures: list[float] | None,
    category: str | None,
) -> pd.DataFrame:
    """Load the details CSV and filter it by model, temperature and category.

    Returns
    -------
    pd.DataFrame
        Filtered details.

    Raises
    ------
    ValueError
        If no row is left after filtering.
    """
    details = pd.read_csv(details_csv_path)
    details["found_by_model"] = (
        details["found_by_model"].astype(str).str.lower().eq("true")
    )
    # Filter by model and temperature when specified
    if models is not None:
        details = filter_by_values(details, "model_name", models)
    if temperatures is not None:
        details = filter_by_values(details, "temperature", temperatures)
    # Filter by category unless it is the overall micro score
    if category not in (None, "OVERALL_MICRO"):
        details = details[details["category"] == category]
    if details.empty:
        msg = "No row left after filtering, check models, temperatures and category."
        raise ValueError(msg)
    return details


def add_annotator_columns(details: pd.DataFrame) -> pd.DataFrame:
    """Add the annotator name and the normalized entity key.

    Returns
    -------
    pd.DataFrame
        Copy of details with annotator and entity_key columns.
    """
    details = details.copy()
    # Add the temperature to the name when several are compared: "gpt-4o (T=0.5)"
    if (details["temperature"] != details["temperature"].iloc[0]).any():
        details["annotator"] = (
            details["model_name"] + " (T=" + details["temperature"].astype(str) + ")"
        )
    else:
        details["annotator"] = details["model_name"]
    # Entity key: (input file, normalized text, category)
    details["entity_key"] = [
        (path, normalize_text(text), category)
        for path, text, category in zip(
            details["input_json_path"],
            details["entity_predicted_by_model"],
            details["category"],
            strict=True,
        )
    ]
    return details


def flag_true_positives(
    details: pd.DataFrame, base_dir: str | Path | None
) -> pd.DataFrame:
    """Flag the predicted entities that are present in the ground truth.

    Returns
    -------
    pd.DataFrame
        Details with an is_true_positive column.

    Raises
    ------
    ValueError
        If input_json_path is missing, true positives cannot be computed.
    """
    if details["input_json_path"].isna().any():
        msg = "input_json_path is missing, true positives cannot be computed."
        raise ValueError(msg)
    base_path = Path(base_dir) if base_dir else Path()
    # Load the ground truth of each input file once
    ground_truth = {
        path: load_ground_truth_keys(base_path / path.lstrip("/"))
        for path in details["input_json_path"].unique()
    }
    # Compare (text, category) with the ground truth of the same input file
    details["is_true_positive"] = [
        (text, category) in ground_truth[path]
        for path, text, category in details["entity_key"]
    ]
    return details


def compute_pairwise_agreement(
    entity_sets: pd.Series, annotators: list[str]
) -> pd.DataFrame:
    """Compute the agreement (%) between each pair of annotators.

    Returns
    -------
    pd.DataFrame
        Square matrix of intersection over union percentages.
    """
    annotator_sets = {name: entity_sets.get(name, set()) for name in annotators}
    matrix = pd.DataFrame(np.nan, index=annotators, columns=annotators)
    for row_name in annotators:
        for column_name in annotators:
            union = annotator_sets[row_name] | annotator_sets[column_name]
            # Agreement stays NaN when both annotators found nothing
            if union:
                shared = annotator_sets[row_name] & annotator_sets[column_name]
                matrix.loc[row_name, column_name] = 100 * len(shared) / len(union)
    return matrix


def compute_agreement_matrix(
    details_csv_path: str | Path,
    models: list[str] | None = None,
    temperatures: list[float] | None = None,
    category: str | None = None,
    *,
    only_true_positives: bool = False,
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compute the pairwise agreement (%) between annotators.

    Returns
    -------
    pd.DataFrame
        Square matrix indexed by annotator (model and temperature).
    """
    details = load_filtered_details(details_csv_path, models, temperatures, category)
    details = add_annotator_columns(details)
    if only_true_positives:
        details = flag_true_positives(details, base_dir)
    # Keep the entities actually predicted by each model
    voted = details[details["found_by_model"]]
    if only_true_positives:
        voted = voted[voted["is_true_positive"]]
    entity_sets = voted.groupby("annotator")["entity_key"].agg(set)
    # List annotators, even those without any prediction
    annotators = (
        details.drop_duplicates(["model_name", "temperature"])
        .sort_values(["model_name", "temperature"])["annotator"]
        .tolist()
    )
    return compute_pairwise_agreement(entity_sets, annotators)


def plot_model_agreement_heatmap(
    details_csv_path: str | Path,
    models: list[str] | None = None,
    temperatures: list[float] | None = None,
    category: str | None = None,
    *,
    only_true_positives: bool = False,
    base_dir: str | Path | None = None,
) -> go.Figure:
    """Plot the pairwise agreement (%) between annotators as a heatmap.

    Returns
    -------
    go.Figure
        Heatmap without title.
    """
    matrix = compute_agreement_matrix(
        details_csv_path,
        models=models,
        temperatures=temperatures,
        category=category,
        only_true_positives=only_true_positives,
        base_dir=base_dir,
    )
    # Greys by default, else category color (pastel at 0%, dark at 100%)
    colorscale = (
        [[0.0, COLORS[category]], [1.0, "#20242c"]] if category in COLORS else "Greys"
    )
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            zmin=0,
            zmax=100,
            colorscale=colorscale,
            texttemplate="%{z:.0f}",
            textfont={"size": 13},
            colorbar={"title": {"text": "Agreement (%)"}},
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-30)
    # Figure size grows with the number of annotators
    side = min(max(80 * len(matrix) + 280, 480), 1000)
    fig.update_layout(
        template="simple_white",
        font={"size": 13},
        title=None,
        margin={"t": 30, "b": 60, "l": 80, "r": 60},
        width=side + 120,
        height=side,
    )
    return fig


def select_and_rename_columns(
    df_model: pd.DataFrame, text_id_col: str, suffix: str
) -> pd.DataFrame:
    """Select the evaluation columns of a model and suffix them.

    Returns
    -------
    pd.DataFrame
        Evaluation columns renamed with the model suffix.
    """
    columns = [
        text_id_col,
        "category",
        "groundtruth_by_category",
        "prediction_by_category",
        "tp_entities",
        "fp_entities",
        "fn_entities",
    ]
    return df_model[columns].rename(
        columns={
            "prediction_by_category": f"pred_{suffix}",
            "tp_entities": f"tp_{suffix}",
            "fp_entities": f"fp_{suffix}",
            "fn_entities": f"fn_{suffix}",
        }
    )


def prepare_aligned_data(
    llm_parquet_path: Path | str,
    gliner_parquet_path: Path | str,
    gliner_model_name: str,
    best_llm_name: str,
) -> pd.DataFrame:
    """Load and align the results of both models by text and category.

    Returns
    -------
    pd.DataFrame
        One row per text and category, with entity sets of both models.
    """
    df_llm = pd.read_parquet(llm_parquet_path)
    df_gliner = pd.read_parquet(gliner_parquet_path)
    # Keep the selected models
    df_llm = df_llm[df_llm["model_name"] == best_llm_name]
    mask_gliner = df_gliner["model_name"].str.contains(gliner_model_name, regex=False)
    df_gliner = df_gliner[mask_gliner]
    # Select and rename columns before merging
    text_id_col = (
        "response_metadata" if "response_metadata" in df_llm.columns else "text"
    )
    llm_columns = select_and_rename_columns(df_llm, text_id_col, "llm")
    gliner_columns = select_and_rename_columns(df_gliner, text_id_col, "gliner")
    # Outer merge to keep categories seen by either model
    merged = gliner_columns.merge(
        llm_columns.drop(columns=["groundtruth_by_category"]),
        on=[text_id_col, "category"],
        how="outer",
    )
    # Replace missing values by empty sets
    for column in SET_COLUMNS:
        merged[column] = merged[column].apply(
            lambda value: (
                set(value) if isinstance(value, set | list | np.ndarray) else set()
            )
        )
    merged["text_id"] = merged[text_id_col]
    return merged


def get_classes_with_entities(group: pd.DataFrame, column: str) -> set[str]:
    """Get the classes having at least one entity in a column.

    Returns
    -------
    set[str]
        Class names.
    """
    return set(group.loc[group[column].apply(len) > 0, "category"])


def analyze_text_level_class_agreement(
    df_aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Evaluate whether both models detect the same classes in each text.

    Returns
    -------
    tuple[pd.DataFrame, float]
        Agreement per text and percentage of texts with identical class sets.
    """
    records = []
    for text_id, group in df_aligned.groupby("text_id"):
        # Classes where each source has at least one entity
        gliner_classes = get_classes_with_entities(group, "pred_gliner")
        llm_classes = get_classes_with_entities(group, "pred_llm")
        gt_classes = get_classes_with_entities(group, "groundtruth_by_category")
        union = gliner_classes | llm_classes
        jaccard = len(gliner_classes & llm_classes) / len(union) if union else 1.0
        records.append(
            {
                "text_id": text_id,
                "nb_classes_gt": len(gt_classes),
                "nb_classes_gliner": len(gliner_classes),
                "nb_classes_llm": len(llm_classes),
                "classes_count_diff": len(gliner_classes) - len(llm_classes),
                "same_class_count": len(gliner_classes) == len(llm_classes),
                "exact_class_set_match": gliner_classes == llm_classes,
                "jaccard_classes": jaccard,
            }
        )
    df_text_agreement = pd.DataFrame(records)
    pct_same_count = df_text_agreement["same_class_count"].mean() * 100
    pct_exact_match = df_text_agreement["exact_class_set_match"].mean() * 100
    mean_jaccard = df_text_agreement["jaccard_classes"].mean()
    # Print the global summary
    print("=== Concordance globale au niveau Texte ===")
    print(f"Même nombre de classes détectées : {pct_same_count:.1f}% des textes")
    print(f"Ensemble exact de classes identique : {pct_exact_match:.1f}% des textes")
    print(f"Similarité moyenne de Jaccard des classes : {mean_jaccard:.3f}")
    return df_text_agreement, pct_exact_match


def summarize_category_overlap(category: str, group: pd.DataFrame) -> dict:
    """Summarize the overlap of both models for one category.

    Returns
    -------
    dict
        Shared and exclusive TP and FP counts, and recalls.
    """
    total_gt = sum(len(entities) for entities in group["groundtruth_by_category"])
    # Avoid a division by zero when the category has no ground truth
    denominator = max(total_gt, 1)
    # Entities found by each model
    gliner_tp = set().union(*group["tp_gliner"])
    llm_tp = set().union(*group["tp_llm"])
    gliner_fp = set().union(*group["fp_gliner"])
    llm_fp = set().union(*group["fp_llm"])
    union_tp = gliner_tp | llm_tp
    return {
        "category": category,
        "total_groundtruth": total_gt,
        "both_correct (TP ∩ TP)": len(gliner_tp & llm_tp),
        "gliner_unique_correct": len(gliner_tp - llm_tp),
        "llm_unique_correct": len(llm_tp - gliner_tp),
        "missed_by_both (FN)": max(0, total_gt - len(union_tp)),
        "shared_hallucinations (FP)": len(gliner_fp & llm_fp),
        "gliner_only_fp": len(gliner_fp - llm_fp),
        "llm_only_fp": len(llm_fp - gliner_fp),
        "gliner_recall": len(gliner_tp) / denominator,
        "llm_recall": len(llm_tp) / denominator,
        "ensemble_oracle_recall": len(union_tp) / denominator,
    }


def compute_detailed_category_overlap(df_aligned: pd.DataFrame) -> pd.DataFrame:
    """Compute the overlap and complementarity of both models per category.

    Returns
    -------
    pd.DataFrame
        One row per category, sorted by ground truth size.
    """
    summaries = list(
        starmap(summarize_category_overlap, df_aligned.groupby("category"))
    )
    return pd.DataFrame(summaries).sort_values(by="total_groundtruth", ascending=False)


def plot_complementarity_profile(
    df_cat_summary: pd.DataFrame,
    gliner_label: str,
    llm_label: str,
    output_fig_path: Path | str | None = None,
) -> go.Figure:
    """Plot the share of correct entities found by each model per class.

    Returns
    -------
    go.Figure
        Stacked horizontal bars in % of ground truth entities.
    """
    df_cat = df_cat_summary.set_index("category")
    # Bar segments: name -> (column, color)
    segments = {
        "Both Correct": ("both_correct (TP ∩ TP)", "#2CA02C"),
        f"Only {gliner_label} Correct": ("gliner_unique_correct", "#1F77B4"),
        f"Only {llm_label} Correct": ("llm_unique_correct", "#FF7F0E"),
        "Missed by Both": ("missed_by_both (FN)", "#D62728"),
    }
    fig = go.Figure()
    for name, (column, color) in segments.items():
        fig.add_bar(
            x=100 * df_cat[column] / df_cat["total_groundtruth"],
            y=df_cat.index,
            name=name,
            orientation="h",
            marker_color=color,
        )
    # Stack bars, put the legend on top
    fig.update_layout(
        barmode="stack",
        template="simple_white",
        xaxis={"title": "% of Ground Truth Entities", "range": [0, 100]},
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.02},
        width=900,
        height=550,
    )
    if output_fig_path:
        Path(output_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(output_fig_path, scale=3)
    return fig
