"""Compare fine-grained NER error distributions across two models.

The module classifies entity-level errors within each document, aggregates
their counts by entity category, and plots the resulting percentage
distributions for a GLiNER model and an LLM.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Display order for error types in the output tables and stacked bars.
ERROR_TYPES = [
    "Missed (FN)",
    "Wrong boundary",
    "Wrong type",
    "Wrong (FP)",
    "Hallucination (Not in text)",
]

# One color per error type, in the same order as ERROR_TYPES.
PALETTE = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#990000"]


def matches_boundary(entity: str, mapping: dict[str, str]) -> bool:
    """Check whether an entity has a non-exact substring match in a mapping.

    This is used as a simple proxy for a boundary error: one entity string
    contains the other, but the strings are not identical.

    Parameters
    ----------
    entity : str
        Entity string to check.
    mapping : dict[str, str]
        Entity strings mapped to their categories. Only the keys are checked.

    Returns
    -------
    bool
        True if the entity partially matches at least one mapped string.
    """
    return any(
        (entity in target or target in entity) and entity != target
        for target in mapping
    )


def classify_text_errors(
    group: pd.DataFrame,
) -> list[dict[str, str | int]]:
    """Count error types for each category in one document.

    Entity strings from all rows in the document are mapped to their true
    and predicted categories. These document-level mappings help distinguish
    missed entities and wrong-type predictions from other errors.

    Parameters
    ----------
    group : pd.DataFrame
        Rows belonging to a single document. Each row represents a category
        and provides its false-negative entities, false-positive entities,
        and hallucinated entities.

    Returns
    -------
    list[dict[str, str | int]]
        One record per input row, containing its category and error counts.
    """
    truth_map = {
        token: row["category"]
        for _, row in group.iterrows()
        for token in row.get("groundtruth_by_category", [])
    }
    prediction_map = {
        token: row["category"]
        for _, row in group.iterrows()
        for token in row.get("prediction_by_category", [])
    }

    records = []
    for _, row in group.iterrows():
        category = row["category"]
        fn_tokens = set(row.get("fn_entities", []))
        fp_tokens = set(row.get("fp_entities", []))
        hallucinations = set(row.get("hallucinated_by_category", []))

        # A false negative is "missed" only if no related prediction exists.
        pure_fn = sum(
            1
            for token in fn_tokens
            if not matches_boundary(token, prediction_map)
            and not (token in prediction_map and prediction_map[token] != category)
        )

        # Classify false positives by comparing them with ground-truth
        # entities and the document's hallucination annotations.
        wrong_boundary = sum(
            1 for token in fp_tokens if matches_boundary(token, truth_map)
        )
        wrong_type = sum(
            1
            for token in fp_tokens
            if token in truth_map and truth_map[token] != category
        )
        spurious_fp = sum(
            1
            for token in fp_tokens
            if token not in truth_map
            and token not in hallucinations
            and not matches_boundary(token, truth_map)
        )
        hallucinated_fp = sum(
            1
            for token in fp_tokens
            if token in hallucinations
            and token not in truth_map
            and not matches_boundary(token, truth_map)
        )

        records.append(
            {
                "category": category,
                "Missed (FN)": pure_fn,
                "Wrong boundary": wrong_boundary,
                "Wrong type": wrong_type,
                "Wrong (FP)": spurious_fp,
                "Hallucination (Not in text)": hallucinated_fp,
            }
        )

    return records


def compute_model_error_percentages(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the error-type distribution for each entity category.

    Rows are grouped by document, using ``response_metadata`` when present
    and ``text`` otherwise. Error counts are summed across documents, then
    normalized so that each category's error types total 100%. Categories
    with no counted errors receive 0% for every error type.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Evaluation rows for one model.

    Returns
    -------
    pd.DataFrame
        Percentage breakdown indexed by entity category, with columns in
        ``ERROR_TYPES`` order.
    """
    id_col = "response_metadata" if "response_metadata" in dataframe.columns else "text"

    records = [
        record
        for _, group in dataframe.groupby(id_col)
        for record in classify_text_errors(group)
    ]

    error_counts = pd.DataFrame(records).groupby("category")[ERROR_TYPES].sum()
    totals = error_counts.sum(axis=1).replace(0, float("nan"))

    return error_counts.div(totals, axis=0).mul(100).fillna(0)


def render_error_profile_bars(
    axes: list[plt.Axes],
    dataframes: list[pd.DataFrame],
    model_names: list[str],
) -> None:
    """Draw one horizontal stacked error-distribution chart per model.

    Parameters
    ----------
    axes : list[plt.Axes]
        Matplotlib axes on which to draw the charts.
    dataframes : list[pd.DataFrame]
        Category-level percentage tables, one per model.
    model_names : list[str]
        Model names used as subplot titles.
    """
    for axis, data, title, label in zip(
        axes, dataframes, model_names, ["a", "b"], strict=False
    ):
        offset = pd.Series(0.0, index=data.index)

        for error_type, color in zip(ERROR_TYPES, PALETTE, strict=False):
            values = data[error_type]
            axis.barh(
                data.index,
                values,
                left=offset,
                color=color,
                label=error_type,
                height=0.6,
            )
            offset += values

        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.text(
            -0.08,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
        )
        axis.set_xlim(0, 100)
        axis.set_xlabel("Share of errors (%)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)


def plot_error_profile(
    llm_parquet_path: Path | str,
    gliner_parquet_path: Path | str,
    gliner_name: str,
    llm_name: str,
    output_fig_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare GLiNER and LLM error distributions by entity category.

    The function loads both evaluation datasets, selects the requested
    models, computes their error percentages, and plots categories shared
    by both models. If an output path is supplied, it also saves the figure.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        GLiNER and LLM percentage tables, restricted to their shared
        categories and ordered identically.

    Raises
    ------
    ValueError
        If no rows remain for either selected model.
    """
    df_llm = pd.read_parquet(llm_parquet_path)
    df_gliner = pd.read_parquet(gliner_parquet_path)

    # LLM names must match exactly; GLiNER names need only contain the
    # requested identifier.
    df_llm = df_llm[df_llm["model_name"] == llm_name]
    gliner_mask = df_gliner["model_name"].str.contains(gliner_name, regex=False)
    df_gliner = df_gliner[gliner_mask]

    if df_llm.empty or df_gliner.empty:
        missing = llm_name if df_llm.empty else gliner_name
        msg = f"No records found for model: '{missing}'"
        raise ValueError(msg)

    gliner_pct = compute_model_error_percentages(df_gliner)
    llm_pct = compute_model_error_percentages(df_llm)

    # Both panels must display the same categories in the same order.
    shared_categories = sorted(set(gliner_pct.index) & set(llm_pct.index))
    gliner_pct = gliner_pct.loc[shared_categories]
    llm_pct = llm_pct.loc[shared_categories]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    render_error_profile_bars(
        axes,
        [gliner_pct, llm_pct],
        [gliner_name, llm_name],
    )
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        frameon=False,
    )
    plt.tight_layout()

    if output_fig_path:
        Path(output_fig_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_fig_path, bbox_inches="tight", dpi=300)

    plt.show()
    return gliner_pct, llm_pct


def extract_error_examples(
    parquet_path: Path | str,
    model_name: str,
    samples_per_type: int = 3,
) -> pd.DataFrame:
    """Extract concrete entity examples for each fine-grained error category.

    Returns
    -------
    pd.DataFrame
        Table containing sample entities, document identifiers,
        categories, and error types.
    """
    dataframe = pd.read_parquet(parquet_path)
    mask = dataframe["model_name"].str.contains(model_name, regex=False)
    dataframe = dataframe[mask]

    id_col = "response_metadata" if "response_metadata" in dataframe.columns else "text"
    collected_examples = []

    for document_id, group in dataframe.groupby(id_col):
        truth_map = {
            token: row["category"]
            for _, row in group.iterrows()
            for token in row.get("groundtruth_by_category", [])
        }
        pred_map = {
            token: row["category"]
            for _, row in group.iterrows()
            for token in row.get("prediction_by_category", [])
        }

        for _, row in group.iterrows():
            category = row["category"]
            false_negatives = set(row.get("fn_entities", []))
            false_positives = set(row.get("fp_entities", []))
            hallucinations = set(row.get("hallucinated_by_category", []))

            # 1. Missed (FN)
            for token in false_negatives:
                is_boundary = any(
                    (token in pred or pred in token) and token != pred
                    for pred in pred_map
                )
                is_type = token in pred_map and pred_map[token] != category
                if not is_boundary and not is_type:
                    collected_examples.append(
                        {
                            "error_type": "Missed (FN)",
                            "category": category,
                            "entity": token,
                            "document": document_id,
                            "context_notes": "Entity completely omitted by model",
                        }
                    )

            # 2. False Positives Breakdown
            for token in false_positives:
                matching_boundary = [
                    target
                    for target in truth_map
                    if (token in target or target in token) and token != target
                ]
                if matching_boundary:
                    collected_examples.append(
                        {
                            "error_type": "Wrong boundary",
                            "category": category,
                            "entity": token,
                            "document": document_id,
                            "context_notes": "Overlaps with ground truth "
                            f"'{matching_boundary[0]}'",
                        }
                    )
                elif token in truth_map and truth_map[token] != category:
                    collected_examples.append(
                        {
                            "error_type": "Wrong type",
                            "category": category,
                            "entity": token,
                            "document": document_id,
                            "context_notes": f"True category is '{truth_map[token]}'",
                        }
                    )
                elif token in hallucinations and token not in truth_map:
                    collected_examples.append(
                        {
                            "error_type": "Hallucination (Not in text)",
                            "category": category,
                            "entity": token,
                            "document": document_id,
                            "context_notes": "Token absent from input raw text",
                        }
                    )
                elif token not in truth_map:
                    collected_examples.append(
                        {
                            "error_type": "Wrong (FP)",
                            "category": category,
                            "entity": token,
                            "document": document_id,
                            "context_notes": "Present in text but not an entity",
                        }
                    )

    df_examples = pd.DataFrame(collected_examples)
    return (
        df_examples.groupby("error_type")
        .apply(lambda group: group.head(samples_per_type))
        .reset_index(drop=True)
    )
