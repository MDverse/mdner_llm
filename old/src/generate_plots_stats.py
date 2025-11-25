# generate_plots_stats.py

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger


# ======================================================================================
# Configuration
# ======================================================================================

DATE_TIME_STR = input(
    "Enter the DATE_TIME_STR and time string to make plots (YYYY-MM-DD_HH-MM-SS): "
)

BASE_DIR = Path("llm_outputs")

MODEL_ORDER = [
    "gpt-4.1-2025-04-14",
    # "gpt-4.1-mini-2025-04-14",
    # "gpt-4.1-nano-2025-04-14",
    # "gpt-4o-2024-11-20",
    # "o4-mini-2025-04-16",
    "o3-2025-04-16",
    "o3-mini-2025-01-31",
]

# MODEL_ORDER = [
#     "qwen-qwq-32b",
#     "meta-llama/llama-4-maverick-17b-128e-instruct",
#     "deepseek-r1-distill-llama-70b",
#     "mistral-saba-24b",
#     "gemma2-9b-it",
#     # "llama-3.3-70b-versatile",
# ]

PROMPT_ORDER = [
    "zero_shot",
    "one_shot",
    "few_shot",
]

TAGS = ["MOL", "SOFTN", "SOFTV", "STIME", "TEMP", "FFM"]

# Plots to build - set to True to build the plot, False to skip
# ---------------------------------------------------------------------------
# This plots the amount of LLM responses that have at least one entity verified
# from the input text.
ONE_ENTITY_VERIF = True # requires `quality_control_results.csv`
# # Plots the amount of LLM responses where the output text
# # is the same as the input text. ONLY USED IN XML-STYLE OUTPUTS.
# TEXT_UNCHANGED = True # requires `quality_control_results.csv`
# Plots the average precision of LLM responses across different models and prompts.
PRECISION = True # requires `scoring_results.csv`
# Plots the average recall of LLM responses across different models and prompts
RECALL = True # requires `scoring_results.csv`
# Plots the entity existence percentages by prompt within each model.
VALIDITY = True # requires `quality_control_results.csv`
# Plots the contingency metrics for each entity type.
ENTITY_CONTINGENCY = True # requires `scoring_results.csv`


# ======================================================================================
# Plotting functions
# ======================================================================================


def plot_one_entity_verified(qc_df: pd.DataFrame, out_path: Path) -> None:
    """Plots the count of LLM responses that have at least one entity verified
    from the input text.

    Args:
        qc_df (pd.DataFrame): DataFrame containing quality control results
        out_path (Path): Output path for the plot image
    """
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(
        data=qc_df[qc_df["one_entity_verified"]],
        x="prompt",
        hue="model",
        palette="viridis",
        order=PROMPT_ORDER,
        hue_order=MODEL_ORDER,
    )
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", padding=5)
    plt.title("LLM responses with ≥1 entity from the input text (100 texts)")
    plt.xlabel("Prompt")
    plt.ylabel("Responses with one entity verified")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved entity‑verified count plot → {out_path}")


# def plot_text_unchanged(qc_df: pd.DataFrame, out_path: Path) -> None:
#     """Plots the count of LLM responses where the output text
#     is the same as the input text.

#     Args:
#         qc_df (pd.DataFrame): DataFrame containing quality control results
#         out_path (Path): Output path for the plot image
#     """
#     combos = pd.MultiIndex.from_product(
#         [qc_df["prompt"].unique(), qc_df["model"].unique()],
#         names=["prompt", "model"],
#     ).to_frame(index=False)
#     counts = (
#         qc_df[qc_df["text_unchanged"]]
#         .groupby(["prompt", "model"])
#         .size()
#         .reset_index(name="count")
#     )
#     df = pd.merge(combos, counts, on=["prompt", "model"], how="left").fillna(0)
#     df["count"] = df["count"].astype(int)

#     plt.figure(figsize=(12, 8))
#     ax = sns.barplot(
#         data=df,
#         x="prompt",
#         y="count",
#         hue="model",
#         palette="viridis",
#         order=PROMPT_ORDER,
#         hue_order=MODEL_ORDER,
#     )
#     for c in ax.containers:
#         ax.bar_label(c, fmt="%.0f", padding=5)
#     plt.title("LLM responses where output text equals input text (100 texts)")
#     plt.xlabel("Prompt")
#     plt.ylabel("Unchanged responses")
#     plt.legend(title="Model")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     logger.info(f"Saved unchanged‑text plot → {out_path}")


def plot_precision(scoring_df: pd.DataFrame, out_path: Path) -> None:
    """Plots the average precision of LLM responses
    across different models and prompts.

    Args:
        scoring_df (pd.DataFrame): DataFrame containing scoring results
        out_path (Path): Output path for the precision plot image
    """
    agg = (
        scoring_df.groupby(["model", "prompt"])[["total_correct", "total_fp"]]
        .sum()
        .reset_index()
    )
    agg["precision"] = agg["total_correct"] / (agg["total_correct"] + agg["total_fp"])
    agg["model"] = pd.Categorical(agg["model"], MODEL_ORDER, ordered=True)
    agg["prompt"] = pd.Categorical(agg["prompt"], PROMPT_ORDER, ordered=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=agg,
        x="prompt",
        y="precision",
        hue="model",
        palette="viridis",
        order=(PROMPT_ORDER),
        hue_order=MODEL_ORDER,
        legend=True,
    )
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=5)
    plt.title("Average Precision by Model and Prompt")
    plt.xlabel("Prompt")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved precision plot → {out_path}")


def plot_recall(scoring_df: pd.DataFrame, out_path: Path) -> None:
    """Plots the average recall of LLM responses

    Args:
        scoring_df (pd.DataFrame): DataFrame containing scoring results
        out_path (Path): Output path for the recall plot image
    """
    agg = (
        scoring_df.groupby(["model", "prompt"])[["total_correct", "total"]]
        .sum()
        .reset_index()
    )
    agg["recall"] = agg["total_correct"] / agg["total"]
    agg["model"] = pd.Categorical(agg["model"], MODEL_ORDER, ordered=True)
    agg["prompt"] = pd.Categorical(agg["prompt"], PROMPT_ORDER, ordered=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=agg,
        x="prompt",
        y="recall",
        hue="model",
        palette="viridis",
        order=PROMPT_ORDER,
        hue_order=MODEL_ORDER,
        legend=True,
    )
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=5)
    plt.title("Average Recall by Model and Prompt")
    plt.xlabel("Prompt")
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved recall plot → {out_path}")


def plot_entity_contingency(df: pd.DataFrame, entity: str, out_path: Path) -> None:
    """Plots the contingency metrics for a specific entity type.

    Args:
        df (pd.DataFrame): DataFrame containing scoring results
        entity (str): Entity name to plot contingency for (e.g., "MOL", "SOFTN")
        out_path (Path): Output path for the contingency plot image
    """
    out_path = out_path / f"contingency_{entity}.png"

    correct_col = f"{entity}_correct"
    total_col = f"{entity}_total"
    fp_col = f"{entity}_FP"
    fn_col = f"{entity}_FN"

    # Parse FP / FN semicolon lists → counts
    df[f"{entity}_FP_count"] = (
        df[fp_col]
        .fillna("")
        .apply(lambda x: len([v for v in x.split(";") if v.strip()]))
    )
    df[f"{entity}_FN_count"] = (
        df[fn_col]
        .fillna("")
        .apply(lambda x: len([v for v in x.split(";") if v.strip()]))
    )
    df[f"{entity}_pred_total"] = df[correct_col] + df[f"{entity}_FP_count"]

    plot_df = pd.DataFrame(
        {
            "True Positives": df[correct_col],
            "False Positives": df[f"{entity}_FP_count"],
            "False Negatives": df[f"{entity}_FN_count"],
            "Ground Truth Total": df[total_col],
            "Predicted Total": df[f"{entity}_pred_total"],
        }
    )

    agg = plot_df.sum().reset_index()
    agg.columns = ["Metric", "Count"]

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=agg, x="Metric", y="Count", hue="Metric", palette="viridis")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", padding=5)
    plt.title(f"Contingency Metrics for '{entity}' Entity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved {entity} contingency plot → {out_path}")


def plot_validity(qc_df: pd.DataFrame, out_path: Path) -> None:
    """Plots the entity existence percentages by prompt within each model.

    Args:
        qc_df (pd.DataFrame): DataFrame containing quality control results
        out_path (Path): Output path for the validity plot image
    """
    # 1) aggregate by model & prompt
    agg = (
        qc_df.groupby(["prompt", "model"])[
            ["fully_valid", "partially_valid", "invalid"]
        ]
        .sum()
        .reset_index()
    )

    # ---- collapse partially_valid + invalid → invalid ------------------- #
    agg["existing"] = agg["fully_valid"]
    agg["hallucinated"] = agg["partially_valid"] + agg["invalid"]
    agg = agg.drop(columns="partially_valid")  # keep only fully_valid & invalid

    # Calculate percentages
    agg["total"] = agg["existing"] + agg["hallucinated"]
    agg["existing"] = (agg["existing"] / agg["total"]) * 100
    agg["hallucinated"] = (agg["hallucinated"] / agg["total"]) * 100

    # preserve order
    agg["model"] = pd.Categorical(agg["model"], MODEL_ORDER, ordered=True)
    agg["prompt"] = pd.Categorical(agg["prompt"], PROMPT_ORDER, ordered=True)

    # 2) long form for seaborn
    melted = agg.melt(
        id_vars=["prompt", "model"],
        value_vars=["existing", "hallucinated"],
        var_name="Existing status",
        value_name="Percentage",
    )

    # 3) bar plot: x=model, hue=prompt, facet by Validation
    g = sns.catplot(
        data=melted,
        kind="bar",
        x="prompt",
        y="Percentage",
        hue="model",
        col="Existing status",
        order=PROMPT_ORDER,
        palette="viridis",
        errorbar=None,
        height=6,
        aspect=1,
        sharey=False,
    )

    g.set_axis_labels("Prompt", "Entity Percentage")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        for c in ax.containers:
            ax.bar_label(c, fmt="%.1f%%", padding=2, fontsize=9)

    g.figure.suptitle(
        "Entity existence percentages by prompt within each model", y=1.03
    )
    g.tight_layout()
    g.savefig(out_path, dpi=300)
    plt.close(g.figure)
    logger.info(f"Saved validity plot → {out_path}")


# ======================================================================================
# Main logic
# ======================================================================================


def main():
    stats_dir = BASE_DIR / DATE_TIME_STR / "stats/"
    images_dir = BASE_DIR / DATE_TIME_STR / "images"

    logger.info(f"Working directory: {Path.cwd()}")

    scoring_path = stats_dir / "scoring_results.csv"
    qc_path = stats_dir / "quality_control_results.csv"

    if not scoring_path.exists() or not qc_path.exists():
        logger.error(f"full path is {scoring_path}")
        logger.error(f"CSV files not found in {stats_dir}")
        raise SystemExit(1)

    logger.info("Reading CSV files …")
    scoring_df = pd.read_csv(scoring_path)
    qc_df = pd.read_csv(qc_path)

    if ONE_ENTITY_VERIF:
        logger.info("Plotting entity‑verified count …")
        plot_one_entity_verified(qc_df, images_dir / "one_entity_verified_count.png")
    # if TEXT_UNCHANGED:    # For XML-style outputs only
    #     logger.info("Plotting unchanged‑text count …")
    #     plot_text_unchanged(qc_df, images_dir / "text_unchanged_count.png")
    if PRECISION:
        logger.info("Plotting precision …")
        plot_precision(scoring_df, images_dir / "precision.png")
    if RECALL:
        logger.info("Plotting recall …")
        plot_recall(scoring_df, images_dir / "recall.png")
    if VALIDITY:
        logger.info("Plotting validity …")
        plot_validity(qc_df, images_dir / "validity.png")
    if ENTITY_CONTINGENCY:
        logger.info("Plotting entity contingency …")
        for entity in TAGS:
            plot_entity_contingency(scoring_df, entity, images_dir)

    logger.success(f"Plots saved at {images_dir.resolve()}")


if __name__ == "__main__":
    main()
