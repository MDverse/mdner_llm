# Snakefile for training and evaluating GLiNER models.

import shutil
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import UTC, datetime

from mdner_llm.gliner.train_gliner import load_config, train_all_folds
from mdner_llm.logger import create_logger

# Global paths.
CONFIG_MODELS_PATH = Path("workflow/configs/gliner_models.yaml")
GROUNDTRUTH_DIR = Path("data/groundtruth")
FFM_DB_PATH = Path("data/normalization/md_forcefields_registry.json")
SOFTNAME_DB_PATH = Path("data/normalization/software_names_registry.json")

# Evaluation directory with timestamp.
EVAL_DATE = datetime.now(UTC).strftime("%Y-%m-%d")
EVAL_DIR = f"results/gliner/evaluation/{EVAL_DATE}"

# Load models specification from YAML.
with open(CONFIG_MODELS_PATH, "r", encoding="utf-8") as file_stream:
    MODELS = yaml.safe_load(file_stream)

ALL_MODELS = list(MODELS.keys())
TRAINABLE_MODELS = [model for model, meta in MODELS.items() if meta.get("is_trainable", False)]
ZEROSHOT_MODELS = [model for model, meta in MODELS.items() if not meta.get("is_trainable", False)]


# Load number of cross-validation folds from each model's training config.
MODEL_FOLDS = {}
for model in TRAINABLE_MODELS:
    config_file = Path(MODELS[model]["training_config_path"])
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    n_folds = cfg.get("data", {}).get("cv_folds", 5)
    MODEL_FOLDS[model] = list(range(1, n_folds + 1))


def get_all_evaluation_targets(wildcards) -> dict[str, list[str]]:
    """Generate expected intermediate evaluation targets for the current evaluation date."""
    csv_targets = []
    parquet_targets = []

    # Trainable models produce evaluation files per model fold under the dated directory
    for model in TRAINABLE_MODELS:
        for fold in MODEL_FOLDS[model]:
            csv_targets.append(
                f"{EVAL_DIR}/folds/{model}/fold_{fold}/grouped_evaluation_metrics.csv"
            )
            parquet_targets.append(
                f"{EVAL_DIR}/folds/{model}/fold_{fold}/per_text_and_category_confusion_metrics.parquet"
            )

    # Zero-shot models produce a single evaluation under the dated directory
    for model in ZEROSHOT_MODELS:
        csv_targets.append(
            f"{EVAL_DIR}/zeroshot/{model}/grouped_evaluation_metrics.csv"
        )
        parquet_targets.append(
            f"{EVAL_DIR}/zeroshot/{model}/per_text_and_category_confusion_metrics.parquet"
        )

    return {"csvs": csv_targets, "parquets": parquet_targets}


def format_seconds_to_hhmmss(total_seconds: float) -> str:
    """Convert a duration in seconds into hh:mm:ss format."""
    total_seconds_int = int(round(total_seconds))
    hours = total_seconds_int // 3600
    minutes = (total_seconds_int % 3600) // 60
    seconds = total_seconds_int % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


rule run_benchmark_gliner:
    input:
        f"{EVAL_DIR}/benchmark_models_grouped_metrics.csv",
        f"{EVAL_DIR}/all_folds_grouped_metrics.parquet",
        f"{EVAL_DIR}/all_folds_per_text_and_category_confusion_metrics.parquet",


# To distinguish from fold wildcards, we restrict the model wildcard to not include slashes.
wildcard_constraints:
    model="[^/]+",
    fold=r"\d+",


# Train all cross-validation folds in a single execution per model.
rule train_gliner:
    input:
        config=lambda wildcards: MODELS[wildcards.model]["training_config_path"],
    output:
        flag=touch("results/gliner/models/{model}/.done"),
    resources:
        gpu=1,
    run:
        training_config = load_config(Path(input.config))
        logger = create_logger(level="INFO")
        logger.info(
            f"Starting training for {wildcards.model} across all folds using {input.config}."
        )
        train_all_folds(training_config, logger=logger)


# Run inference for trainable models across cross-validation folds.
rule run_gliner_inference_folds:
    input:
        training_done="results/gliner/models/{model}/.done",
    output:
        flag=touch("results/gliner/inferences/folds/{model}/fold_{fold}/.done"),
    params:
        text="results/gliner/models/{model}/fold_{fold}/data/test.jsonl",
        metadata="results/gliner/models/{model}/fold_{fold}/data/test_metadata.txt",
        model_path=lambda wildcards: MODELS[wildcards.model][
            "model_path"
        ].format(fold=wildcards.fold),
        out_dir="results/gliner/inferences/folds/{model}/fold_{fold}",
    shell:
        """
        uv run extract-entities-with-gliner-all-texts \
            --text-path {params.text} \
            --metadata-path {params.metadata} \
            --model-path {params.model_path} \
            --output-dir {params.out_dir}
        """


# Run inference for zero-shot models directly on the ground truth directory.
rule run_gliner_inference_zeroshot:
    input:
        groundtruth_dir=str(GROUNDTRUTH_DIR),
    output:
        flag=touch("results/gliner/inferences/{model}/.done"),
    params:
        out_dir="results/gliner/inferences/{model}",
        model_path=lambda wildcards: MODELS[wildcards.model]["model_path"],
    shell:
        """
        uv run extract-entities-with-gliner-all-texts \
            --texts-path {input.groundtruth_dir} \
            --model-path {params.model_path} \
            --output-dir {params.out_dir}
        """


# Normalize entities extracted from fold-based inferences.
rule normalize_gliner_folds:
    input:
        flag="results/gliner/inferences/folds/{model}/fold_{fold}/.done",
        ffm_db=str(FFM_DB_PATH),
        soft_db=str(SOFTNAME_DB_PATH),
    output:
        flag=touch(
            "results/gliner/inferences_normalized/folds/{model}/fold_{fold}/.done"
        ),
    params:
        inf_dir="results/gliner/inferences/folds/{model}/fold_{fold}",
        out_dir="results/gliner/inferences_normalized/folds/{model}/fold_{fold}",
    shell:
        """
        uv run normalize-extracted-entities \
            --inferences-dir {params.inf_dir} \
            --ffm-db-path {input.ffm_db} \
            --softname-db-path {input.soft_db} \
            --output-dir {params.out_dir}
        """


# Normalize entities extracted from zero-shot inferences.
rule normalize_gliner_zeroshot:
    wildcard_constraints:
        model="|".join(ZEROSHOT_MODELS) if ZEROSHOT_MODELS else "$^",
    input:
        flag="results/gliner/inferences/{model}/.done",
        ffm_db=str(FFM_DB_PATH),
        soft_db=str(SOFTNAME_DB_PATH),
    output:
        flag=touch("results/gliner/inferences_normalized/{model}/.done"),
    params:
        inf_dir="results/gliner/inferences/{model}",
        out_dir="results/gliner/inferences_normalized/{model}",
    shell:
        """
        uv run normalize-extracted-entities \
            --inferences-dir {params.inf_dir} \
            --ffm-db-path {input.ffm_db} \
            --softname-db-path {input.soft_db} \
            --output-dir {params.out_dir}
        """


# Evaluate predictions on a single cross-validation fold using normalized entities.
rule evaluate_single_fold:
    input:
        "results/gliner/inferences_normalized/folds/{model}/fold_{fold}/.done",
    output:
        csv=temp(
            f"{EVAL_DIR}/folds/{{model}}/fold_{{fold}}/grouped_evaluation_metrics.csv"
        ),
        parquet=temp(
            f"{EVAL_DIR}/folds/{{model}}/fold_{{fold}}/per_text_and_category_confusion_metrics.parquet"
        ),
    params:
        inf_dir="results/gliner/inferences_normalized/folds/{model}/fold_{fold}",
        res_dir=f"{EVAL_DIR}/folds/{{model}}/fold_{{fold}}",
    shell:
        """
        uv run evaluate-entities-extraction \
            --inferences-dir {params.inf_dir} \
            --results-dir {params.res_dir}
        """


# Evaluate predictions for zero-shot models using normalized entities.
rule evaluate_zeroshot:
    wildcard_constraints:
        model="|".join(ZEROSHOT_MODELS) if ZEROSHOT_MODELS else "$^",
    input:
        "results/gliner/inferences_normalized/{model}/.done",
    output:
        csv=temp(
            f"{EVAL_DIR}/zeroshot/{{model}}/grouped_evaluation_metrics.csv"
        ),
        parquet=temp(
            f"{EVAL_DIR}/zeroshot/{{model}}/per_text_and_category_confusion_metrics.parquet"
        ),
    params:
        inf_dir="results/gliner/inferences_normalized/{model}",
        res_dir=f"{EVAL_DIR}/zeroshot/{{model}}",
    shell:
        """
        uv run evaluate-entities-extraction \
            --inferences-dir {params.inf_dir} \
            --results-dir {params.res_dir}
        """


# Merge evaluations from both folds and zero-shot runs into single parquet files.
rule merge_all_evaluations:
    input:
        unpack(get_all_evaluation_targets),
    output:
        all_grouped_parquet=f"{EVAL_DIR}/all_folds_grouped_metrics.parquet",
        all_detailed_parquet=f"{EVAL_DIR}/all_folds_per_text_and_category_confusion_metrics.parquet",
    run:
        grouped_frames = []
        detailed_frames = []

        for model in TRAINABLE_MODELS:
            for fold in MODEL_FOLDS[model]:
                csv_path = f"{EVAL_DIR}/folds/{model}/fold_{fold}/grouped_evaluation_metrics.csv"
                df_grp = pd.read_csv(csv_path)
                df_grp["fold"] = fold
                df_grp["model"] = model
                grouped_frames.append(df_grp)

                parquet_path = f"{EVAL_DIR}/folds/{model}/fold_{fold}/per_text_and_category_confusion_metrics.parquet"
                df_det = pd.read_parquet(parquet_path)
                df_det["fold"] = fold
                df_det["model"] = model
                detailed_frames.append(df_det)

        for model in ZEROSHOT_MODELS:
            csv_path = (
                f"{EVAL_DIR}/zeroshot/{model}/grouped_evaluation_metrics.csv"
            )
            df_grp = pd.read_csv(csv_path)
            df_grp["fold"] = "all"
            df_grp["model"] = model
            grouped_frames.append(df_grp)

            parquet_path = f"{EVAL_DIR}/zeroshot/{model}/per_text_and_category_confusion_metrics.parquet"
            df_det = pd.read_parquet(parquet_path)
            df_det["fold"] = "all"
            df_det["model"] = model
            detailed_frames.append(df_det)

        pd.concat(grouped_frames, ignore_index=True).to_parquet(
            output.all_grouped_parquet, index=False
        )
        pd.concat(detailed_frames, ignore_index=True).to_parquet(
            output.all_detailed_parquet, index=False
        )

# Generate benchmark_models grouped metrics CSV across all models and categories.
rule generate_benchmark_models_csv:
    input:
        all_grouped_parquet=f"{EVAL_DIR}/all_folds_grouped_metrics.parquet",
    output:
        benchmark_csv=f"{EVAL_DIR}/benchmark_models_grouped_metrics.csv",
    run:
        # Read the combined grouped metrics parquet file.
        df_all = pd.read_parquet(input.all_grouped_parquet)
        rows = []
        # Iterate over each model.
        for model_key, meta in MODELS.items():
            df_model = df_all[df_all["model"] == model_key]
            if df_model.empty:
                continue
            # Compute metrics for each category within each model,
            # aggregating the results across all folds for trainable models.
            for category, df_cat in df_model.groupby("category"):
                total_cost = float(df_cat["total_cost_usd"].sum()) if "total_cost_usd" in df_cat else 0.0
                total_time = float(df_cat["total_inference_time_sec"].sum()) if "total_inference_time_sec" in df_cat else 0.0
                total_preds = int(df_cat["nb_predicted_entities_raw"].sum()) if "nb_predicted_entities_raw" in df_cat else 0

                cost_per_ent = (total_cost / total_preds) if total_preds > 0 else None
                time_per_ent = (total_time / total_preds) if total_preds > 0 else None

                rows.append({
                    "Inference_date": str(df_cat["inference_date"].max()) if "inference_date" in df_cat else "",
                    "Model": meta["display_name"],
                    "Category": category,
                    "Number_of_texts_with_category": int(df_cat["nb_texts_with_category"].sum()) if "nb_texts_with_category" in df_cat else 0,
                    "Correct_format_(%)": float(df_cat["correct_format_pct"].mean()) if "correct_format_pct" in df_cat else 100.0,
                    "Hallucinations_(%)": float(df_cat["hallucinations_pct"].mean()) if "hallucinations_pct" in df_cat else 0.0,
                    "Precision": float(df_cat["precision"].mean()),
                    "Precision_with_no_hallucination": float(df_cat["precision_no_hallucination"].mean()) if "precision_no_hallucination" in df_cat else float(df_cat["precision"].mean()),
                    "Recall": float(df_cat["recall"].mean()),
                    "F1": float(df_cat["f1"].mean()),
                    "F1_with_no_hallucination": float(df_cat["f1_no_hallucination"].mean()) if "f1_no_hallucination" in df_cat else float(df_cat["f1"].mean()),
                    "Fbeta_0.5": float(df_cat["fbeta_0_5"].mean()) if "fbeta_0_5" in df_cat else np.nan,
                    "Fbeta_0.5_with_no_hallucination": float(df_cat["fbeta_0_5_no_hallucination"].mean()) if "fbeta_0_5_no_hallucination" in df_cat else np.nan,
                    "Cost_by_entity_($)": cost_per_ent,
                    "Inference_time_by_entity_(s)": time_per_ent,
                    "Number_of_predicted_entities": total_preds,
                    "Cost_total_($)": total_cost,
                    "Inference_time_total_(s)": total_time,
                    "Inference_time_total_(hh:mm:ss)": format_seconds_to_hhmmss(total_time),
                })

        pd.DataFrame(rows).to_csv(output.benchmark_csv, index=False)