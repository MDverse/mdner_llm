# Snakefile for training and evaluating GLiNER models.

import shutil
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from mdner_llm.gliner.train_gliner import load_config, train_all_folds
from mdner_llm.logger import create_logger

# Global paths.
CONFIG_TRAINING_PATH = Path("workflow/configs/gliner_training.yaml")
CONFIG_MODELS_PATH = Path("workflow/configs/gliner_models.yaml")

# Load models specification from YAML.
with open(CONFIG_MODELS_PATH, "r", encoding="utf-8") as f:
    MODELS = yaml.safe_load(f)
ALL_MODELS = list(MODELS.keys())

# Load training config to retrieve cv_folds automatically.
cfg = load_config(CONFIG_TRAINING_PATH)
N_FOLDS = cfg.data.cv_folds
FOLDS = list(range(1, N_FOLDS + 1))


def get_model_dependencies(wildcards) -> list[str]:
    """Determine the dependencies for a given model based on its configuration."""
    cfg = MODELS[wildcards.model]
    if cfg.get("is_trainable"):
        return [f"results/gliner/models/{wildcards.model}/fold_{wildcards.fold}/best"]
    return []


def get_adapter_arg(wildcards) -> str:
    """Build adapter argument if available."""
    adapter = MODELS[wildcards.model].get("adapter_path")
    return f"--adapter-path {adapter.format(fold=wildcards.fold)}" if adapter else ""


def format_seconds_to_hhmmss(total_seconds: float) -> str:
    """Convert a duration in seconds into hh:mm:ss format."""
    total_seconds = int(round(total_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


rule gliner_all:
    input:
        "results/gliner/evaluation/models_benchmark_mean_std_summary.csv",
        "results/gliner/evaluation/benchmark_models_grouped_metrics.csv",
        "results/gliner/evaluation/all_folds_grouped_metrics.parquet",
        "results/gliner/evaluation/all_folds_per_text_and_category_confusion_metrics.parquet",


# Train cross-validation folds sequentially on GPU.
rule train_gliner:
    input:
        config=str(CONFIG_TRAINING_PATH),
        models_config=str(CONFIG_MODELS_PATH),
    output:
        directory(expand("results/gliner/models/{{model}}/fold_{fold}/best", fold=FOLDS)),
        expand("results/gliner/models/{{model}}/fold_{fold}/data/test.jsonl", fold=FOLDS),
        expand("results/gliner/models/{{model}}/fold_{fold}/data/test_metadata.txt", fold=FOLDS),
    resources:
        gpu=1
    run:
        meta = MODELS[wildcards.model]
        cfg.model.name = meta["base_model"]
        cfg.model.experiment_name = wildcards.model
        cfg.training.use_lora = meta.get("use_lora", False)
        cfg.training.save_adapter_only = meta.get("use_lora", False)
        cfg.training.encoder_lr = float(meta["encoder_lr"])
        cfg.training.task_lr = float(meta["task_lr"])
        cfg.training.warmup_ratio = float(meta["warmup_ratio"])
        logger = create_logger(level="INFO")
        logger.info(f"Starting training for {wildcards.model}.")
        train_all_folds(cfg, logger=logger)


# Run entity extraction for each model and fold.
rule run_gliner_inference:
    input:
        text="results/gliner/models/gliner2-base-v1-finetuned/fold_{fold}/data/test.jsonl",
        metadata="results/gliner/models/gliner2-base-v1-finetuned/fold_{fold}/data/test_metadata.txt",
        checkpoints=get_model_dependencies,
    output:
        flag=touch("results/gliner/inferences/{model}/fold_{fold}/.done"),
    params:
        out_dir="results/gliner/inferences/{model}/fold_{fold}",
        model_path=lambda wildcards: MODELS[wildcards.model]["model_path"].format(fold=wildcards.fold),
        adapter_arg=get_adapter_arg,
    shell:
        """
        uv run extract-entities-with-gliner-all-texts \
            --text-path {input.text} \
            --metadata-path {input.metadata} \
            --model-path {params.model_path} \
            {params.adapter_arg} \
            --output-dir {params.out_dir}
        """


# Evaluate predictions on a single fold.
rule evaluate_single_fold:
    input:
        "results/gliner/inferences/{model}/fold_{fold}/.done",
    output:
        csv=temp("results/gliner/evaluation/folds/{model}/fold_{fold}/grouped_evaluation_metrics.csv"),
        parquet=temp("results/gliner/evaluation/folds/{model}/fold_{fold}/per_text_and_category_confusion_metrics.parquet"),
    params:
        inf_dir="results/gliner/inferences/{model}/fold_{fold}",
        res_dir="results/gliner/evaluation/folds/{model}/fold_{fold}",
    shell:
        """
        uv run evaluate-entities-extraction \
            --inferences-dir {params.inf_dir} \
            --results-dir {params.res_dir}
        """


# Merge fold evaluations and clean up intermediate files.
rule merge_all_folds_evaluations:
    input:
        csvs=expand(
            "results/gliner/evaluation/folds/{model}/fold_{fold}/grouped_evaluation_metrics.csv",
            model=ALL_MODELS,
            fold=FOLDS,
        ),
        parquets=expand(
            "results/gliner/evaluation/folds/{model}/fold_{fold}/per_text_and_category_confusion_metrics.parquet",
            model=ALL_MODELS,
            fold=FOLDS,
        ),
    output:
        all_grouped_parquet="results/gliner/evaluation/all_folds_grouped_metrics.parquet",
        all_detailed_parquet="results/gliner/evaluation/all_folds_per_text_and_category_confusion_metrics.parquet",
    run:
        grouped_dfs = []
        detailed_dfs = []

        for model in ALL_MODELS:
            for fold in FOLDS:
                csv_path = f"results/gliner/evaluation/folds/{model}/fold_{fold}/grouped_evaluation_metrics.csv"
                df_grp = pd.read_csv(csv_path)
                df_grp["fold"] = fold
                df_grp["model"] = model
                grouped_dfs.append(df_grp)

                parquet_path = f"results/gliner/evaluation/folds/{model}/fold_{fold}/per_text_and_category_confusion_metrics.parquet"
                df_det = pd.read_parquet(parquet_path)
                df_det["fold"] = fold
                df_det["model"] = model
                detailed_dfs.append(df_det)

        df_all_grouped = pd.concat(grouped_dfs, ignore_index=True)
        df_all_grouped.to_parquet(output.all_grouped_parquet, index=False)

        df_all_detailed = pd.concat(detailed_dfs, ignore_index=True)
        df_all_detailed.to_parquet(output.all_detailed_parquet, index=False)

        folds_dir = Path("results/gliner/evaluation/folds")
        if folds_dir.exists():
            shutil.rmtree(folds_dir)


# Aggregate benchmark metrics across folds (Mean / Std summary).
rule generate_summary_csv:
    input:
        all_grouped_parquet="results/gliner/evaluation/all_folds_grouped_metrics.parquet",
    output:
        summary_csv=report(
            "results/gliner/evaluation/models_benchmark_mean_std_summary.csv",
            category="Models Benchmark Summary Across Folds",
        ),
    run:
        df_all = pd.read_parquet(input.all_grouped_parquet)
        models_metrics = {}

        for model_key, meta in MODELS.items():
            df_model = df_all[df_all["model"] == model_key]
            metrics_list = []
            for fold in FOLDS:
                df_fold = df_model[df_model["fold"] == fold]
                if df_fold.empty:
                    continue

                micro = df_fold[df_fold["category"] == "OVERALL_MICRO"].iloc[0]
                macro = df_fold[df_fold["category"] == "OVERALL_MACRO"].iloc[0]

                n_preds = micro["nb_predicted_entities_raw"]
                cost = (micro["total_cost_usd"] / n_preds) if n_preds > 0 else 0.0
                latency = (micro["total_inference_time_sec"] / n_preds) if n_preds > 0 else 0.0

                metrics_list.append({
                    "inference_date": str(micro.get("inference_date", "")),
                    "precision": float(micro["precision"]),
                    "recall": float(micro["recall"]),
                    "micro_f1": float(micro["f1"]),
                    "macro_f1": float(macro["f1"]),
                    "cost_per_ent": float(cost),
                    "time_per_ent": float(latency),
                })

            df_f = pd.DataFrame(metrics_list)
            if not df_f.empty:
                models_metrics[model_key] = {
                    "Inference_date": df_f["inference_date"].max(),
                    "Model_name": meta["display_name"],
                    "Precision_mean": float(df_f["precision"].mean()),
                    "Precision_std": float(df_f["precision"].std(ddof=1)),
                    "Recall_mean": float(df_f["recall"].mean()),
                    "Recall_std": float(df_f["recall"].std(ddof=1)),
                    "Micro_F1_mean": float(df_f["micro_f1"].mean()),
                    "Micro_F1_std": float(df_f["micro_f1"].std(ddof=1)),
                    "Macro_F1_mean": float(df_f["macro_f1"].mean()),
                    "Macro_F1_std": float(df_f["macro_f1"].std(ddof=1)),
                    "Cost_by_entity_mean ($)": float(df_f["cost_per_ent"].mean()),
                    "Cost_by_entity_std ($)": float(df_f["cost_per_ent"].std(ddof=1)),
                    "Inference_time_by_entity_mean (s)": float(df_f["time_per_ent"].mean()),
                    "Inference_time_by_entity_std (s)": float(df_f["time_per_ent"].std(ddof=1)),
                }

        name_to_metrics = {m["Model_name"]: m for m in models_metrics.values()} | models_metrics
        for model_key, row in models_metrics.items():
            meta = MODELS[model_key]
            if meta.get("is_trainable"):
                base = name_to_metrics.get(meta.get("base_model"))
                if base:
                    row["Delta_Micro_F1"] = float(row["Micro_F1_mean"] - base["Micro_F1_mean"])
                    row["Delta_Macro_F1"] = float(row["Macro_F1_mean"] - base["Macro_F1_mean"])

        pd.DataFrame(list(models_metrics.values())).to_csv(output.summary_csv, index=False)


# Generate benchmark_models grouped metrics CSV across all models and categories.
rule generate_benchmark_models_csv:
    input:
        all_grouped_parquet="results/gliner/evaluation/all_folds_grouped_metrics.parquet",
    output:
        benchmark_csv=report(
            "results/gliner/evaluation/benchmark_models_grouped_metrics.csv",
            category="Detailed Benchmark Metrics Across Categories",
        ),
    run:
        df_all = pd.read_parquet(input.all_grouped_parquet)
        rows = []

        for model_key, meta in MODELS.items():
            df_model = df_all[df_all["model"] == model_key]
            if df_model.empty:
                continue

            for category, df_cat in df_model.groupby("category"):
                total_cost = float(df_cat["total_cost_usd"].sum()) if "total_cost_usd" in df_cat else 0.0
                total_time = float(df_cat["total_inference_time_sec"].sum()) if "total_inference_time_sec" in df_cat else 0.0
                total_preds = int(df_cat["nb_predicted_entities_raw"].sum()) if "nb_predicted_entities_raw" in df_cat else 0

                cost_per_ent = (total_cost / total_preds) if total_preds > 0 else 0.0
                time_per_ent = (total_time / total_preds) if total_preds > 0 else 0.0

                rows.append({
                    "Inference_date": str(df_cat["inference_date"].max()) if "inference_date" in df_cat else "",
                    "Name": meta["display_name"],
                    "category": category,
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
