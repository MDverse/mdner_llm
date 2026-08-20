# Snakefile for training and evaluating GLiNER models.

import shutil
from pathlib import Path
import pandas as pd

# Global parameters.
CONFIG_PATH = Path("src/mdner_llm/gliner/training_config.yaml")
N_FOLDS = 1
FOLDS = list(range(1, N_FOLDS + 1))

# Models specification.
MODELS = {
    # 1. Zero-shot GLiNER fine-tuned Biomedical Models
    "gliner-biomed-base-v1.0": {
        "display_name": "ihor/gliner-biomed-base-v1.0",
        "model_path": "ihor/gliner-biomed-base-v1.0",
        "adapter_path": None,
        "is_trainable": False,
    },
    "gliner-biomed-large-v1.0": {
        "display_name": "ihor/gliner-biomed-large-v1.0",
        "model_path": "ihor/gliner-biomed-large-v1.0",
        "adapter_path": None,
        "is_trainable": False,
    },
    # 2. Zero-shot GLiNER2 Models
    "gliner2-base-v1": {
        "display_name": "fastino/gliner2-base-v1",
        "model_path": "fastino/gliner2-base-v1",
        "adapter_path": None,
        "is_trainable": False,
    },
    "gliner2-large-v1": {
        "display_name": "fastino/gliner2-large-v1",
        "model_path": "fastino/gliner2-large-v1",
        "adapter_path": None,
        "is_trainable": False,
    },
    # 3. GLiNER2 Full Fine-tuned
    "gliner2-base-v1-finetuned": {
        "display_name": "fastino/gliner2-base-v1-finetuned",
        "base_model": "fastino/gliner2-base-v1",
        "model_path": "results/gliner/models/gliner2-base-v1-finetuned/fold_{fold}/best",
        "adapter_path": None,
        "use_lora": False,
        "is_trainable": True,
        "encoder_lr": 2e-6,
        "task_lr": 2e-5,
        "warmup_ratio": 0.08,
    },
    #"gliner2-large-v1-finetuned": {
    #    "display_name": "fastino/gliner2-large-v1-finetuned",
    #    "base_model": "fastino/gliner2-large-v1",
    #    "model_path": "results/gliner/models/gliner2-large-v1-finetuned/fold_{fold}/best",
    #    "adapter_path": None,
    #    "use_lora": False,
    #    "is_trainable": True,
    #    "encoder_lr": 2e-6,
    #    "task_lr": 2e-5,
    #   "warmup_ratio": 0.08,
    #},
    # 4. GLiNER2 LoRA Fine-tuned
    "gliner2-base-v1-finetuned-lora": {
        "display_name": "fastino/gliner2-base-v1-finetuned-lora",
        "base_model": "fastino/gliner2-base-v1",
        "model_path": "fastino/gliner2-base-v1",
        "adapter_path": "results/gliner/models/gliner2-base-v1-finetuned-lora/fold_{fold}/best",
        "use_lora": True,
        "is_trainable": True,
        "encoder_lr": 1e-5,
        "task_lr": 5e-5,
        "warmup_ratio": 0.15,
    },
    #"gliner2-large-v1-finetuned-lora": {
    #    "display_name": "fastino/gliner2-large-v1-finetuned-lora",
    #    "base_model": "fastino/gliner2-large-v1",
    #    "model_path": "fastino/gliner2-large-v1",
    #    "adapter_path": "results/gliner/models/gliner2-large-v1-finetuned-lora/fold_{fold}/best",
    #    "use_lora": True,
    #    "is_trainable": True,
    #    "encoder_lr": 1e-5,
    #    "task_lr": 5e-5,
    #    "warmup_ratio": 0.15,
    #},
}
ALL_MODELS = list(MODELS.keys())


def get_model_dependencies(wildcards) -> list[str]:
    """Determine the dependencies for a given model based on its configuration.

    Returns
    -------
    list[str]
        List of paths to the best model checkpoints for the specified model and fold.
    """
    cfg = MODELS[wildcards.model]
    if cfg.get("is_trainable"):
        return [f"results/gliner/models/{wildcards.model}/fold_{wildcards.fold}/best"]
    return []


def get_adapter_arg(wildcards) -> str:
    """Build adapter argument if available.

    Returns
    -------
    str
        Adapter argument string for the inference command, 
        or an empty string if no adapter is specified.
    """
    adapter = MODELS[wildcards.model]["adapter_path"]
    return f"--adapter-path {adapter.format(fold=wildcards.fold)}" if adapter else ""


rule gliner_all:
    input:
        "results/gliner/evaluation/models_benchmark_mean_std_summary.csv",
        "results/gliner/evaluation/all_folds_grouped_metrics.parquet",
        "results/gliner/evaluation/all_folds_per_text_and_category_confusion_metrics.parquet",


# Train cross-validation folds sequentially on GPU.
rule train_gliner:
    input:
        config=str(CONFIG_PATH),
    output:
        directory(expand("results/gliner/models/{{model}}/fold_{fold}/best", fold=FOLDS)),
        expand("results/gliner/models/{{model}}/fold_{fold}/data/test.jsonl", fold=FOLDS),
        expand("results/gliner/models/{{model}}/fold_{fold}/data/test_metadata.txt", fold=FOLDS),
    resources:
        gpu=1
    run:
        from mdner_llm.gliner.train_gliner import load_config, train_all_folds
        from mdner_llm.logger import create_logger
        # Load configuration and set up logger
        logger = create_logger(level="INFO")
        cfg = load_config(input.config, logger=logger)
        if cfg is None:
            raise ValueError(f"Could not load configuration from {input.config}.")
        # Depending on the model,
        # configure training parameters.
        meta = MODELS[wildcards.model]
        cfg.model.name = meta["base_model"]
        cfg.model.experiment_name = wildcards.model
        cfg.training.cv_folds = N_FOLDS
        cfg.training.use_lora = meta["use_lora"]
        cfg.training.save_adapter_only = meta["use_lora"]
        cfg.training.encoder_lr = meta["encoder_lr"]
        cfg.training.task_lr = meta["task_lr"]
        cfg.training.warmup_ratio = meta["warmup_ratio"]
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


# Aggregate benchmark metrics across folds.
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
        # Compute fold-level statistics for each model.
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
        # Compute delta F1 against corresponding base models.
        summary_rows = []
        for model_key, row_dict in models_metrics.items():
            meta = MODELS[model_key]
            delta_f1 = None
            if meta.get("is_trainable"):
                base_identifier = meta.get("base_model")
                # Look up base model metrics by key or display name.
                base_entry = None
                for candidate_key, candidate_metrics in models_metrics.items():
                    if candidate_key == base_identifier or candidate_metrics["Model_name"] == base_identifier:
                        base_entry = candidate_metrics
                        break
                if base_entry is not None:
                    delta_f1 = float(row_dict["Micro_F1_mean"] - base_entry["Micro_F1_mean"])
            row_dict["Delta_F1"] = delta_f1
            summary_rows.append(row_dict)
        pd.DataFrame(summary_rows).to_csv(output.summary_csv, index=False)