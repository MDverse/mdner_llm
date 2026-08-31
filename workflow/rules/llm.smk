# Snakefile for LLM benchmark strategies, models, and consensus evaluation.

import json
import shutil
import time
from pathlib import Path
import pandas as pd
from filelock import FileLock

from mdner_llm.common import sanitize_filename


# Load pipeline configuration file.
configfile: "workflow/configs/llm_models.yaml"

# Global paths and inputs configuration.
texts_directory = Path(config["texts_path"])
json_files_list = sorted(file_path.name for file_path in texts_directory.glob("*.json"))
target_files = json_files_list[:config.get("max_samples")] if config.get("max_samples") else json_files_list
consensus_threshold_value = config.get("consensus_threshold", 0.5)
api_sleep_delay = config.get("api_sleep_delay", 5)

# Model dictionaries mapping safe filesystem names to full identifiers.
benchmark_models = {sanitize_filename(m): m for m in config.get("benchmark_models", [])}
full_eval_models = {sanitize_filename(m): m for m in config.get("full_eval_models", [])}
consensus_models = {sanitize_filename(m): m for m in config.get("consensus_models", [])}
# Unified registry of all known model full names
all_models_dict = {**benchmark_models, **full_eval_models, **consensus_models}


evaluation_targets = []
# Scenario 1: Benchmark prompting strategies.
for strategy_name in config["benchmark_strategies"]:
    for safe_model_name in benchmark_models:
        evaluation_targets.append(
            f"results/llm/evaluation/benchmark_strategies/{strategy_name}/{safe_model_name}/.done"
        )
# Scenario 2: Full evaluation on benchmark and reference models.
for safe_model_name in full_eval_models:
    evaluation_targets.append(
        f"results/llm/evaluation/benchmark_models/with_instructor_with_guidelines/{safe_model_name}/.done"
    )
# Scenario 3: Consensus aggregation runs.
if consensus_models:
    consensus_temperatures = [str(t) for t in config.get("consensus_temperatures", [1.0])]
    consensus_setups = [
        f"temp_{'_and_'.join(consensus_temperatures[:end_idx])}"
        for end_idx in range(1, len(consensus_temperatures) + 1)
    ]
    for setup_identifier in consensus_setups:
        evaluation_targets.append(
            f"results/llm/evaluation/consensus/{setup_identifier}/.done"
        )

# ==============================================================================
# HELPER: INCREMENTAL GLOBAL AGGREGATION
# ==============================================================================

def update_global_aggregates(new_csv_path: str, new_parquet_path: str, scenario_name: str, combo_name: str, model_safe_name: str):
    """Update global benchmark CSV/Parquet files incrementally with file locking."""
    eval_dir = Path("results/llm/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    lock_path = eval_dir / ".aggregation.lock"

    all_grouped_csv = eval_dir / "all_grouped_evaluation_metrics.csv"
    all_detailed_parquet = eval_dir / "all_per_text_and_category_confusion_metrics.parquet"
    benchmark_strategies_csv = eval_dir / "benchmark_strategies.csv"
    benchmark_models_csv = eval_dir / "benchmark_models.csv"

    with FileLock(str(lock_path), timeout=60):
        # Update Grouped Metrics CSV
        new_df = pd.read_csv(new_csv_path)
        new_df["scenario"] = scenario_name
        new_df["combo"] = combo_name
        new_df["model_safe"] = model_safe_name
        if all_grouped_csv.exists():
            existing_df = pd.read_csv(all_grouped_csv)
            mask = (
                (existing_df["scenario"] == scenario_name) &
                (existing_df["combo"] == combo_name) &
                (existing_df["model_safe"] == model_safe_name)
            )
            full_grouped_df = pd.concat([existing_df[~mask], new_df], ignore_index=True)
        else:
            full_grouped_df = new_df
        full_grouped_df.to_csv(all_grouped_csv, index=False)
        # Update Detailed Metrics Parquet
        if Path(new_parquet_path).exists():
            new_det_df = pd.read_parquet(new_parquet_path)
            new_det_df["scenario"] = scenario_name
            new_det_df["combo"] = combo_name
            new_det_df["model_safe"] = model_safe_name
            if all_detailed_parquet.exists():
                existing_det_df = pd.read_parquet(all_detailed_parquet)
                mask_det = (
                    (existing_det_df["scenario"] == scenario_name) &
                    (existing_det_df["combo"] == combo_name) &
                    (existing_det_df["model_safe"] == model_safe_name)
                )
                full_detailed_df = pd.concat([existing_det_df[~mask_det], new_det_df], ignore_index=True)
            else:
                full_detailed_df = new_det_df
            full_detailed_df.to_parquet(all_detailed_parquet, index=False)

        # Update Formatted Benchmark Summaries
        column_mapping = {
            "inference_date": "Inference_date",
            "model_name": "Name",
            "framework_name": "Framework",
            "category": "category",
            "nb_texts_with_category": "Number_of_texts_with_category",
            "pct_correct_format": "Correct_format_(%)",
            "pct_hallucinations": "Hallucinations_(%)",
            "precision": "Precision",
            "precision_no_hallucination": "Precision_with_no_hallucination",
            "recall": "Recall",
            "f1": "F1",
            "f1_no_hallucination": "F1_with_no_hallucination",
            "fbeta_0.5": "Fbeta_0.5",
            "fbeta_0.5_no_hallucination": "Fbeta_0.5_with_no_hallucination",
            "nb_predicted_entities": "Number_of_predicted_entities",
            "total_cost_usd": "Cost_total_($)",
            "total_inference_time_sec": "Inference_time_total_(s)",
        }
        working_df = full_grouped_df.rename(columns=column_mapping).copy()
        total_preds = (
            working_df["nb_predicted_entities_raw"].replace(0, pd.NA)
            if "nb_predicted_entities_raw" in working_df
            else working_df["Number_of_predicted_entities"].replace(0, pd.NA)
        )
        working_df["Cost_by_entity_($)"] = working_df["Cost_total_($)"] / total_preds
        working_df["Inference_time_by_entity_(s)"] = working_df["Inference_time_total_(s)"] / total_preds
        working_df["Inference_time_total_(hh:mm:ss)"] = working_df["Inference_time_total_(s)"].apply(
            lambda s: str(pd.to_timedelta(int(s), unit="s")).split()[-1] if pd.notna(s) else pd.NA
        )
        columns_order = [
            "Inference_date", "Name", "Framework", "With_Guideline", "category",
            "Number_of_texts_with_category", "Correct_format_(%)", "Hallucinations_(%)",
            "Precision", "Precision_with_no_hallucination", "Recall", "F1",
            "F1_with_no_hallucination", "Fbeta_0.5", "Fbeta_0.5_with_no_hallucination",
            "Cost_by_entity_($)", "Inference_time_by_entity_(s)", "Number_of_predicted_entities",
            "Cost_total_($)", "Inference_time_total_(s)", "Inference_time_total_(hh:mm:ss)",
        ]
        # Strategies
        strat_df = working_df[working_df["scenario"] == "benchmark_strategies"].copy()
        if not strat_df.empty:
            # Add a flag column to indicate whether the strategy includes guidelines
            strat_df["With_Guideline"] = strat_df["combo"].apply(lambda c: "with_guidelines" in str(c))
            avail_cols = [col for col in columns_order if col in strat_df.columns]
            strat_df[avail_cols].to_csv(benchmark_strategies_csv, index=False)

        # Models
        models_df = working_df[working_df["scenario"].isin(["benchmark_models", "consensus"])].copy()
        if not models_df.empty:
            model_cols = [col for col in columns_order if col not in ("Framework", "With_Guideline")]
            avail_cols = [col for col in model_cols if col in models_df.columns]
            models_df[avail_cols].to_csv(benchmark_models_csv, index=False)


rule llm_all:
    input:
        evaluation_targets


# ==============================================================================
# SCENARIOS 1 & 2: EXTRACTION, NORMALIZATION, AND EVALUATION
# ==============================================================================

rule extract_benchmark_and_full:
    input:
        texts_files=expand("{path}/{json_file}", path=texts_directory, json_file=target_files),
        prompt=config["prompt_path"],
        examples=config["examples_path"],
    output:
        out_dir=directory("results/llm/inferences/raw/{combo}/{model_safe}"),
    resources:
        api_calls=1
    params:
        model=lambda wildcards: all_models_dict[wildcards.model_safe],
        framework=lambda wildcards: config["benchmark_strategies"].get(wildcards.combo, {}).get("framework", "instructor"),
        guidelines=lambda wildcards: config["benchmark_strategies"].get(wildcards.combo, {}).get("guidelines", config["guidelines_path"]),
        sleep_time=api_sleep_delay,
    shell:
        """
        mkdir -p {output.out_dir}
        for file in {input.texts_files}; do
            uv run extract-entities-with-llm \
                --text-path "$file" \
                --model "{params.model}" \
                --prompt-path {input.prompt} \
                --guidelines-path {params.guidelines} \
                --examples-path {input.examples} \
                --framework {params.framework} \
                --temperature 1.0 \
                --output-dir {output.out_dir}
        done
        sleep {params.sleep_time}
        """

rule normalize_benchmark_and_full:
    input:
        inferences_dir="results/llm/inferences/raw/{combo}/{model_safe}",
        ffm_db=config["ffm_db_path"],
        softname_db=config["softname_db_path"],
    output:
        norm_dir=directory("results/llm/inferences_normalized/{scenario}/{combo}/{model_safe}"),
    resources:
        api_calls=1
    params:
        norm_model=config["normalization_model"],
        sleep_time=api_sleep_delay,
    shell:
        """
        uv run normalize-extracted-entities \
            --inferences-dir {input.inferences_dir} \
            --ffm-db-path {input.ffm_db} \
            --softname-db-path {input.softname_db} \
            --model-name "{params.norm_model}" \
            --output-dir {output.norm_dir}
        sleep {params.sleep_time}
        """

rule evaluate_benchmark_and_full:
    input:
        inferences_dir="results/llm/inferences_normalized/{scenario}/{combo}/{model_safe}",
    output:
        eval_csv="results/llm/evaluation/{scenario}/{combo}/{model_safe}/grouped_evaluation_metrics.csv",
        eval_parquet="results/llm/evaluation/{scenario}/{combo}/{model_safe}/per_text_and_category_confusion_metrics.parquet",
        done=touch("results/llm/evaluation/{scenario}/{combo}/{model_safe}/.done"),
    run:
        shell(
            """
            mkdir -p $(dirname {output.eval_csv})
            uv run evaluate-entities-extraction \
                --inferences-dir {input.inferences_dir} \
                --results-dir $(dirname {output.eval_csv})
            """
        )
        update_global_aggregates(
            new_csv_path=output.eval_csv,
            new_parquet_path=output.eval_parquet,
            scenario_name=wildcards.scenario,
            combo_name=wildcards.combo,
            model_safe_name=wildcards.model_safe,
        )


# ==============================================================================
# SCENARIO 3: CONSENSUS RUNS AND MULTI-TEMPERATURE AGGREGATIONS
# ==============================================================================

rule extract_consensus_runs:
    input:
        texts_files=expand("{path}/{json_file}", path=texts_directory, json_file=target_files),
        prompt=config["prompt_path"],
        guidelines=config["guidelines_path"],
        examples=config["examples_path"],
    output:
        out_dir=directory("results/llm/inferences/consensus_raw/temp_{temp}/{model_safe}"),
    resources:
        api_calls=1
    wildcard_constraints:
        temp=r"(?!1(\.0)?$).*"  # Ensure that the temperature is not 1 or 1.0 for consensus runs
    params:
        model=lambda wildcards: consensus_models[wildcards.model_safe],
        temp=lambda wildcards: wildcards.temp,
        sleep_time=api_sleep_delay,
    shell:
        """
        mkdir -p {output.out_dir}
        for file in {input.texts_files}; do
            uv run extract-entities-with-llm \
                --text-path "$file" \
                --model "{params.model}" \
                --prompt-path {input.prompt} \
                --guidelines-path {input.guidelines} \
                --examples-path {input.examples} \
                --temperature {params.temp} \
                --framework instructor \
                --output-dir {output.out_dir}
        done
        sleep {params.sleep_time}
        """

def get_consensus_inferences_input(wildcards) -> list[str]:
    temperatures_string = wildcards.setup.replace("temp_", "")
    included_temperatures = temperatures_string.split("_and_")
    
    input_directories = []
    for temperature_value in included_temperatures:
        for model_identifier in consensus_models.keys():
            # If the temperature is 1 or 1.0, use the raw inferences with guidelines
            if str(temperature_value) in ["1", "1.0"]:
                input_directories.append(
                    f"results/llm/inferences/raw/with_instructor_with_guidelines/{model_identifier}"
                )
            else:
                input_directories.append(
                    f"results/llm/inferences/consensus_raw/temp_{temperature_value}/{model_identifier}"
                )
    return input_directories

rule aggregate_consensus:
    input:
        inferences=get_consensus_inferences_input,
    output:
        consensus_dir=directory("results/llm/inferences/consensus_aggregated/{setup}"),
    params:
        threshold=consensus_threshold_value,
        staging_dir="results/llm/inferences_consensus_staging/{setup}",
    shell:
        """
        mkdir -p {params.staging_dir}
        mkdir -p {output.consensus_dir}
        shopt -s nullglob

        for source_directory in {input.inferences}; do
            if [ -d "$source_directory" ]; then
                cp "$source_directory"/*.json {params.staging_dir}/ 2>/dev/null || true
            fi
        done

        uv run aggregate-consensus-entities \
            --inferences-dir {params.staging_dir} \
            --threshold {params.threshold} \
            --output-dir {output.consensus_dir}

        rm -rf {params.staging_dir}
        rmdir results/llm/inferences_consensus_staging 2>/dev/null || true
        """

rule normalize_consensus:
    input:
        inferences_dir="results/llm/inferences/consensus_aggregated/{setup}",
        ffm_db=config["ffm_db_path"],
        softname_db=config["softname_db_path"],
    output:
        norm_dir=directory("results/llm/inferences_normalized/consensus/{setup}"),
    resources:
        api_calls=1
    params:
        norm_model=config["normalization_model"],
        sleep_time=api_sleep_delay,
    shell:
        """
        uv run normalize-extracted-entities \
            --inferences-dir {input.inferences_dir} \
            --ffm-db-path {input.ffm_db} \
            --softname-db-path {input.softname_db} \
            --model-name "{params.norm_model}" \
            --output-dir {output.norm_dir}
        sleep {params.sleep_time}
        """

rule evaluate_consensus:
    input:
        inferences_dir="results/llm/inferences_normalized/consensus/{setup}",
    output:
        eval_csv="results/llm/evaluation/consensus/{setup}/grouped_evaluation_metrics.csv",
        eval_parquet="results/llm/evaluation/consensus/{setup}/per_text_and_category_confusion_metrics.parquet",
        done=touch("results/llm/evaluation/consensus/{setup}/.done"),
    run:
        shell(
            """
            mkdir -p $(dirname {output.eval_csv})
            uv run evaluate-entities-extraction \
                --inferences-dir {input.inferences_dir} \
                --results-dir $(dirname {output.eval_csv})
            """
        )
        update_global_aggregates(
            new_csv_path=output.eval_csv,
            new_parquet_path=output.eval_parquet,
            scenario_name="consensus",
            combo_name=wildcards.setup,
            model_safe_name="consensus",
        )