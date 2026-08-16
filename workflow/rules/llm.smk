"""Snakefile for LLM benchmark strategies, models, and consensus evaluation."""

import json
import shutil
from pathlib import Path
import pandas as pd

from mdner_llm.common import sanitize_filename


# Load pipeline configuration file.
configfile: "workflow/configs/llm.yaml"
# Global paths and inputs configuration.
texts_directory = Path(config["texts_path"])
json_files_list = sorted(file_path.name for file_path in texts_directory.glob("*.json"))
target_files = json_files_list[:config.get("max_samples")] if config.get("max_samples") else json_files_list
consensus_threshold_value = config.get("consensus_threshold", 0.5)
base_output_directory = Path(config["output_dir_base"])
# Model dictionaries mapping safe filesystem names to full identifiers.
benchmark_models = {sanitize_filename(model_name): model_name for model_name in config["benchmark_models"]}
full_eval_models = {sanitize_filename(model_name): model_name for model_name in config["full_eval_models"]}
consensus_models = {sanitize_filename(model_name): model_name for model_name in config["consensus_models"]}


# Build target paths for per-scenario evaluations.
evaluation_targets = []
# Scenario 1: Benchmark prompting strategies.
for strategy_name in config["benchmark_strategies"]:
    for safe_model_name in benchmark_models:
        evaluation_targets.append(
            base_output_directory / "evaluation" / "benchmark_strategies" / strategy_name / safe_model_name / "grouped_evaluation_metrics.csv"
        )
# Scenario 2: Full evaluation on benchmark and reference models.
for safe_model_name in full_eval_models:
    evaluation_targets.append(
        base_output_directory / "evaluation" / "benchmark_models" / "with_instructor_with_guidelines" / safe_model_name / "grouped_evaluation_metrics.csv"
    )
# Scenario 3: Consensus aggregation runs.
consensus_temperatures = config.get("consensus_temperatures")
consensus_setups = [
    f"temp_{'_and_'.join(str(temp) for temp in consensus_temperatures[:end_idx])}"
    for end_idx in range(1, len(consensus_temperatures) + 1)
]
for setup_identifier in consensus_setups:
    evaluation_targets.append(
        base_output_directory / "evaluation" / "consensus" / setup_identifier / "grouped_evaluation_metrics.csv"
    )


rule llm_all:
    input:
        benchmark_strategies_csv=base_output_directory / "evaluation" / "benchmark_strategies.csv",
        benchmark_models_csv=base_output_directory / "evaluation" / "benchmark_models.csv",
        all_grouped_csv=base_output_directory / "evaluation" / "all_grouped_evaluation_metrics.csv",
        all_detailed_parquet=base_output_directory / "evaluation" / "all_per_text_and_category_confusion_metrics.parquet",


# ==============================================================================
# SCENARIOS 1 & 2: EXTRACTION, NORMALIZATION, AND EVALUATION
# ==============================================================================

# Extract entities.
rule extract_benchmark_and_full:
    input:
        texts_files=expand("{path}/{json_file}", path=texts_directory, json_file=target_files),
        prompt=config["prompt_path"],
        examples=config["examples_path"],
    output:
        out_dir=directory("results/llm/inferences/{scenario}/{combo}/{model_safe}"),
    wildcard_constraints:
        scenario="(?!consensus_raw).*",  # To ensure this rule does not match consensus runs.
    params:
        model=lambda wildcards: benchmark_models.get(wildcards.model_safe) or full_eval_models.get(wildcards.model_safe),
        framework=lambda wildcards: config["benchmark_strategies"].get(wildcards.combo, {}).get("framework", "instructor"),
        guidelines=lambda wildcards: config["benchmark_strategies"].get(wildcards.combo, {}).get("guidelines", config["guidelines_path"]),
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
                --output-dir {output.out_dir}
        done
        """

# Normalize extracted entities.
rule normalize_benchmark_and_full:
    input:
        inferences_dir="results/llm/inferences/{scenario}/{combo}/{model_safe}",
        ffm_db=config["ffm_db_path"],
        softname_db=config["softname_db_path"],
    output:
        norm_dir=directory("results/llm/inferences_normalized/{scenario}/{combo}/{model_safe}"),
    params:
        norm_model=config["normalization_model"],
    shell:
        """
        uv run normalize-extracted-entities \
            --inferences-dir {input.inferences_dir} \
            --ffm-db-path {input.ffm_db} \
            --softname-db-path {input.softname_db} \
            --model-name "{params.norm_model}" \
            --output-dir {output.norm_dir}
        """

# Evaluate normalized predictions against ground truth.
rule evaluate_benchmark_and_full:
    input:
        inferences_dir="results/llm/inferences_normalized/{scenario}/{combo}/{model_safe}",
    output:
        eval_csv="results/llm/evaluation/{scenario}/{combo}/{model_safe}/grouped_evaluation_metrics.csv",
        eval_parquet="results/llm/evaluation/{scenario}/{combo}/{model_safe}/per_text_and_category_confusion_metrics.parquet",
    shell:
        """
        uv run evaluate-entities-extraction \
            --inferences-dir {input.inferences_dir} \
            --results-dir $(dirname {output.eval_csv})
        """


# ==============================================================================
# SCENARIO 3: CONSENSUS RUNS AND MULTI-TEMPERATURE AGGREGATIONS
# ==============================================================================

# Helper to resolve input directories based on the specified temperature setup.
def get_consensus_inferences_input(wildcards) -> list[str]:
    """Resolve raw inference paths for each temperature."""
    # wildcards.setup is e.g. "temp_1.0" or "temp_1.0_and_2.0"
    temperatures_string = wildcards.setup.replace("temp_", "")
    included_temperatures = temperatures_string.split("_and_")
    
    input_directories = []
    for temperature_value in included_temperatures:
        for model_identifier in consensus_models.keys():
            input_directories.append(
                f"results/llm/inferences/consensus_raw/temp_{temperature_value}/{model_identifier}"
            )
    return input_directories


# Extract entities across varying temperature settings.
rule extract_consensus_runs:
    input:
        texts_files=expand("{path}/{json_file}", path=texts_directory, json_file=target_files),
        prompt=config["prompt_path"],
        guidelines=config["guidelines_path"],
        examples=config["examples_path"],
    output:
        out_dir=directory("results/llm/inferences/consensus_raw/temp_{temp}/{model_safe}"),
    params:
        model=lambda wildcards: consensus_models[wildcards.model_safe],
        temp=lambda wildcards: wildcards.temp,
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
        """


# Aggregate consensus predictions across multiple temperature runs.
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

        # Enable nullglob so empty patterns do not throw errors.
        shopt -s nullglob

        # Copy JSON predictions from all model inference directories into staging.
        for source_directory in {input.inferences}; do
            if [ -d "$source_directory" ]; then
                cp "$source_directory"/*.json {params.staging_dir}/ 2>/dev/null || true
            fi
        done

        # Run the consensus aggregation CLI tool.
        uv run aggregate-consensus-entities \
            --inferences-dir {params.staging_dir} \
            --threshold {params.threshold} \
            --output-dir {output.consensus_dir}

        # Clean up temporary staging directory.
        rm -rf {params.staging_dir}
        rmdir results/llm/inferences_consensus_staging 2>/dev/null || true
        """

# Normalize consensus extracted entities against registries.
rule normalize_consensus:
    input:
        inferences_dir="results/llm/inferences/consensus_aggregated/{setup}",
        ffm_db=config["ffm_db_path"],
        softname_db=config["softname_db_path"],
    output:
        norm_dir=directory("results/llm/inferences_normalized/consensus/{setup}"),
    params:
        norm_model=config["normalization_model"],
    shell:
        """
        uv run normalize-extracted-entities \
            --inferences-dir {input.inferences_dir} \
            --ffm-db-path {input.ffm_db} \
            --softname-db-path {input.softname_db} \
            --model-name "{params.norm_model}" \
            --output-dir {output.norm_dir}
        """

# Evaluate consensus prediction quality.
rule evaluate_consensus:
    input:
        inferences_dir="results/llm/inferences_normalized/consensus/{setup}",
    output:
        eval_csv="results/llm/evaluation/consensus/{setup}/grouped_evaluation_metrics.csv",
        eval_parquet="results/llm/evaluation/consensus/{setup}/per_text_and_category_confusion_metrics.parquet",
    shell:
        """
        uv run evaluate-entities-extraction \
            --inferences-dir {input.inferences_dir} \
            --results-dir $(dirname {output.eval_csv})
        """


# ==============================================================================
# GLOBAL AGGREGATION & CLEANUP
# ==============================================================================

def serialize_cell(val):
    # First, value is a numpy array or similar object.
    if hasattr(val, "tolist"):
        return json.dumps(val.tolist())
    # Second, value is a list or dictionary.
    if isinstance(val, (list, dict)):
        return json.dumps(
            val,
            default=lambda obj: obj.tolist() if hasattr(obj, "tolist") else str(obj),
        )
    # Third, value is a pandas NA or None.
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (ValueError, TypeError):
        pass
    return str(val)


rule aggregate_all_evaluations:
    input:
        evaluation_csv_files=evaluation_targets,
    output:
        benchmark_strategies_csv=base_output_directory / "evaluation" / "benchmark_strategies.csv",
        benchmark_models_csv=base_output_directory / "evaluation" / "benchmark_models.csv",
        all_grouped_csv=base_output_directory / "evaluation" / "all_grouped_evaluation_metrics.csv",
        all_detailed_parquet=base_output_directory / "evaluation" / "all_per_text_and_category_confusion_metrics.parquet",
    run:
        grouped_dataframes = []
        detailed_dataframes = []

        # Parse every evaluated CSV and corresponding parquet file.
        for csv_path in input.evaluation_csv_files:
            file_path = Path(csv_path)
            parquet_path = file_path.with_name("per_text_and_category_confusion_metrics.parquet")
            # Extract scenario hierarchy components from folder path.
            if "consensus" in file_path.parts:
                scenario_name = "consensus"
                combo_name = file_path.parent.name
                model_safe_name = "consensus"
            else:
                scenario_name = file_path.parts[-4]
                combo_name = file_path.parts[-3]
                model_safe_name = file_path.parts[-2]
            # Read the grouped evaluation metrics CSV and append scenario metadata.
            current_grouped_dataframe = pd.read_csv(file_path)
            current_grouped_dataframe["scenario"] = scenario_name
            current_grouped_dataframe["combo"] = combo_name
            current_grouped_dataframe["model_safe"] = model_safe_name
            grouped_dataframes.append(current_grouped_dataframe)
            # Read the detailed confusion metrics parquet file if it exists and append scenario metadata.
            if parquet_path.exists():
                current_detailed_dataframe = pd.read_parquet(parquet_path)
                current_detailed_dataframe["scenario"] = scenario_name
                current_detailed_dataframe["combo"] = combo_name
                current_detailed_dataframe["model_safe"] = model_safe_name
                detailed_dataframes.append(current_detailed_dataframe)
        # Merge grouped metrics and detailed confusion metrics.
        full_grouped_dataframe = pd.concat(grouped_dataframes, ignore_index=True)
        full_grouped_dataframe.to_csv(output.all_grouped_csv, index=False)
        # Merge detailed confusion metrics if any exist.
        if detailed_dataframes:
            full_detailed_dataframe = pd.concat(detailed_dataframes, ignore_index=True)
            # Convert complex/nested objects to JSON strings for Parquet compatibility.
            for column_name in full_detailed_dataframe.columns:
                if full_detailed_dataframe[column_name].dtype == "object":
                    full_detailed_dataframe[column_name] = full_detailed_dataframe[
                        column_name
                    ].apply(serialize_cell)
            full_detailed_dataframe.to_parquet(output.all_detailed_parquet, index=False)
        # Standardize column names mapping for report tables.
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
        working_dataframe = full_grouped_dataframe.rename(columns=column_mapping).copy()
        # Compute cost and latency metrics per entity without rounding.
        total_predictions = (
            working_dataframe["nb_predicted_entities_raw"].replace(0, pd.NA)
            if "nb_predicted_entities_raw" in working_dataframe
            else working_dataframe["Number_of_predicted_entities"].replace(0, pd.NA)
        )
        working_dataframe["Cost_by_entity_($)"] = working_dataframe["Cost_total_($)"] / total_predictions
        working_dataframe["Inference_time_by_entity_(s)"] = working_dataframe["Inference_time_total_(s)"] / total_predictions
        working_dataframe["Inference_time_total_(hh:mm:ss)"] = working_dataframe["Inference_time_total_(s)"].apply(
            lambda seconds_value: str(pd.to_timedelta(int(seconds_value), unit="s")).split()[-1]
            if pd.notna(seconds_value)
            else pd.NA
        )
        # Build benchmark_strategies.csv table.
        strategies_mask = working_dataframe["scenario"] == "benchmark_strategies"
        strategies_dataframe = working_dataframe[strategies_mask].copy()
        # Derive boolean flag for guidelines from strategy combo names.
        strategies_dataframe["With_Guideline"] = strategies_dataframe["combo"].apply(
            lambda combo_string: True if "with_guidelines" in str(combo_string) or "guidelines" in str(combo_string) else False
        )
        columns_order = [
            "Inference_date",
            "Name",
            "Framework",
            "With_Guideline",
            "category",
            "Number_of_texts_with_category",
            "Correct_format_(%)",
            "Hallucinations_(%)",
            "Precision",
            "Precision_with_no_hallucination",
            "Recall",
            "F1",
            "F1_with_no_hallucination",
            "Fbeta_0.5",
            "Fbeta_0.5_with_no_hallucination",
            "Cost_by_entity_($)",
            "Inference_time_by_entity_(s)",
            "Number_of_predicted_entities",
            "Cost_total_($)",
            "Inference_time_total_(s)",
            "Inference_time_total_(hh:mm:ss)",
        ]
        available_strategy_columns = [column for column in columns_order if column in strategies_dataframe.columns]
        strategies_dataframe[available_strategy_columns].to_csv(output.benchmark_strategies_csv, index=False)
        # Build benchmark_models.csv table.
        models_mask = working_dataframe["scenario"].isin(["benchmark_models", "consensus"])
        models_dataframe = working_dataframe[models_mask].copy()
        models_columns_order = [
            column for column in columns_order if column not in ("Framework", "With_Guideline")
        ]
        available_model_columns = [column for column in models_columns_order if column in models_dataframe.columns]
        models_dataframe[available_model_columns].to_csv(output.benchmark_models_csv, index=False)
        # Clean up intermediate evaluation folders.
        redundant_folders = [
            base_output_directory / "evaluation" / "benchmark_strategies",
            base_output_directory / "evaluation" / "benchmark_models",
            base_output_directory / "evaluation" / "consensus",
        ]
        for folder_path in redundant_folders:
            if folder_path.exists() and folder_path.is_dir():
                shutil.rmtree(folder_path)