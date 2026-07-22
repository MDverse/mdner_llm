import pandas as pd
from pathlib import Path

configfile: "config_test.yaml"

# Load groundtruth JSON files with optional subsampling
texts_dir = Path(config["texts_path"])
json_files = sorted(f.name for f in texts_dir.glob("*.json"))
target_files = json_files[:config.get("max_samples")] if config.get("max_samples") else json_files

# Helper to sanitize model names (org/model -> org_model)
def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_")

# Model mappings (safe_name -> full_name)
bench_models = {safe_name(m): m for m in config["benchmark_models"]}
full_models = {safe_name(m): m for m in config["full_eval_models"]}
cons_models = {safe_name(m): m for m in config["consensus_models"]}


# --- Target Outputs ---
eval_csv_outputs = []
base_out = Path(config["output_dir_base"])
# Scenario 1: Benchmark strategies
for strategy in config["benchmark_strategies"]:
    for m in bench_models:
        eval_csv_outputs.append(str(base_out / "evaluation" / "benchmark_strategies" / strategy / m / "grouped_evaluation_metrics.csv"))
# Scenario 2: Full evaluation on SoTA models
for m in full_models:
    eval_csv_outputs.append(str(base_out / "evaluation" / "benchmark_models" / "instructor_guidelines" / m / "grouped_evaluation_metrics.csv"))

rule all:
    input: 
        str(base_out / "evaluation" / "eval_all_models.csv")


# ================================================
# SCENARIO 1 & 2: BENCHMARK STRATEGIES AND MODELS
# ================================================


rule extract_benchmark_and_full:
    input:
        texts_files=expand("{path}/{json_file}", path=texts_dir, json_file=target_files),
        prompt=config["prompt_path"],
        examples=config["examples_path"]
    output:
        out_dir=directory("{base}/inferences/{scenario}/{combo}/{model_safe}")
    params:
        model=lambda wildcards: bench_models.get(wildcards.model_safe) or full_models.get(wildcards.model_safe),
        framework=lambda wildcards: config["benchmark_strategies"].get(wildcards.combo, {}).get("framework", "instructor"),
        guidelines=lambda wildcards: config["benchmark_strategies"].get(wildcards.combo, {}).get("guidelines", config["guidelines_path"])
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

rule normalize_benchmark_and_full:
    input:
        inferences_dir="{base}/inferences/{scenario}/{combo}/{model_safe}",
        ffm_db=config["ffm_db_path"],
        softname_db=config["softname_db_path"]
    output:
        norm_dir=directory("{base}/inferences_normalized/{scenario}/{combo}/{model_safe}")
    params:
        norm_model=config["normalization_model"]
    shell:
        """
        uv run normalize-extracted-entities \
            --inferences-dir {input.inferences_dir} \
            --ffm-db-path {input.ffm_db} \
            --softname-db-path {input.softname_db} \
            --model-name "{params.norm_model}" \
            --output-dir {output.norm_dir}
        """

rule evaluate_benchmark_and_full:
    input:
        inferences_dir="{base}/inferences_normalized/{scenario}/{combo}/{model_safe}"
    output:
        eval_csv="{base}/evaluation/{scenario}/{combo}/{model_safe}/grouped_evaluation_metrics.csv"
    shell:
        """
        uv run evaluate-entities-extraction \
            --inferences-dir {input.inferences_dir} \
            --results-dir $(dirname {output.eval_csv})
        """

rule aggregate_evaluations:
    input:
        eval_csv_outputs
    output:
        combined_csv=str(base_out / "evaluation" / "eval_all_models.csv")
    run:
        dfs = []
        for csv_path in input:
            p = Path(csv_path)
            # p.parts donne: ('results', 'evaluation', scenario, combo, model_safe, 'grouped_evaluation_metrics.csv')
            scenario = p.parts[-4]
            combo = p.parts[-3]
            model_safe = p.parts[-2]

            df = pd.read_csv(csv_path)
            df["scenario"] = scenario
            df["combo"] = combo
            df["model_safe"] = model_safe
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(output.combined_csv, index=False)