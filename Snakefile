import pandas as pd
from pathlib import Path

configfile: "config.yaml"

# Load groundtruth JSON files with optional subsampling
texts_dir = Path(config["texts_path"])
json_files = sorted(f.name for f in texts_dir.glob("*.json"))
target_files = json_files[:config.get("max_samples")] if config.get("max_samples") else json_files

# Consensus threshold parameter
consensus_threshold = config.get("consensus_threshold", 0.5)

# Helper to sanitize model names (org/model -> org_model)
def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_")

# Model mappings (safe_name -> full_name)
bench_models = {safe_name(m): m for m in config["benchmark_models"]}
full_models = {safe_name(m): m for m in config["full_eval_models"]}
cons_models = {safe_name(m): m for m in config["consensus_models"]}

base_out = Path(config["output_dir_base"])

# --- Target Outputs ---
eval_csv_outputs = []

# Scenario 1: Benchmark strategies
for strategy in config["benchmark_strategies"]:
    for m in bench_models:
        eval_csv_outputs.append(str(base_out / "evaluation" / "benchmark_strategies" / strategy / m / "grouped_evaluation_metrics.csv"))

# Scenario 2: Full evaluation on SoTA models
for m in full_models:
    eval_csv_outputs.append(str(base_out / "evaluation" / "benchmark_models" / "instructor_guidelines" / m / "grouped_evaluation_metrics.csv"))

# Scenario 3: Consensus runs (temp_1 et temp_1_and_2)
consensus_setups = ["temp_1", "temp_1_and_2"]
for setup in consensus_setups:
    eval_csv_outputs.append(str(base_out / "evaluation" / "consensus" / setup / "grouped_evaluation_metrics.csv"))


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
    wildcard_constraints:
        scenario="(?!consensus_raw).*"  # Empêche Snakemake d'associer 'consensus_raw' à cette règle
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


# ================================================
# SCENARIO 3: CONSENSUS & TEMPERATURE RUNS
# ================================================

rule extract_consensus_runs:
    input:
        texts_files=expand("{path}/{json_file}", path=texts_dir, json_file=target_files),
        prompt=config["prompt_path"],
        guidelines=config["guidelines_path"],
        examples=config["examples_path"]
    output:
        out_dir=directory("{base}/inferences/consensus_raw/temp_{temp}/{model_safe}")
    params:
        model=lambda wildcards: cons_models[wildcards.model_safe],
        temp=lambda wildcards: wildcards.temp
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

rule aggregate_consensus_temp_1:
    input:
        inferences=expand(
            "{base}/inferences/consensus_raw/temp_1/{model_safe}",
            base=base_out,
            model_safe=cons_models.keys()
        )
    output:
        merged_dir=directory("{base}/inferences_consensus/temp_1"),
        consensus_dir=directory("{base}/inferences/consensus_aggregated/temp_1")
    params:
        threshold=consensus_threshold
    shell:
        """
        mkdir -p {output.merged_dir}
        mkdir -p {output.consensus_dir}
        
        for dir in {input.inferences}; do
            cp "$dir"/*.json {output.merged_dir}/
        done
        
        uv run aggregate-consensus-entities \
            --inferences-dir {output.merged_dir} \
            --threshold {params.threshold} \
            --output-dir {output.consensus_dir}
        
        rm -rf {output.merged_dir}
        """

rule aggregate_consensus_temp_1_and_2:
    input:
        inferences_t1=expand(
            "{base}/inferences/consensus_raw/temp_1/{model_safe}",
            base=base_out,
            model_safe=cons_models.keys()
        ),
        inferences_t2=expand(
            "{base}/inferences/consensus_raw/temp_2/{model_safe}",
            base=base_out,
            model_safe=cons_models.keys()
        )
    output:
        merged_dir=directory("{base}/inferences_consensus/temp_1_and_2"),
        consensus_dir=directory("{base}/inferences/consensus_aggregated/temp_1_and_2")
    params:
        threshold=consensus_threshold
    shell:
        """
        mkdir -p {output.merged_dir}
        mkdir -p {output.consensus_dir}
        
        for dir in {input.inferences_t1} {input.inferences_t2}; do
            cp "$dir"/*.json {output.merged_dir}/
        done
        
        uv run aggregate-consensus-entities \
            --inferences-dir {output.merged_dir} \
            --threshold {params.threshold} \
            --output-dir {output.consensus_dir}
    
        rm -rf {output.merged_dir}
        """

rule normalize_consensus:
    input:
        inferences_dir="{base}/inferences/consensus_aggregated/{setup}",
        ffm_db=config["ffm_db_path"],
        softname_db=config["softname_db_path"]
    output:
        norm_dir=directory("{base}/inferences_normalized/consensus/{setup}")
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

rule evaluate_consensus:
    input:
        inferences_dir="{base}/inferences_normalized/consensus/{setup}"
    output:
        eval_csv="{base}/evaluation/consensus/{setup}/grouped_evaluation_metrics.csv"
    shell:
        """
        uv run evaluate-entities-extraction \
            --inferences-dir {input.inferences_dir} \
            --results-dir $(dirname {output.eval_csv})
        """


# ================================================
# FINAL METRICS AGGREGATION
# ================================================

rule aggregate_evaluations:
    input:
        eval_csv_outputs
    output:
        combined_csv=str(base_out / "evaluation" / "eval_all_models.csv")
    run:
        dfs = []
        for csv_path in input:
            p = Path(csv_path)
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