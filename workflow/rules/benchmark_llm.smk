# Snakefile for the LLM benchmark.
#
# Scenarios (configured in workflow/configs/llm_models.yaml):
#   1. benchmark_strategies: a few models x 4 prompting strategies (with and without instructor/guidelines).
#   2. benchmark_models: many models with the best strategy (with instructor with guidelines).
#   3. determinism: extra runs of some models to measure the variability of their outputs.
#   4. consensus: vote of small models (consensus), and single runs of these
#      models at other temperatures (default is T=1).
#
# Every scenario follows the same steps:
# Extraction -> Normalization -> Evaluation -> update the summary files.
import re
from itertools import combinations
from pathlib import Path

import pandas as pd
from filelock import FileLock

from mdner_llm.common import sanitize_filename


# ==============================================================================
# CONFIGURATION
# ==============================================================================
configfile: "workflow/configs/llm_models.yaml"
# List of texts to extract entities from, in JSON format (one text per file).
ALL_TEXT_FILES = [
    str(path) for path in sorted(Path(config["texts_path"]).glob("*.json"))
]
# Limit the number of texts to extract. 
TEXT_FILES = ALL_TEXT_FILES[: config.get("max_samples", len(ALL_TEXT_FILES))]
# Sleep time between API calls, in seconds.
# This is to avoid hitting the rate limit of the LLM API.
API_SLEEP = config.get("api_sleep_delay", 5)
# Strategy used by scenarios 2, 3 and 4.
BEST_STRATEGY = "with_instructor_with_guidelines"
EVAL_DIR = "results/llm/evaluation"


def by_safe_name(models) -> dict[str, str]:
    """Map filesystem-safe names to model ids."""
    return {sanitize_filename(model): model for model in models}

# Models used in each scenario.
STRATEGY_MODELS = by_safe_name(config.get("benchmark_models", []))
FULL_EVAL_MODELS = by_safe_name(config.get("full_eval_models", []))
DETERMINISM_MODELS = by_safe_name(config.get("determinism_models", []))
DETERMINISM_NB_RUNS = int(config.get("determinism_nb_runs", 5))
CONSENSUS_GROUPS = config.get("consensus_groups", {})
CONSENSUS_MODELS = by_safe_name(
    model for models in CONSENSUS_GROUPS.values() for model in models
)
ALL_MODELS = {**STRATEGY_MODELS, **FULL_EVAL_MODELS, **CONSENSUS_MODELS, **DETERMINISM_MODELS}
# "{:g}" removes trailing zeros so that 1.0 and 1 give the same label "1".
CONSENSUS_TEMPERATURES = [f"{float(t):g}" for t in config.get("consensus_temperatures", [1])]
CONSENSUS_THRESHOLDS = [f"{float(t):g}" for t in config.get("consensus_thresholds", [0.5])]
# All non-empty subsets of temperatures, e.g. temp_1, temp_1_and_1.5.
CONSENSUS_SETUPS = [
    f"temp_{'_and_'.join(subset)}"
    for size in range(1, len(CONSENSUS_TEMPERATURES) + 1)
    for subset in combinations(CONSENSUS_TEMPERATURES, size)
]


# ==============================================================================
# TARGETS
# ==============================================================================

TARGETS = [
    # 1. Prompting strategies.
    *[
        f"{EVAL_DIR}/benchmark_strategies/{strategy}/{model}/.done"
        for strategy in config["benchmark_strategies"]
        for model in STRATEGY_MODELS
    ],
    # 2. Models with the best strategy.
    *[f"{EVAL_DIR}/benchmark_models/{BEST_STRATEGY}/{model}/.done" for model in FULL_EVAL_MODELS],
    # 3. Determinism (mean and std are added to benchmark_models.csv).
    *([f"{EVAL_DIR}/determinism_per_run.csv"] if DETERMINISM_MODELS else []),
    # 4. Consensus of small models...
    *[
        f"{EVAL_DIR}/consensus/{group}/{setup}/th_{threshold}/.done"
        for group in CONSENSUS_GROUPS
        for threshold in CONSENSUS_THRESHOLDS
        for setup in CONSENSUS_SETUPS
    ],
    # ... and their single runs (T = 1 is already the scenario 2 run).
    *[
        f"{EVAL_DIR}/consensus_raw/temp_{temp}/{model}/.done"
        for model in CONSENSUS_MODELS
        for temp in CONSENSUS_TEMPERATURES
        if temp != "1"
    ],
]


# Path separators stay unambiguous.
wildcard_constraints:
    combo="[^/]+",
    model_safe="[^/]+",
    group="[^/]+",
    setup="temp_[^/]+",
    threshold="[0-9.]+",
    # Scenarios with one run per (combo, model), consensus is handled apart.
    scenario="benchmark_strategies|benchmark_models|determinism|consensus_raw",
    raw_dir="raw|consensus_raw",


rule run_benchmark_llm:
    input:
        TARGETS,


# ==============================================================================
# 1. EXTRACTION (LLM API)
# ==============================================================================

def extraction_settings(wildcards) -> dict:
    """Return the framework, guidelines and temperature of an extraction.

    - raw/{combo}: combo is a strategy name, with an optional "_run{N}" suffix
      for the determinism runs. Temperature 1.
    - consensus_raw/temp_{T}: best strategy at temperature T.
    """
    if wildcards.raw_dir == "consensus_raw":
        return {
            "framework": "instructor",
            "guidelines": config["guidelines_path"],
            "temperature": wildcards.combo.removeprefix("temp_"),
        }
    # Remove the "_run{N}" suffix to get the strategy name, 
    # e.g. "with_instructor_with_guidelines_run2" -> "with_instructor_with_guidelines".
    strategy = config["benchmark_strategies"].get(re.sub(r"_run\d+$", "", wildcards.combo), {})
    return {
        "framework": strategy.get("framework", "instructor"),
        "guidelines": strategy.get("guidelines", config["guidelines_path"]),
        "temperature": "1",
    }


rule extract_entities:
    input:
        texts_files=TEXT_FILES,
        prompt=config["prompt_path"],
        examples=config["examples_path"],
    output:
        done=touch("results/llm/inferences/{raw_dir}/{combo}/{model_safe}/.done"),
    params:
        out_dir="results/llm/inferences/{raw_dir}/{combo}/{model_safe}",
        model=lambda wildcards: ALL_MODELS[wildcards.model_safe],
        framework=lambda wildcards: extraction_settings(wildcards)["framework"],
        guidelines=lambda wildcards: extraction_settings(wildcards)["guidelines"],
        temperature=lambda wildcards: extraction_settings(wildcards)["temperature"],
        sleep_time=API_SLEEP,
    shell:
        """
        mkdir -p {params.out_dir}
        for file in {input.texts_files}; do
            # Skip texts already extracted (resume after an interruption).
            stem=$(basename "$file" .json)
            if compgen -G "{params.out_dir}/${{stem}}_*.json" > /dev/null || \
               compgen -G "{params.out_dir}/${{stem}}_*.txt" > /dev/null; then
                continue
            fi
            uv run extract-entities-with-llm \
                --text-path "$file" \
                --model "{params.model}" \
                --prompt-path {input.prompt} \
                --guidelines-path {params.guidelines} \
                --examples-path {input.examples} \
                --framework {params.framework} \
                --temperature {params.temperature} \
                --output-dir {params.out_dir}
        done
        sleep {params.sleep_time}
        """


# ==============================================================================
# 2. CONSENSUS VOTE (scenario 4 only)
# ==============================================================================

def consensus_inputs(wildcards) -> list[str]:
    """Return the raw inferences of every model and temperature of a consensus."""
    # Temperatures of this setup, e.g. "temp_1_and_1.5" -> ["1", "1.5"].
    temperatures = wildcards.setup.removeprefix("temp_").split("_and_")
    return [
        # Temperature 1 reuses the scenario 2 run.
        f"results/llm/inferences/raw/{BEST_STRATEGY}/{model}/.done"
        if temperature == "1"
        else f"results/llm/inferences/consensus_raw/temp_{temperature}/{model}/.done"
        for temperature in temperatures
        for model in by_safe_name(CONSENSUS_GROUPS[wildcards.group])
    ]


rule aggregate_consensus:
    input:
        done_files=consensus_inputs,
    output:
        consensus_dir=directory("results/llm/inferences/consensus_aggregated/{group}/{setup}/th_{threshold}"),
    params:
        staging_dir="results/llm/inferences_consensus_staging/{group}_{setup}_th_{threshold}",
    shell:
        """
        rm -rf {params.staging_dir}
        mkdir -p {params.staging_dir} {output.consensus_dir}
        # Copy all inferences in one folder, prefixed by their source folder.
        # Example: "results/llm/inferences/raw/with_instructor_with_guidelines/gpt-3.5-turbo/zenodo_1234.json"
        for done_file in {input.done_files}; do
            source_dir=$(dirname "$done_file")
            folder_tag=$(echo "$source_dir" | tr '/' '_')
            for json_file in "$source_dir"/*.json; do
                [ -e "$json_file" ] || continue
                cp "$json_file" "{params.staging_dir}/${{folder_tag}}__$(basename "$json_file")"
            done
        done
        uv run aggregate-consensus-entities \
            --inferences-dir {params.staging_dir} \
            --threshold {wildcards.threshold} \
            --output-dir {output.consensus_dir}
        rm -rf {params.staging_dir}
        """


# ==============================================================================
# 3. NORMALIZATION (LLM API for STIME)
# ==============================================================================

NORMALIZE_SHELL = """
    mkdir -p {output.norm_dir}
    uv run normalize-extracted-entities \
        --inferences-dir {params.inferences_dir} \
        --ffm-db-path {input.ffm_db} \
        --softname-db-path {input.softname_db} \
        --model-name "{params.norm_model}" \
        --output-dir {output.norm_dir}
    if [ -z "$(find {output.norm_dir} -maxdepth 1 -name '*.json')" ]; then
        echo "[ERROR] No normalized files produced in {output.norm_dir}." >&2
        rm -rf {output.norm_dir}
        exit 1
    fi
    sleep {params.sleep_time}
    """


def raw_inferences_dir(wildcards) -> str:
    """Return the raw inferences folder of an evaluation (scenarios 1 to 3, consensus_raw)."""
    raw_dir = "consensus_raw" if wildcards.scenario == "consensus_raw" else "raw"
    return f"results/llm/inferences/{raw_dir}/{wildcards.combo}/{wildcards.model_safe}"


rule normalize_entities:
    input:
        done=lambda wildcards: f"{raw_inferences_dir(wildcards)}/.done",
        ffm_db=config["ffm_db_path"],
        softname_db=config["softname_db_path"],
    output:
        norm_dir=directory("results/llm/inferences_normalized/{scenario}/{combo}/{model_safe}"),
    params:
        inferences_dir=raw_inferences_dir,
        norm_model=config["normalization_model"],
        sleep_time=API_SLEEP,
    resources:
        api_calls=1,
    shell:
        NORMALIZE_SHELL


rule normalize_consensus:
    input:
        inferences_dir="results/llm/inferences/consensus_aggregated/{group}/{setup}/th_{threshold}",
        ffm_db=config["ffm_db_path"],
        softname_db=config["softname_db_path"],
    output:
        norm_dir=directory("results/llm/inferences_normalized/consensus/{group}/{setup}/th_{threshold}"),
    params:
        inferences_dir="results/llm/inferences/consensus_aggregated/{group}/{setup}/th_{threshold}",
        norm_model=config["normalization_model"],
        sleep_time=API_SLEEP,
    resources:
        api_calls=1,
    shell:
        NORMALIZE_SHELL


# ==============================================================================
# 4. EVALUATION
# ==============================================================================

EVALUATE_SHELL = """
    mkdir -p {params.results_dir}
    uv run evaluate-entities-extraction \
        --inferences-dir {input.inferences_dir} \
        --results-dir {params.results_dir}
    """


rule evaluate_entities:
    input:
        inferences_dir="results/llm/inferences_normalized/{scenario}/{combo}/{model_safe}",
    output:
        eval_csv=f"{EVAL_DIR}/{{scenario}}/{{combo}}/{{model_safe}}/grouped_evaluation_metrics.csv",
        eval_parquet=f"{EVAL_DIR}/{{scenario}}/{{combo}}/{{model_safe}}/per_text_and_category_confusion_metrics.parquet",
        done=touch(f"{EVAL_DIR}/{{scenario}}/{{combo}}/{{model_safe}}/.done"),
    params:
        results_dir=f"{EVAL_DIR}/{{scenario}}/{{combo}}/{{model_safe}}",
    run:
        shell(EVALUATE_SHELL)
        update_summaries(
            output.eval_csv, output.eval_parquet,
            wildcards.scenario, wildcards.combo, wildcards.model_safe,
        )


rule evaluate_consensus:
    input:
        inferences_dir="results/llm/inferences_normalized/consensus/{group}/{setup}/th_{threshold}",
    output:
        eval_csv=f"{EVAL_DIR}/consensus/{{group}}/{{setup}}/th_{{threshold}}/grouped_evaluation_metrics.csv",
        eval_parquet=f"{EVAL_DIR}/consensus/{{group}}/{{setup}}/th_{{threshold}}/per_text_and_category_confusion_metrics.parquet",
        done=touch(f"{EVAL_DIR}/consensus/{{group}}/{{setup}}/th_{{threshold}}/.done"),
    params:
        results_dir=f"{EVAL_DIR}/consensus/{{group}}/{{setup}}/th_{{threshold}}",
    run:
        shell(EVALUATE_SHELL)
        update_summaries(
            output.eval_csv, output.eval_parquet,
            "consensus",
            f"consensus_{wildcards.group}_{wildcards.setup}_th_{wildcards.threshold}",
            wildcards.group,
        )


# ==============================================================================
# 5. DETERMINISM: ALL RUNS IN ONE FILE
# ==============================================================================

def determinism_eval_csvs(wildcards) -> list[str]:
    """Return the evaluation CSV of every run of every determinism model."""
    # Run 1 is the scenario 2 run.
    run_dirs = [f"{EVAL_DIR}/benchmark_models/{BEST_STRATEGY}"] + [
        f"{EVAL_DIR}/determinism/{BEST_STRATEGY}_run{run}"
        for run in range(2, DETERMINISM_NB_RUNS + 1)
    ]
    return [
        f"{run_dir}/{model}/grouped_evaluation_metrics.csv"
        for model in DETERMINISM_MODELS
        for run_dir in run_dirs
    ]


rule summarize_determinism:
    input:
        eval_csvs=determinism_eval_csvs,
    output:
        per_run_csv=f"{EVAL_DIR}/determinism_per_run.csv",
    run:
        frames = []
        for csv_path in input.eval_csvs:
            # Run number from the combo folder, e.g. "..._run2" -> 2, else run 1.
            match = re.search(r"_run(\d+)$", Path(csv_path).parent.parent.name)
            frames.append(pd.read_csv(csv_path).assign(run=int(match.group(1)) if match else 1))
        pd.concat(frames, ignore_index=True).to_csv(output.per_run_csv, index=False)


# ==============================================================================
# SUMMARY FILES (updated after each evaluation)
# ==============================================================================
#   all_grouped_evaluation_metrics.csv: every evaluation, original column names.
#   all_per_text_and_category_confusion_metrics.parquet: per-text details.
#   benchmark_strategies.csv: scenario 1.
#   benchmark_models.csv: scenarios 2 to 4, mean and std over runs (scenario 3).

ALL_GROUPED_CSV = Path(EVAL_DIR) / "all_grouped_evaluation_metrics.csv"
ALL_DETAILED_PARQUET = Path(EVAL_DIR) / "all_per_text_and_category_confusion_metrics.parquet"
STRATEGIES_CSV = Path(EVAL_DIR) / "benchmark_strategies.csv"
MODELS_CSV = Path(EVAL_DIR) / "benchmark_models.csv"

# Evaluation column -> summary column.
RENAMED_COLUMNS = {
    "model_name": "name",
    "framework_name": "framework",
    "nb_texts_with_category": "number_of_texts_with_category",
    "pct_correct_format": "correct_format_(%)",
    "pct_hallucinations": "hallucinations_(%)",
    "precision_no_hallucination": "precision_with_no_hallucination",
    "f1_no_hallucination": "f1_with_no_hallucination",
    "fbeta_0.5_no_hallucination": "fbeta_0.5_with_no_hallucination",
    "nb_predicted_entities": "number_of_predicted_entities",
    "total_cost_usd": "cost_total_($)",
    "total_inference_time_sec": "inference_time_total_(s)",
}
# Metrics with a standard deviation over runs.
STD_METRICS = [
    "correct_format_(%)", "hallucinations_(%)",
    "precision", "precision_with_no_hallucination", "recall",
    "f1", "f1_with_no_hallucination", "fbeta_0.5", "fbeta_0.5_with_no_hallucination",
]
# Metrics averaged over runs.
MEAN_METRICS = STD_METRICS + [
    "cost_by_entity_($)", "inference_time_by_entity_(s)", "number_of_predicted_entities",
    "cost_total_($)", "inference_time_total_(s)",
]
STRATEGIES_COLUMNS = [
    "inference_date", "name", "framework", "with_guideline", "category",
    "number_of_texts_with_category", "correct_format_(%)", "hallucinations_(%)",
    "precision", "precision_with_no_hallucination", "recall",
    "f1", "f1_with_no_hallucination", "fbeta_0.5", "fbeta_0.5_with_no_hallucination",
    "cost_by_entity_($)", "inference_time_by_entity_(s)", "number_of_predicted_entities",
    "cost_total_($)", "inference_time_total_(s)", "inference_time_total_(hh:mm:ss)",
]
MODELS_COLUMNS = [
    col for col in STRATEGIES_COLUMNS if col not in ("framework", "with_guideline")
] + ["number_of_runs"] + [f"{col}_std" for col in STD_METRICS]


def format_seconds_to_hhmmss(seconds: float | int | None) -> str | None:
    """Format total seconds into HH:MM:SS without truncating days."""
    if pd.isna(seconds):
        return pd.NA
    total_sec = int(seconds)
    return f"{total_sec // 3600:02d}:{total_sec % 3600 // 60:02d}:{total_sec % 60:02d}"


def upsert(path: Path, new_df: pd.DataFrame, scenario: str, combo: str, model_safe: str) -> pd.DataFrame:
    """Replace the rows of one evaluation in a CSV or Parquet table, and save it."""
    new_df = new_df.assign(scenario=scenario, combo=combo, model_safe=model_safe)
    is_csv = path.suffix == ".csv"
    if path.exists():
        old_df = pd.read_csv(path) if is_csv else pd.read_parquet(path)
        is_same_evaluation = (
            (old_df["scenario"] == scenario)
            & (old_df["combo"] == combo)
            & (old_df["model_safe"] == model_safe)
        )
        new_df = pd.concat([old_df[~is_same_evaluation], new_df], ignore_index=True)
    if is_csv:
        new_df.to_csv(path, index=False)
    else:
        new_df.to_parquet(path, index=False)
    return new_df


def build_summary_df(grouped_df: pd.DataFrame) -> pd.DataFrame:
    """Rename the evaluation columns and add cost and time per entity."""
    df = grouped_df.rename(columns=RENAMED_COLUMNS)
    nb_entities = df.get("nb_predicted_entities_raw", df["number_of_predicted_entities"])
    nb_entities = nb_entities.replace(0, pd.NA)
    df["cost_by_entity_($)"] = df["cost_total_($)"] / nb_entities
    df["inference_time_by_entity_(s)"] = df["inference_time_total_(s)"] / nb_entities
    df["inference_time_total_(hh:mm:ss)"] = df["inference_time_total_(s)"].apply(format_seconds_to_hhmmss)
    return df


def average_repeated_runs(models_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Replace scenario 2 metrics by their mean over all runs and add std columns.

    Run 1 is the scenario 2 run; runs 2..n come from the determinism scenario.
    Models with a single run get number_of_runs = 1 and empty std columns.
    """
    is_base = models_df["scenario"] == "benchmark_models"
    runs_df = pd.concat(
        [models_df[is_base], summary_df[summary_df["scenario"] == "determinism"]],
        ignore_index=True,
    )
    grouped = runs_df.groupby(["model_safe", "category"])
    stats_df = grouped[MEAN_METRICS].mean()  # NaN values (e.g. missing cost) are skipped
    stats_df = stats_df.join(grouped[STD_METRICS].std().add_suffix("_std"))  # ddof=1
    stats_df["number_of_runs"] = grouped.size()
    # Overwrite scenario 2 rows only; consensus rows keep their own values.
    base_df = models_df[is_base].drop(columns=MEAN_METRICS).merge(
        stats_df.reset_index(), on=["model_safe", "category"], how="left"
    )
    base_df.index = models_df.index[is_base]  # merge resets the index
    base_df["inference_time_total_(hh:mm:ss)"] = base_df["inference_time_total_(s)"].apply(
        format_seconds_to_hhmmss
    )
    other_df = models_df[~is_base].assign(number_of_runs=1)
    # Keep the original row order (display functions keep the last duplicate).
    return pd.concat([base_df, other_df]).sort_index()


def write_strategies_csv(summary_df: pd.DataFrame) -> None:
    """Write benchmark_strategies.csv (scenario 1)."""
    strategies_df = summary_df[summary_df["scenario"] == "benchmark_strategies"].copy()
    if strategies_df.empty:
        return
    strategies_df["with_guideline"] = strategies_df["combo"].str.contains("with_guidelines")
    columns = [col for col in STRATEGIES_COLUMNS if col in strategies_df.columns]
    strategies_df[columns].to_csv(STRATEGIES_CSV, index=False)


def write_models_csv(summary_df: pd.DataFrame) -> None:
    """Write benchmark_models.csv (scenarios 2 to 4)."""
    models_df = summary_df[
        summary_df["scenario"].isin(["benchmark_models", "consensus", "consensus_raw"])
    ]
    if models_df.empty:
        return
    models_df = average_repeated_runs(models_df, summary_df)
    # Scenario 2 runs of consensus models are their "T = 1" runs.
    is_t1_run = (models_df["scenario"] == "benchmark_models") & models_df["model_safe"].isin(CONSENSUS_MODELS)
    models_df.loc[is_t1_run, "name"] = models_df.loc[is_t1_run, "name"].apply(
        lambda name: name if str(name).endswith(")") else f"{name} (T = 1)"
    )
    columns = [col for col in MODELS_COLUMNS if col in models_df.columns]
    models_df[columns].to_csv(MODELS_CSV, index=False)


def update_summaries(eval_csv: str, eval_parquet: str, scenario: str, combo: str, model_safe: str) -> None:
    """Add one evaluation to the global tables, then rewrite the summary CSVs."""
    Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)
    # Lock: several evaluations can finish at the same time.
    with FileLock(f"{EVAL_DIR}/.aggregation.lock", timeout=600):
        new_df = pd.read_csv(eval_csv)
        # Names used in the plots for consensus rows.
        if scenario == "consensus":
            new_df["model_name"] = combo
        elif scenario == "consensus_raw":
            model = CONSENSUS_MODELS.get(model_safe, model_safe)
            new_df["model_name"] = f"{model} (T = {combo.removeprefix('temp_')})"
        grouped_df = upsert(ALL_GROUPED_CSV, new_df, scenario, combo, model_safe)
        if Path(eval_parquet).exists():
            upsert(ALL_DETAILED_PARQUET, pd.read_parquet(eval_parquet), scenario, combo, model_safe)
        summary_df = build_summary_df(grouped_df)
        write_strategies_csv(summary_df)
        write_models_csv(summary_df)
