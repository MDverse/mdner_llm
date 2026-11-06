import json
import os
import re
import subprocess
import time
from pathlib import Path
import pandas as pd

from mdner_llm.visualization.annotations import export_annotations_to_html
from mdner_llm.common import sanitize_filename

# Load pipeline configuration file.
configfile: "workflow/configs/mdverse_annotation.yaml"

RAW_ZENODO = config["raw_datasets"]["zenodo"]
RAW_FIGSHARE = config["raw_datasets"]["figshare"]
MDVERSE_DIR = Path(config.get("mdverse_dir", "data/mdverse"))
SELECTED_ENGINES = set(config.get("engines", []))

# Build dictionaries for quick lookup by sanitized model name.
GLINER_MODELS = {
    sanitize_filename(m.get("name", m["path"])): m["path"]
    for m in config.get("gliner_settings", {}).get("models", [])
}
GLINER_DEVICES = config.get("gliner_settings", {}).get("devices", ["cpu"])

LLM_MODELS = {
    sanitize_filename(m.get("name", m["model_name"])): m["model_name"]
    for m in config.get("llm_settings", {}).get("models", [])
}
LLM_FRAMEWORK = config.get("llm_settings", {}).get("framework", "instructor")
LLM_TEMPERATURE = config.get("llm_settings", {}).get("temperature", 1.0)


def count_dataset_repository_sources(directory_path: str):
    """Inspect generated or annotated JSON files and count sources."""
    path = Path(directory_path)
    zenodo_count = 0
    figshare_count = 0
    total_count = 0

    for file_path in path.glob("*.json"):
        total_count += 1
        name = file_path.name.lower()
        if name.startswith("zenodo_"):
            zenodo_count += 1
        elif name.startswith("figshare_"):
            figshare_count += 1

    return total_count, zenodo_count, figshare_count


def get_all_mdverse_targets(wildcards):
    """Resolve all HTML reports and benchmark logs across engines, models, and devices."""
    targets = []

    if "gliner" in SELECTED_ENGINES:
        for model_name in GLINER_MODELS:
            for device in GLINER_DEVICES:
                targets.append(f"results/mdverse/gliner/{model_name}_{device}/annotations.html")
                targets.append(f"logs/snakemake/mdverse/gliner/{model_name}_{device}.log")

    if "llm" in SELECTED_ENGINES:
        for model_name in LLM_MODELS:
            targets.append(f"results/mdverse/llm/{model_name}_api/annotations.html")
            targets.append(f"logs/snakemake/mdverse/llm/{model_name}_api.log")

    return targets


# Top-level coordinator rule for MDverse annotation pipeline.
rule run_mdverse_annotation:
    input:
        get_all_mdverse_targets,


# Extract individual JSON entries from raw Zenodo and Figshare parquet files.
rule extract_mdverse_jsons_from_parquets:
    input:
        zenodo=RAW_ZENODO,
        figshare=RAW_FIGSHARE,
    output:
        flag=touch(f"{MDVERSE_DIR}/.done"),
    params:
        out_dir=str(MDVERSE_DIR),
    run:
        output_directory = Path(params.out_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

        zenodo_df = pd.read_parquet(input.zenodo)
        figshare_df = pd.read_parquet(input.figshare)
        combined_df = pd.concat([zenodo_df, figshare_df], ignore_index=True)

        for _, row in combined_df.iterrows():
            repository = str(row["dataset_repository_name"]).strip().lower()
            dataset_id = str(row["dataset_id_in_repository"]).strip()
            json_filename = f"{repository}_{dataset_id}.json"
            target_path = output_directory / json_filename

            title = str(row.get("title", "") or "")
            description = str(row.get("description", "") or "")
            raw_text = f"{title}\n{description}".strip()

            entry_payload = {
                "raw_text": raw_text,
                "entities": [],
                "url": row.get("dataset_url_in_repository", ""),
            }
            target_path.write_text(
                json.dumps(entry_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


# Annotate MDverse JSON files using GLiNER models (supports both CPU and GPU).
rule annotate_mdverse_with_gliner:
    input:
        flag=f"{MDVERSE_DIR}/.done",
    output:
        flag=touch("results/mdverse/gliner/{model_name}_{device}/.done"),
    log:
        "logs/snakemake/mdverse/gliner/{model_name}_{device}.log",
    params:
        texts_dir=str(MDVERSE_DIR),
        model_path=lambda wildcards: GLINER_MODELS[wildcards.model_name],
        device=lambda wildcards: wildcards.device,
        out_dir="results/mdverse/gliner/{model_name}_{device}",
    resources:
        gpu_slot=1,
    run:
        Path(params.out_dir).mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if params.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

        cmd = [
            "uv", "run", "extract-entities-with-gliner-all-texts",
            "--texts-path", str(params.texts_dir),
            "--model-path", str(params.model_path),
            "--output-dir", str(params.out_dir),
            "--device", str(params.device),
        ]

        start_time = time.perf_counter()
        subprocess.run(cmd, check=True, env=env)
        elapsed_time = time.perf_counter() - start_time

        total_ds, zenodo_ds, figshare_ds = count_dataset_repository_sources(params.texts_dir)
        total_seconds = int(elapsed_time)
        hours, remainder = divmod(total_seconds, 3600)
        mins, secs = divmod(remainder, 60)
        log_content = (
            f"=== GLiNER Annotation Log ===\n"
            f"Model: {params.model_path} ({wildcards.model_name})\n"
            f"Device: {params.device.upper()}\n"
            f"Total Datasets Processed: {total_ds}\n"
            f"  - Zenodo: {zenodo_ds}\n"
            f"  - Figshare: {figshare_ds}\n"
            f"Elapsed Time: {elapsed_time:.2f} seconds ({hours}h{mins}m{secs:02d}s)\n"
            f"Average Speed: {elapsed_time / max(total_ds, 1):.2f} seconds/sample\n"
        )
        Path(log[0]).write_text(log_content, encoding="utf-8")


# Annotate MDverse JSON files using LLMs.
rule annotate_mdverse_with_llm:
    input:
        flag=f"{MDVERSE_DIR}/.done",
    output:
        flag=touch("results/mdverse/llm/{model_name}_api/.done"),
    log:
        "logs/snakemake/mdverse/llm/{model_name}_api.log",
    params:
        texts_dir=str(MDVERSE_DIR),
        model_name=lambda wildcards: LLM_MODELS[wildcards.model_name],
        framework=LLM_FRAMEWORK,
        temperature=LLM_TEMPERATURE,
        out_dir="results/mdverse/llm/{model_name}_api",
    resources:
        gpu_slot=1,
    run:
        Path(params.out_dir).mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv", "run", "extract-entities-with-llm-all-texts",
            "--texts-path", str(params.texts_dir),
            "--model-name", str(params.model_name),
            "--framework-name", str(params.framework),
            "--temperature", str(params.temperature),
            "--output-dir", str(params.out_dir),
        ]

        start_time = time.perf_counter()
        subprocess.run(cmd, check=True)
        elapsed_time = time.perf_counter() - start_time

        total_ds, zenodo_ds, figshare_ds = count_dataset_repository_sources(params.texts_dir)
        total_seconds = int(elapsed_time)
        hours, remainder = divmod(total_seconds, 3600)
        mins, secs = divmod(remainder, 60)
        log_content = (
            f"=== LLM Annotation Log ===\n"
            f"Model: {params.model_name} ({wildcards.model_name})\n"
            f"Device/Interface: Openrouter API\n"
            f"Framework: {params.framework}\n"
            f"Temperature: {params.temperature}\n"
            f"Total Datasets Processed: {total_ds}\n"
            f"  - Zenodo: {zenodo_ds}\n"
            f"  - Figshare: {figshare_ds}\n"
            f"Elapsed Time: {elapsed_time:.2f} seconds ({hours}h{mins}m{secs:02d}s)\n"
            f"Average Speed: {elapsed_time / max(total_ds, 1):.2f} seconds/sample\n"
        )
        Path(log[0]).write_text(log_content, encoding="utf-8")


# Generate standalone HTML visualization for GLiNER annotations.
rule render_gliner_annotations_html:
    input:
        flag="results/mdverse/gliner/{model_name}_{device}/.done",
    output:
        html="results/mdverse/gliner/{model_name}_{device}/annotations.html",
    run:
        export_annotations_to_html(
            annotation_dir=f"results/mdverse/gliner/{wildcards.model_name}_{wildcards.device}",
            output_html_path=output.html,
        )


# Generate standalone HTML visualization for LLM annotations.
rule render_llm_annotations_html:
    input:
        flag="results/mdverse/llm/{model_name}_api/.done",
    output:
        html="results/mdverse/llm/{model_name}_api/annotations.html",
    run:
        export_annotations_to_html(
            annotation_dir=f"results/mdverse/llm/{wildcards.model_name}_api",
            output_html_path=output.html,
        )