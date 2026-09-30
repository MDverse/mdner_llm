# Rule file for MDverse extraction and entity annotation pipelines.

import json
from pathlib import Path
import pandas as pd

from mdner_llm.visualization.annotations import export_annotations_to_html

# Load pipeline configuration file.
configfile: "workflow/configs/mdverse_annotation.yaml"

RAW_ZENODO = config["raw_datasets"]["zenodo"]
RAW_FIGSHARE = config["raw_datasets"]["figshare"]
MDVERSE_DIR = Path(config.get("mdverse_dir"))
SELECTED_ENGINES = config.get("engines")


def get_mdverse_annotation_targets(wildcards):
    """Resolve expected target files based on configured annotation engines."""
    targets = []
    if "gliner" in SELECTED_ENGINES:
        targets.append("results/mdverse/gliner_annotations.html")
    if "llm" in SELECTED_ENGINES:
        targets.append("results/mdverse/llm_annotations.html")
    return targets


# Top-level coordinator rule for MDverse annotation pipeline.
rule run_mdverse_annotation:
    input:
        get_mdverse_annotation_targets,


# Generate standalone HTML visualization for GLiNER annotations.
rule render_gliner_annotations_html:
    input:
        flag="results/mdverse/gliner/.done",
    output:
        html="results/mdverse/gliner_annotations.html",
    run:
        export_annotations_to_html(
            annotation_dir="results/mdverse/gliner",
            output_html_path=output.html,
        )


# Generate standalone HTML visualization for LLM annotations.
rule render_llm_annotations_html:
    input:
        flag="results/mdverse/llm/.done",
    output:
        html="results/mdverse/llm_annotations.html",
    run:
        export_annotations_to_html(
            annotation_dir="results/mdverse/llm",
            output_html_path=output.html,
        )

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
        # Load and concatenate repository datasets.
        zenodo_df = pd.read_parquet(input.zenodo)
        figshare_df = pd.read_parquet(input.figshare)
        combined_df = pd.concat([zenodo_df, figshare_df], ignore_index=True)
        # Generate one JSON file per dataset entry.
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


# Annotate MDverse JSON files using GLiNER model.
rule annotate_mdverse_with_gliner:
    input:
        flag=f"{MDVERSE_DIR}/.done",
    output:
        flag=touch("results/mdverse/gliner/.done"),
    params:
        texts_dir=str(MDVERSE_DIR),
        model_path=config["models"]["gliner"]["model_path"],
        out_dir="results/mdverse/gliner",
    shell:
        """
        uv run extract-entities-with-gliner-all-texts \
            --texts-path {params.texts_dir} \
            --model-path {params.model_path} \
            --output-dir {params.out_dir}
        """


# Annotate MDverse JSON files using configured LLM.
rule annotate_mdverse_with_llm:
    input:
        flag=f"{MDVERSE_DIR}/.done",
    output:
        flag=touch("results/mdverse/llm/.done"),
    params:
        texts_dir=str(MDVERSE_DIR),
        model_name=config["models"]["llm"]["model_name"],
        framework=config["models"]["llm"]["framework"],
        temperature=config["models"]["llm"]["temperature"],
        out_dir="results/mdverse/llm",
    shell:
        """
        uv run extract-entities-with-llm-all-texts \
            --texts-path {params.texts_dir} \
            --model-name {params.model_name} \
            --framework-name {params.framework} \
            --temperature {params.temperature} \
            --output-dir {params.out_dir}
        """