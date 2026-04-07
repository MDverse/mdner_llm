# LLM Prompting for Molecular Dynamics Named Entity Recognition (MDNER)

## Introduction

This project explores methods for automatically annotating dataset descriptions and scientific texts related to Molecular Dynamics (MD).

## Annotation dataset

A dataset of about 280 annotated texts is available in the `annotations` folder.
These texts are the title and description of molecular dynamics simulation datasets scraped from Zenodo and Figshare.

These texts have been annotated with [annotation rules](docs/annotation_rules.md).

## Using large language models

Because Large Language Models (LLMs) are inherently non-deterministic, we aim to enforce structured and reproducible outputs using a strict [Pydantic](https://docs.pydantic.dev/1.10/) schema. Below is a Mermaid diagram that summarizes the schema used to capture detected entities:

```mermaid
classDiagram
    class ListOfEntities {
        entities: list[Molecule | SimulationTime | ForceFieldModel | Temperature | SoftwareName | SoftwareVersion]
    }

    class SoftwareVersion {
        label: str = 'SOFTVERS'
        text: str
    }

    class Temperature {
        label: str = 'TEMP'
        text: str
    }

    class SimulationTime {
        label: str = 'STIME'
        text: str
    }

    class Molecule {
        label: str = 'MOL'
        text: str
    }

    class SoftwareName {
        label: str = 'SOFTNAME'
        text: str
    }

    class ForceFieldModel {
        label: str = 'FFM'
        text: str
    }

    class Entity {
        label: str
        text: str
    }

    ListOfEntities ..> Molecule
    ListOfEntities ..> SoftwareVersion
    ListOfEntities ..> SimulationTime
    ListOfEntities ..> Temperature
    ListOfEntities ..> SoftwareName
    ListOfEntities ..> ForceFieldModel
```

To assess robustness and accuracy, we benchmark several LLMs (GPT-5, Gemini 3 Pro, etc.) together with extraction libraries such as **Instructor**, **LlamaIndex**, and **Pydantic**. Our goal is to identify the best model–framework combinations for accurate, consistent, and schema-compliant Molecular Dynamics Named Entity Recognition (MDNER).

## Setup environment

We use [uv](https://docs.astral.sh/uv/getting-started/installation/)
to manage dependencies and the project environment.

Clone the GitHub repository:

```sh
git clone git@github.com:MDverse/mdner_llm.git
cd mdner_llm
```

Sync dependencies:

```sh
uv sync
```

## Add OpenAI and OpenRouter API key

Create an .env file with a valid [OpenAI](https://platform.openai.com/docs/api-reference/authentication) and [OpenRouter](https://openrouter.ai/docs/api/reference/authentication) API key:

```sh
OPENAI_API_KEY=<your-openai-api-key>
OPENROUTER_API_KEY=<your-openrouter-api-key>
```

> Remark: This .env file is ignored by git.

## Usage

### Extract entities of one text 📃

To extract structured entities from a single text using a specified LLM ([from OpenRouter available models]((https://openrouter.ai/models))) and framework, run :

```sh
uv run extract-entities \
    --text-path annotations/v3/figshare_2018634.json \
    --model openai/gpt-5.2 \
    --framework instructor
```
> This command generates two outputs: a `.txt` file containing the raw LLM response, and a `.json` file containing the extracted entities along with metadata about the extraction (model, framework, input file, and run details).


```
# TXT output example:
{
  "entities": [
    {
      "label": "MOL",
      "text": "Phosphatidylcholine"
    },
    {
      "label": "MOL",
      "text": "1,2-diauroyl-sn-glycero-3-phospocholine"
    },
    {
      "label": "MOL",
      "text": "DLPC"
    },
    {
      "label": "MOL",
      "text": "DMPC"
    },
    {
      "label": "MOL",
      "text": "DPPC"
    },
    {
      "label": "FFM",
      "text": "AMBER"
    }
  ]
}
```
```json
# JSON output example:
{
    "timestamp": "2026-03-09T21:06:24",
    "output_file": "results/llm_annotations/figshare_2018634_openai_gpt-5.2_instructor.txt",
    "text_file": "annotations/v3/figshare_2018634.json",
    "framework_name": "instructor",
    "model_name": "openai/gpt-5.2",
    "prompt_path": "json_few_shot.txt",
    "tag_prompt": "json",
    "inference_time_sec": 14.71316909790039,
    "raw_llm_response": "<raw LLM response string>",
    "groundtruth": "<ground-truth entities string>",
}
```

### Extract entities for multiple texts 📑

To extract structured entities from multiple dataset descriptions listed in [results/100_selected_md_dataset_description_paths.txt](results/100_selected_md_dataset_description_paths.txt), execute:

```sh
uv run extract-entities-all-texts \
    --texts-path results/100_selected_md_dataset_description_paths.txt \
    --model openai/gpt-5.2 \
    --framework instructor
```

### Evaluate LLM annotations ⚖️

To evaluate the quality of annotations produced by LLMs and different framework, run:

```sh
uv run evaluate-llm-annotations \
        --annotations-dir results/llm_annotations \
        --results-dir results/annotation_evaluation_stats
```

> This command loads all LLM-generated JSON files in results/llm_annotations, computes per-annotation metrics against the ground-truth, and saves the results in `results/annotation_evaluation_stats/per_text_metrics_YYYY-MM-DDTHH-MM-SS.parquet`. It then creates an Excel summary for each model and framework in `results/annotation_evaluation_stats/evaluation_summary_YYYY-MM-DDTHH-MM-SS.xlsx`.

## Train Gliner2 model on Molecular Dynamics annotations

To train the Gliner2 model on the Molecular Dynamics annotations, run:

```sh
uv run train-gliner --config-path src/mdner_llm/gliner/training_config.yaml
```
> This command trains the Gliner2 model using the configuration specified in [src/mdner_llm/gliner/training_config.yaml](src/mdner_llm/gliner/training_config.yaml) and save the trained model with the best validation performance.


## Evaluate Gliner2 models

To evaluate the performance of the trained Gliner2 model or on a test set of annotations, run:

```sh
uv run evaluate-gliner \
    --model-name gliner2_base_large \
    --model-path fastino/gliner2-large-v1 \
    --test-dataset data/gliner/test.jsonl \
    --test-metadata-path data/gliner/test_metadata.txt
```

> This command evaluates the specified Gliner2 model on the test dataset provided in `data/gliner/test.jsonl`. It computes evaluation metrics such as precision, recall, and F1-score for each entity class. It saves the annotation results (per test sample and per labels) into a parquet file, and generates a summary Excel file with overall metrics for each entity class.


## Utilities

### 1. Correct JSON annotations

To vizualize and correct json annotations, open the notebook in `notebooks/review/correct_annotations.ipynb`.

### 2. Count entities per class for each annotation

To perform statistics on the distribution of annotations per files and class, run:

```sh
uv run count-entities --annotations-dir annotations/v3
```

> This command processes all JSON files listed, counts the number of entities per class for each annotation, and outputs a TSV file with the filename, text length, and entity counts per class. It will also produce plots with class distribution for all entities and entity distribution by class.

### 3. Quality Control Inventory of Named Entities

To generate a QC inventory of annotated entities from json files, run:

```sh
uv run build-entity-vocab \
    --texts-path results/groundtruth_paths.txt \
    --out-path results/qc_annotations/entities.tsv
```

> This command will scan all JSON annotations listed in `results/groundtruth_paths.txt`, extract all annotated entities, and compile them into a TSV file at `results/qc_annotations/entities.tsv`. The TSV file will contain columns for the entity text, label, source file, and frequency of occurrence across all annotations.

> 💡 Running a QC inventory on annotation files ensures that all entities are
> consistently aggregated and normalized. This is a crucial step for
> defining **annotation rules in molecular dynamics**, helping standardize
> formats, units, and naming conventions. The generated files can be
> explored in [`notebooks/review/qc_entity_inventory_explorer.ipynb`](notebooks/review/qc_entity_inventory_explorer.ipynb)
> and the rules are documented in [`docs/annotation_rules.md`](docs/annotation_rules.md).

### 4. Select informative annotation JSON files

To select informative annotation JSON files and export their paths in a text file, run:

```sh
uv run select-annotations \
        --annotations-dir annotations/v3 \
        --nb-files 50
```

> This command selects up to 50 annotation JSON files from `annotations/v3` according to entity coverage and recency, and writes their paths to a `.txt` file in the `results` folder. The selection is based on a scoring system that prioritizes files with more entities and more recent modification dates.

### 5. Extract description from parquet files

To extract dataset descriptions from parquet files and save them as `.txt` files, run:

```sh
uv run extract-description \
    --input-dir data/parquets \
    --output-dir annotations/v3
```

> This command reads all parquet files in `data/parquets` and creates one text file per row. For each row, it extracts the title and description from each file, and saves them as `<repository_name>_<dataset_id>.txt`  in the `annotations/v3` directory.

### 6. Transfer annotations to new JSON files

To transfer annotations from old JSON files to new ones, run:

```sh
uv run transfer-annotations \
    --old-annotations-dir annotations/v2 \
    --new-annotations-dir annotations/v3
```

> This command transfers annotations from JSON files in `annotations/v2` to corresponding files in `annotations/v3` based on filename matching. It reads each old annotation file, extracts the entities, and updates the new annotation file with these entities while preserving the original text and metadata. The updated annotations are saved back to the new annotation directory.


### 7. Validate annotations

To validate the annotations in a JSON file or a directory of JSON files, run:

#### Validate a single JSON file
```sh
uv run validate-annotations --json-path annotations/v3/figshare_121241.json
```

#### Validate all JSON files in a directory
```sh
uv run validate-annotations --annotations-dir annotations/v3
```

> This command checks the validity of annotations in the specified JSON file or all JSON files in the given directory. It performs several checks, including verifying that the annotated text matches the corresponding text span in the original text, ensuring that entity spans are valid and do not overlap, and removing unwanted entities based on a predefined list.


### 8. Plot performance metrics

To plot the evaluation metrics of a single gliner model, run:

```sh
uv run plot-metrics \
    --input-file <path-to-xlsx-file>
    --model-name <model-name>
```
> This command reads the evaluation metrics from the specified Excel file, generates and saves a plot of precision, recall, and F1-score for each entity class.