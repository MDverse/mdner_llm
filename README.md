# LLM Prompting for Molecular Dynamics Named Entity Recognition (MDNER)

## Introduction

This project explores methods for automatically annotating dataset descriptions and scientific texts related to Molecular Dynamics (MD).

## Annotation dataset

A dataset of about 374 annotated texts is available in the `annotations` folder.
These texts are build from the title and description of molecular dynamics simulation datasets scraped from Zenodo and Figshare.

These texts have been manually annotated with [annotation rules](docs/annotation_rules.md).

## Using large language models

Because Large Language Models (LLMs) are inherently non-deterministic, we aim to enforce structured and reproducible outputs using a strict [Pydantic](https://docs.pydantic.dev/1.10/) schema. Below is a Mermaid diagram that summarizes the schema used to capture detected entities:

```mermaid
classDiagram
    class ListOfEntities {
        entities: list[Molecule | SimulationTime | ForceFieldModel | SimulationTemperature | SoftwareName | SoftwareVersion]
    }

    class SoftwareVersion {
        category: str = 'SOFTVERS'
        text: str
    }

    class SimulationTemperature {
        category: str = 'STEMP'
        text: str
    }

    class SimulationTime {
        category: str = 'STIME'
        text: str
    }

    class Molecule {
        category: str = 'MOL'
        text: str
    }

    class SoftwareName {
        category: str = 'SOFTNAME'
        text: str
    }

    class ForceFieldModel {
        category: str = 'FFM'
        text: str
    }

    class Entity {
        category: str
        text: str
    }

    ListOfEntities ..> Molecule
    ListOfEntities ..> SoftwareVersion
    ListOfEntities ..> SimulationTime
    ListOfEntities ..> SimulationTemperature
    ListOfEntities ..> SoftwareName
    ListOfEntities ..> ForceFieldModel

To assess robustness and accuracy, we benchmark several LLMs (GPT-5.6, Gemini 3.6, Claude Fable 5, GLM 5.3, etc.) together with extraction libraries such as **Instructor** and **Pydantic**. Our goal is to identify the best model–framework combinations for accurate, consistent, and schema-compliant Molecular Dynamics Named Entity Recognition (MDNER).

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

## Add OpenRouter API key

Create an .env file with a valid [OpenRouter](https://openrouter.ai/docs/api/reference/authentication) API key:

```sh
OPENROUTER_API_KEY=<your-openrouter-api-key>
```

> Remark: This .env file is ignored by git.

## Perform quality control and explore annotation dataset

Perform quality control on manually annotated entities:

```sh
$ uv run validate-annotations --inferences-dir data/groundtruth
2026-07-08 17:05:29 | INFO     | Validating all annotations in directory: data/groundtruth.
2026-07-08 17:05:29 | INFO     | Found 160 JSON files to validate.
2026-07-08 17:05:29 | INFO     | Total text mismatches: 0
2026-07-08 17:05:29 | INFO     | Total span mismatches: 0
2026-07-08 17:05:29 | INFO     | Total overlapping entities: 0
2026-07-08 17:05:29 | INFO     | Total removed entities: 0
2026-07-08 17:05:29 | INFO     | Total entities with invalid boundaries: 0
2026-07-08 17:05:29 | INFO     | Total unknown categories: 0
2026-07-08 17:05:29 | SUCCESS  | Validation completed successfully!
```

Make the inventory of all entities:

```sh
$ uv run build-entity-inventory --annotations-path data/groundtruth --out-path data/entities.tsv
2026-07-08 17:05:51 | INFO     | Starting entity inventory.
2026-07-08 17:05:51 | INFO     | Collecting entities.
2026-07-08 17:05:51 | SUCCESS  | Found 160 JSON files successfully.
2026-07-08 17:05:51 | SUCCESS  | Collected 2519 entities.
2026-07-08 17:05:51 | INFO     | Writing entity inventory TSV file.
2026-07-08 17:05:51 | SUCCESS  | Saved entity inventory in: data/entities.tsv
2026-07-08 17:05:51 | SUCCESS  | Saved entity distribution plot in 'plots/annotations/entity_distribution.png'.
2026-07-08 17:05:51 | SUCCESS  | Saved entity distribution by category plot in 'plots/annotations/entity_distribution_by_category.png'.
2026-07-08 17:05:51 | SUCCESS  | Saved unique entity distribution by category plot in 'plots/annotations/unique_entity_distribution_by_category.png'.
2026-07-08 17:05:52 | SUCCESS  | Saved categories per text distribution plot in 'plots/annotations/categories_per_text_distribution.png'.
2026-07-08 17:05:52 | SUCCESS  | Saved text length distribution plot in 'plots/annotations/text_length_distribution.png'.
2026-07-08 17:05:52 | SUCCESS  | Entity inventory completed successfully!
```

A list of entities per category can be found in [notebooks/explore_entities_from_inventory.ipynb](notebooks/explore_entities_from_inventory.ipynb).

## Usage

### Extract entities of one text 📃

To extract structured entities from a single text using a specified LLM ([from OpenRouter available models](https://openrouter.ai/models)) and framework, run :

```sh
uv run extract-entities-with-llm \
    --text-path data/groundtruth/figshare_121241.json \
    --model google/gemma-4-31b-it \
    --framework instructor \
    --temperature 1.0 \
    --prompt-path docs/prompt_template.md \
    --guidelines-path docs/annotation_rules.md \
    --examples-path docs/few_shot_examples.md \
    --output-dir results/llm/inferences
2026-04-22 00:12:22 | INFO     | Starting the extraction of entities.
2026-04-22 00:12:22 | DEBUG    | Loading text and metadata from data/groundtruth/figshare_121241.json.
2026-04-22 00:12:22 | DEBUG    | Loaded text (1710 chars): Modeling of Arylamide Helix Mimetics in the p53 Peptide Binding Site...
2026-04-22 00:12:22 | DEBUG    | Loading prompt from docs/prompt_template.md.
2026-04-22 00:12:22 | DEBUG    | Loaded prompt (6685 chars) : # Named-Entity Recognition task  ## Role definition  You are a highly speci...
2026-04-22 00:12:22 | DEBUG    | Starting annotation with model google/gemma-4-31b-it using instructor.
2026-04-22 00:12:25 | DEBUG    | Response status: ok.
2026-04-22 00:12:25 | DEBUG    | Provider used: Venice.
2026-04-22 00:12:25 | DEBUG    | Formatted LLM response: 
                                 entities=[ForceField(category='FFM', text='CHARMM36'), Molecule(category='MOL', text='POPC'), SimulationTemperature(category='STEMP', text='310K'), SoftwareName(category='SOFTNAME', text='GROMACS'), SoftwareVersion(category='SOFTVERS', text='5.1.4')]
2026-04-22 00:12:25 | DEBUG    | Inference time: 2.6673661249951692 seconds.
2026-04-22 00:12:25 | DEBUG    | Input tokens: 3236.
2026-04-22 00:12:25 | DEBUG    | Output tokens: 70.
2026-04-22 00:12:25 | DEBUG    | Cost usage: 0.00623 $.
2026-04-22 00:12:25 | DEBUG    | Saved raw response successfully.
2026-04-22 00:12:25 | DEBUG    | Saved formated response with metadata successfully.
2026-04-22 00:12:25 | SUCCESS  | Completed the extraction of entities successfully!
```

> This command generates two outputs: a `.txt` file containing the raw LLM response, and a `.json` file containing the extracted entities along with metadata about the extraction (model, framework, input file, and run details).

```
# Example
# Input text:
# "Simulation data for CHARMM36 POPC bilayer, 100 lipids/leaflet, 310K, GROMACS 5.1.4."

# Output:
{
  "entities": [
    {
      "category": "FFM",
      "text": "CHARMM36"
    },
    {
      "category": "MOL",
      "text": "POPC"
    },
    {
      "category": "STEMP",
      "text": "310K"
    },
    {
      "category": "SOFTNAME",
      "text": "GROMACS"
    },
    {
      "category": "SOFTVERS",
      "text": "5.1.4"
    }
  ]
}
```

### Extract entities for multiple texts 📑

To extract structured entities from multiple dataset descriptions, execute:

```sh
uv run extract-entities-with-llm-all-texts \
    --texts-path data/groundtruth \
    --model google/gemma-4-31b-it \
    --framework instructor \
    --temperature 1.0 \
    --prompt-path docs/prompt_template.md \
    --guidelines-path docs/annotation_rules.md \
    --examples-path docs/few_shot_examples.md \
    --output-dir results/llm/inferences
```

### Normalize extracted entities across multiple annotations 🧹

To normalize extracted entities across multiple annotations, run:

```sh
uv run normalize-extracted-entities \
    --inferences-dir results/llm/inferences \
    --ffm-db-path data/normalization/md_forcefields_registry.json \
    --softname-db-path data/normalization/software_names_registry.json \
    --model-name "deepseek/deepseek-v4-pro" \
    --output-dir results/llm/inferences_normalized
```

> This command normalizes metadata structure across extracted categories, assigns a confidence score, and flags hallucinated mentions that do not match the original text footprint. Refer to the [Normalization Guide](docs/normalization_rules.md) for full mapping rules.

```
# Example
# Input:
{
  "entities": [
    {
      "category": "FFM",
      "text": "CHARMM36"
    },
    {
      "category": "MOL",
      "text": "POPC"
    },
    {
      "category": "STEMP",
      "text": "310K"
    },
    {
      "category": "SOFTNAME",
      "text": "GROMACS"
    },
    {
      "category": "SOFTVERS",
      "text": "5.1.4"
    },
    {
      "category": "STIME",
      "text": "100 ns"
    },
  ]
}
# Output:
{
   "normalized_entities": [
      {
        "category": "FFM",
        "text_normalized": "charmm36",
        "score": 0.95,
        "is_hallucinated": false
      },
      {
        "category": "MOL",
        "text_normalized": "popc",
        "score": 0.95,
        "is_hallucinated": false,
      },
      {
        "category": "STEMP",
        "text_normalized": "310k",
        "score": 0.85,
        "is_hallucinated": false,
        "value": 310.0,
        "unit": "K"
      },
      {
        "category": "SOFTNAME",
        "text_normalized": "gromacs",
        "score": 0.95,
        "is_hallucinated": false
      },
      {
        "category": "SOFTVERS",
        "text_normalized": "5.1.4",
        "score": 0.90,
        "is_hallucinated": false
      },
      {
        "category": "MOL",
        "text_normalized": "cholesterol",
        "score": 0.30,
        "is_hallucinated": true,
      }
    ]
  }
```


### Aggregate consensus entities across multiple annotations 📦

To aggregate consensus entities across multiple annotations, run:

```sh
uv run aggregate-consensus-entities \
    --inferences-dir results/llm/inferences \
    --threshold 0.5 \
    --output-dir results/llm/inferences_consensus
```

> This command loads all LLM JSON file predictions in `results/llm/inferences`, computes per-entity consensus scores across all annotations, and saves the consensus entities with scores above the specified threshold in `results/llm/inferences_consensus`.

### Fine-tune Gliner2 on Molecular Dynamics annotations 🚀

To train the Gliner2 model on the Molecular Dynamics annotations, run:

```sh
uv run train-gliner --config-path workflow/configs/gliner_training.yaml
```

> This command trains the Gliner2 model using the configuration specified in [workflow/configs/gliner_training.yaml](workflow/configs/gliner_training.yaml) and save the trained model with the best validation performance.

Then, to extract entities from new texts using the fine-tuned Gliner2 model, run:

```sh
uv run extract-entities-with-gliner-all-texts \
        --model-path <path_to_best_finetuned_model> \
        --text-path <path_to_test_jsonl_file> \
        --metadata-path <path_to_test_metadata_txt_file> \
        --output-dir results/gliner/inferences
```

### Evaluate extraction performance ⚖️

To evaluate the quality of annotations produced by both LLM and Gliner2 models, run:

```sh
uv run evaluate-entities-extraction \
        --inferences-dir results/llm/inferences \   # or results/gliner/inferences
        --results-dir results/llm/evaluation        # or results/gliner/evaluation
```

> This command loads all LLM-generated JSON files in `results/llm/inferences`, computes per-annotation metrics against the ground-truth, and saves the results in `results/llm/evaluation`. It generates an csv file with overall metrics for each entity class, and a parquet file with detailed annotation results for each test sample and each category.

A comparison of the performance of different LLMs/Gliner2 models and frameworks can be found in [notebooks/compare_models_performance.ipynb](notebooks/compare_models_performance.ipynb).


## Workflow Orchestration with Snakemake 🚀

We provide end-to-end reproducible pipelines orchestrated with [Snakemake](https://snakemake.readthedocs.io/).

```sh
uv run snakemake gliner_all --cores all --resources gpu=1     # only gliner
uv run snakemake llm_all --cores all --resources api_calls=1  # only llm
uv run snakemake all --cores all --resources gpu=1            # all
```

All evaluation metrics, comparison plots, and performance charts across models and architectures can be analyzed and plotted using [notebooks/plot_ner_performance.ipynb](notebooks/plot_ner_performance.ipynb).
