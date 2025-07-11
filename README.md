# LLM Prompting for Molecular Dynamics Named Entity Recognition (MDNER)

## Setup environment

We use [uv](https://docs.astral.sh/uv/getting-started/installation/)
to manage dependencies and the project environment.

Clone the GitHub repository:

```sh
git clone git@github.com:MDverse/mdner_llm.git
cd md_data_schema
```

Sync dependencies:

```sh
uv sync
```

## Run the LLM Prompting

To run LLM prompting, ensure you have an API key from **OpenAI** or **Groq** (or both). These should be saved in a `.env` file located in the root of the repository.

- **Note**: OpenAI requires a paid API key, while Groq currently offers free access.

### Configurable Parameters

You can customize several parameters before running the script:

- **Number of texts to annotate** (selected from the `annotations/` folder)
- **Prompt type**: choose from `zero-shot`, `one-shot`, or `few-shot`
- **LLM models** available via your API keys

These parameters can be modified at the beginning of the script:  
`src/run_llm_annotation.py`

### Running the Script

To execute the LLM prompting process, use the following command:

```sh
uv run src/run_llm_annotation.py
```

### Output

The results are saved in the `llm_outputs/` directory. Each run generates a subfolder named with the `<datetime>` of execution. This folder contains the following folders and files:

- `annotations/` — the complete set of model responses for all annotations
- `images/` — folder that will be used to store plots related to this annotation run of prompting
- `stats/` — any other resulting file (`.csv`, `.json`, etc.) that can be generated from the processsing of the annotations.
- `run_llm_annotation.log` — a log file capturing all parameters and settings used during the run

> This output structure supports reproducibility and easy comparison across runs.


## Run Quality Control (QC) checks on a prompting result

Quality Control (QC) refers to post-prompting analysis aimed at evaluating the reliability of model-generated annotations.  
In this step, we check for:

- **Hallucinated entities**
- **False positives** for each entity label

This process helps identify and quantify potential LLM hallucinations and label errors.

### Running the Script

To run the QC analysis, use the following command:

```sh
uv run src/run_qc_analysis.py
```

### Output

The script generates the following output file:

- `<datetime>/stats/quality_control_results.csv` — a CSV report containing:
  - Entity-level false positive counts
  - Detected hallucinated entities

> This output helps diagnose quality issues in model responses and supports further refinement of prompts or model settings.


## Score the prompting results

You are going to need to have run the control quality before scoring the responses.

The aim of this script is to identify:
- false positives
- false negatives
for each annotations.

To run the script use this command:

```sh
uv run src/run_scoring_analysis.py
```

### Output

The results will be saved as:

`<datetime>/stats/scoring_results.csv`


## Generate plots

> ⚠️ **Prerequisite**: You must run the quality control and/or the scoring analysis in order to generate plots.

Some plots require only QC results, while others may depend on the scoring data. You can configure which plots to generate by editing the script: `src/generate_plots.py`

This plotting does not take temperatures into consideration. Simply prompting types and the different models used. 

To run the plot generation, use the following command:
```sh
uv run src/generate_plots.py
```

### Output

Any plot generated will be saved as:

`<datetime>/images/<image_title>.png`

## Run a consensus confidence scoring and plot the results

Before starting the confidence scoring, make sure to have some conditions met and that a few parameters are correctly set.

To run consensus confidence scoring, you **must have previously prompted at least one model with at least one temperature**.  
When running the `run_llm_annotation.py` script, ensure the following:

- You answer `'yes'` to the question: **"Do you want to use consensus scoring?"**
- You verify and/or modify the temperatures that match your needs

### Configuration

Once LLM responses are available, you can proceed with confidence scoring.  
Before running the script, review the parameters at the top of `src/confidence_scoring.py`. You can modify:

- **Temperatures** used for the LLM generation
- **Prompt type** to run confidence scoring on (`zero-shot`, `one-shot`, or `few-shot`)
- **Model name** to apply consensus scoring on

> ⚠️ Note: The consensus confidence scoring is designed to be run on **one overarching model** and **one prompt type** at a time.

Ensure that these parameters align with those used during LLM prompting.

### Running the Script

To run the consensus scoring, use the following command:

```sh
uv run src/confidence_scoring.py
```

### Output

The script produces the following output files:

#### `/<datetime>/images/`
- `consensus_score_<filename_scored>.png` — a visual plot of all annotations alongside their consensus confidence scores

#### `<datetime>/stats/`
- `consensus_score_<filename_scored>_<model>_<prompt>.json` — a JSON file containing all annotations with their computed confidence scores and associated metadata
