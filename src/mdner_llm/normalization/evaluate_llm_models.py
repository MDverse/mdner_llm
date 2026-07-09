"""Script to normalize the simlation times into standard units."""

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import click
import instructor
from dotenv import load_dotenv
from instructor.core.exceptions import ValidationError
from instructor.exceptions import InstructorRetryException
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# list of model that we are going to test
MODELS = [
    "openai/gpt-4o",
    "openai/gpt-5.5",
    "deepseek/deepseek-v4-pro",
    "google/gemma-4-31b-it",
    "qwen/qwen3.6-27b",
    "minimax/minimax-m2.7",
    "moonshotai/kimi-k2.6",
    "anthropic/claude-opus-4.7",
    "mistralai/mistral-large-2512",
]


class SimulationTime(BaseModel):
    """Define the structure of simulation time entity."""

    value: float | None = Field(
        ..., description="Normalized value ofthe simulation time"
    )
    unit: Literal["ps", "ns", "μs", "ms", "s"] | None = Field(
        ..., max_length=2, description="Normalized unit of the simulation time"
    )


class NormSimuTime(BaseModel):
    """Define the structure for the output of the LLM."""

    input: str = Field(..., description="raw value of one simulaton time")
    output: list[SimulationTime] = Field(
        ..., description="normalized simulation timevalues and units"
    )


# We load the simulation times from the file in a list of simulatuion time to enable
# slicing the list
def load_simulation_times(ground_truth_file: Path) -> list:
    """Load simulation times from a file into a list.

    Parameters
    ----------
    ground_truth_file (Path): Path to the input file containing the simulation times

    Returns
    -------
    list: A list of simulation times loaded from the file
    """
    logger.info(f"Loading the simulation times from {ground_truth_file}...")
    times = []
    with open(ground_truth_file) as gt_file:
        ground_truth = json.load(gt_file)
        for value in ground_truth["groundtruth"]:
            times.append(value["input"])
    logger.success(f"Loaded {len(times)} simulation times successfully.")
    return times


# We load the prompt text from the file
def load_prompt_text(prompt_file_path):
    """Extract the prompt from the txt prompt file.

    Parameters
    ----------
        prompt_file_path(Path): Path to the file containing the prompt

    Returns
    -------
            (str): content of the prompt file
    """
    content = ""
    content = Path(prompt_file_path).read_text()
    return content


# We give to a chosen model a normalisation time, the model is called via an
# openrouter key and use instructor to ensure the structured output of the llm.
# We retrieve the time and the cost of the normalisation
def normalize_simulation_time(
    raw_simulation_time: str, model_name: str, prompt_file_path: Path
):
    """Normalize the units in the simulation time text to standard units.

    Parameters
    ----------
        simulation_time_filepath: Path to the input text file containing simulation time
        values.

    Returns
    -------
        A string containing the normalized simulation time values in JSON format.
    """
    client = instructor.from_openai(
        OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPEN_ROUTER_KEY"),
        )
    )

    logger.info(f"{model_name.replace('-', '_')} | Normalizing: {raw_simulation_time}")
    prompt = load_prompt_text(prompt_file_path)

    try:
        start_time = time.perf_counter()
        completion_pydantic, completion_basic = (
            client.chat.completions.create_with_completion(
                model=model_name,
                max_retries=3,
                response_model=NormSimuTime,
                messages=[
                    {
                        "role": "system",
                        "content": f"{prompt}",
                    },
                    {
                        "role": "user",
                        "content": f"{raw_simulation_time}",
                    },
                ],
            )
        )
    # total_cost = completion_basic.usage.cost_details["upstream_inference_cost"]
    except InstructorRetryException as exc:
        logger.error(f"Normalisation failed after {exc.n_attempts} attempts")
        elapsed_time = time.perf_counter() - start_time
        return None, elapsed_time, None

    except ValidationError as exc:
        logger.error(f"Pydantic validation failed:  {exc}")
        elapsed_time = time.perf_counter() - start_time
        return None, elapsed_time, None

    elapsed_time = time.perf_counter() - start_time
    logger.info(
        f"{model_name.replace('-', '_')} | output: {completion_pydantic.output}"
    )
    cost = completion_basic.usage.cost_details["upstream_inference_cost"]
    return completion_pydantic.model_dump_json(), elapsed_time, cost


# We format the output of the normalized simulation time in a json format.
def format_norm_simulation_time(
    raw_simulation_time: list, model_name: str, prompt_file_path: Path
) -> dict:
    """Format the normalized time to a JSON format with the normalized values.

    Parameters
    ----------
    raw_simulation_time (Path) : Path to the input file containing
                                the raw simulation times
    normalized_simulation_time (Path) : Path to the output file containing
                                the normalized simulation times

    Returns
    -------
    dict[list] : dictonarry that contains the results of the simulation times
    normalisation
    """
    all_simulation_times_norm = []
    normalisation_output = {}
    # We loop on the list of simulation times and normalize each entity
    for simulation_time in raw_simulation_time:
        normalization_result = normalize_simulation_time(
            simulation_time, model_name, prompt_file_path
        )
        if normalization_result:
            # We put the normalized simulation times in a json format
            simulation_time_normalized = json.loads(normalization_result[0])
            all_simulation_times_norm.append(simulation_time_normalized)
        else:
            simulation_time_not_normalized = {
                "input": simulation_time,
                "output": normalization_result,
            }
            all_simulation_times_norm.append(simulation_time_not_normalized)
    normalisation_output["normalisation_output"] = all_simulation_times_norm
    return normalisation_output


def save_norm_simulation_results(
    normalisation_output: dict, normalized_simulation_time: Path
):
    """Generate a JSON file with the results of the simulation times normalisation."""
    logger.info("Saving the normalisation results in the JSON file")
    with open(normalized_simulation_time, "w") as file_1:
        json.dump(normalisation_output, file_1, indent=4, ensure_ascii=False)
    # logger.success("Saving results to JSON file successful")


def normalize_all_entities(
    raw_simulation_times: list,
    model: str,
    ground_truth_dict: dict,
    prompt_file_path: Path,
) -> tuple[int, int, int]:
    """
    Normalize all the simulation times.

    Parameters
    ----------
    raw_simulation_times (list): A list of raw simulation time strings to be normalized.
    model (str): The name of the model to be used for normalization.
    ground_truth_dict (dict): A dictionary mapping raw simulation time strings to their
    corresponding ground truth normalized values.

    Returns
    -------
    int: The number of correctly normalized simulation times compared to the ground
    truth.
    """
    normalised_entity = 0
    normalisation_time = 0
    normalisation_cost = 0
    entity_number = 1

    for raw_simulation_time in raw_simulation_times:
        logger.info("-" * 100)
        logger.info(
            f"{model.replace('-', '_')} | entity: {entity_number} / {len(raw_simulation_times)}"
        )
        normalisation_result = normalize_simulation_time(
            raw_simulation_time,
            model_name=model,
            prompt_file_path=prompt_file_path,
        )
        if normalisation_result:
            if normalisation_result[0]:
                normalized_result_json = normalisation_result[0]
            if normalisation_result[1]:
                normalisation_time += normalisation_result[1]
            if normalisation_result[2]:
                normalisation_cost += normalisation_result[2]
            if normalisation_result[0]:
                normalized_data = json.loads(normalized_result_json)
                ground_truth = ground_truth_dict.get(raw_simulation_time)

                if ground_truth is None:
                    logger.warning(f"No ground truth found for: {raw_simulation_time}")
                    continue

                match = True

                if len(normalized_data["output"]) != len(ground_truth):
                    match = False
                else:
                    for i in range(len(normalized_data["output"])):
                        if (
                            normalized_data["output"][i]["value"]
                            != ground_truth[i]["value"]
                        ):
                            logger.error(
                                f"{model.replace('-', '_')} | Normalisation failed "
                            )
                            match = False
                            break
                        if (
                            normalized_data["output"][i]["unit"]
                            != ground_truth[i]["unit"]
                        ):
                            logger.error(
                                f"{model.replace('-', '_')} | Normalisation failed "
                            )
                            match = False
                            break

                if match:
                    logger.success(
                        f"{model.replace('-', '_')} | Normalisation successfull "
                    )
                    entity_number += 1
                    normalised_entity += 1
                else:
                    entity_number += 1

    # logger.info(f"entity: {normalised_entity} / {len(raw_simulation_times)}")
    return normalised_entity, normalisation_time, normalisation_cost


def evaluate_all_models(
    raw_simulation_times: list,
    ground_truth_file: Path,
    runs: int,
    prompt_file_path: Path,
):
    """Evaluate all models and save results to TSV file.

    Parameters
    ----------
    raw_simulation_times (list): A list of raw simulation time strings to be normalized.
    ground_truth_file (Path): Path to the ground truth JSON file containing the correct
    normalized values for the simulation times.
    runs (int): The number of runs to perform for each model to
    calculate average accuracy.

    Returns
    -------
    list[dict]: A list of dictionaries containing the model names and their
    corresponding accuracy percentages.
    """
    with open(ground_truth_file) as f:
        ground_truth_data = json.load(f)

    ground_truth_dict = {}
    for truth in ground_truth_data["groundtruth"]:
        ground_truth_dict[truth["input"]] = truth["output"]

    results = []

    for model in MODELS:
        logger.info("-" * 20)
        logger.info(f"Model: {model.replace('-', '_')}")
        total_correct = 0
        total_normalisation_time = 0
        total_normalisation_cost = 0

        for run in range(runs):
            logger.info("-" * 80)
            logger.info(f"{model.replace('-', '_')} | Run {run + 1}/{runs}")

            normalisation_results = normalize_all_entities(
                raw_simulation_times, model, ground_truth_dict, prompt_file_path
            )

            normalised_entity = normalisation_results[0]
            normalisation_time = normalisation_results[1]
            normalisation_cost = normalisation_results[2]
            run_accuracy = (normalised_entity / len(raw_simulation_times)) * 100

            logger.info(f"  Run accuracy: {run_accuracy:.1f}%")
            total_correct += normalised_entity
            total_normalisation_time += normalisation_time
            total_normalisation_cost += normalisation_cost

        accuracy = (total_correct / (len(raw_simulation_times) * runs)) * 100
        normalisation_time_by_entity = total_normalisation_time / (
            len(raw_simulation_times) * runs
        )

        results.append(
            {
                "model_name": model,
                "accuracy_percentage": round(accuracy),
                "inference_time_by_entity": round(normalisation_time_by_entity),
                "inference_cost_by_entity_USD": round(
                    total_normalisation_cost / (len(raw_simulation_times) * runs)
                ),
            }
        )

        logger.info(
            f"\n {model.replace('-', '_')} : Accuracy = {accuracy:.1f}% Time = {total_normalisation_time}"
            f" Cost = {total_normalisation_cost / (len(raw_simulation_times) * runs)}\n"
        )

    return results


def save_evaluation_results_in_tsv(
    model_evaluation_file: Path,
    raw_simulation_times: list,
    ground_truth_file: Path,
    prompt_file_path: Path,
    runs: int,
):
    """Save the evaluation results of all models in a TSV file.

    Parameters
    ----------
    model_evaluation_file (Path): Path to the TSV file for model evaluation results.
    raw_simulation_times (list): A list of raw simulation time strings to be normalized.
    ground_truth_file (Path): Path to the ground truth JSON file containing the correct
    normalized values for the simulation times.
    runs (int): The number of runs to perform for each model to calculate the
    average accuracy.
    """
    results = evaluate_all_models(
        raw_simulation_times, ground_truth_file, runs, prompt_file_path
    )
    with open(model_evaluation_file, "w") as f:
        f.write(
            "model_name\taccuracy_percentage\tnormalisation_times_sec\tnormalisation_cost\n"
        )
        f.writelines(
            f"{result['model_name']}\t{result['accuracy_percentage']}\t{result['inference_time_by_entity']}\t{result['inference_cost_by_entity_USD']}\n"
            for result in results
        )


@click.command()
@click.option(
    "--groundtruth-path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="Path to the groundtruth file containing manually normalized simulation times",
)
@click.option(
    "--runs",
    default=10,
    type=int,
    help="Number of runs of the script",
)
@click.option(
    "--model-evaluation-path",
    type=click.Path(file_okay=True, path_type=Path),
    help="Path to the TSV file for model evaluation results",
)
@click.option(
    "--prompt-path",
    type=click.Path(file_okay=True, path_type=Path),
    help="Path to the llm prompt file",
)
def main_normalizing_simulation_times(
    groundtruth_path: Path,
    runs: int,
    model_evaluation_path: Path,
    prompt_path: Path,
):
    """Normalize the simulation times entities bu running all annexe functions."""
    times = load_simulation_times(groundtruth_path)
    times = times[:]
    save_evaluation_results_in_tsv(
        model_evaluation_path,
        times,
        groundtruth_path,
        prompt_path,
        runs,
    )


if __name__ == "__main__":
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    logger_format = (
        "{time:YYYY-MM-DD HH:mm:ss} "
        "| <level>{level:<8}</level> "
        "| <level>{message}</level>"
    )
    logger.remove()
    logger.add(sys.stdout, format=logger_format, level="DEBUG")
    logger.add(
        f"logs/normalize_simulation_time{timestamp}.log",
        level="DEBUG",
        format=logger_format,
    )
    main_normalizing_simulation_times()
