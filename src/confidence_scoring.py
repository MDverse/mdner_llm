import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================================
# Configuration
# ======================================================================================

DATE_TIME_STR = input("Enter the DATE_TIME_STR and time string analyse (YYYY-MM-DD_HH-MM-SS): ")
BASE_DIR = Path("llm_outputs")
ANNOTATIONS_DIR = Path("annotations")
TAGS = ["MOL", "SOFTN", "SOFTV", "STIME", "TEMP", "FFM"]
OUTPUT_DIR = BASE_DIR / DATE_TIME_STR

TEMP_ORDER = ["0_2", "0_6", "1_0", "1_4", "1_8"]
PROMPT_TYPE = "few_shot"
MODEL = "gpt-4.1-2025-04-14"

# ======================================================================================
# Helper Functions
# ======================================================================================


def get_input_text(filename: str) -> str:
    """Retrieve the input text from the annotations file.

    Args:
        filename (str): The name of the file without extension.

    Returns:
        str: The input text from the annotations file.
    """
    with open(ANNOTATIONS_DIR / (filename + ".json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    input_text = data["annotations"][0][0]
    return input_text


def get_llm_response(filename: str, temp: str) -> List[Dict[str, Any]]:
    """Returns the LLM response annotations for a given filename and temperature.

    Args:
        filename (str): The name of the file without extension.
        temp (str): The temperature setting for the LLM response.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the LLM
        response annotations untreated.
    """
    full_path = (
        BASE_DIR
        / DATE_TIME_STR
        / "annotations"
        / PROMPT_TYPE
        / MODEL
        / temp
        / (filename + ".json")
    )
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    llm_top_response = json.loads(data["response"])
    llm_list_response = llm_top_response["response"]
    return llm_list_response


def compare_surface_form(input_text: str, llm_entity: str) -> bool:
    """Compare the surface form of an LLM entity with the input text to
    check if it exists in the input text.

    Args:
        input_text (str): The input text from the annotations file.
        llm_entity (str): The entity text from the LLM response.

    Returns:
        bool: True if the LLM entity is found in the input text, False otherwise.
    """
    return llm_entity in input_text


def calculate_score(model_list: list) -> float:
    """Calculate the percentage of models that agreed on an entity text + label.

    Args:
        model_list (list): A list of models that AGREED on the entity text and label.
        In our case the list of models corresponds to different temperature settings of
        a same model.

    Returns:
        float: The percentage of models that agreed on the entity text and label.
    """
    all_models = len(TEMP_ORDER)
    number_models_agreed = len(model_list)
    percentage_agreement = (number_models_agreed / all_models) * 100
    return percentage_agreement


def aggregate_annotations_for_file(filename: str, model: str) -> List[Dict[str, Any]]:
    """Aggregate annotations for a given file across different temperature settings.

    Args:
        filename (str): The name of the file without extension.
        model (str): The model name to use for aggregation.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the aggregated
        annotations for the file, including label, entity text, models, and score.
    """
    # Here we will store the final annotations
    # for the file across different temperature settings.
    final_annotations = []

    # Get the input text for the file and initialize a label map
    # {entity: {label: [models]}}
    input_text = get_input_text(filename)
    label_map = defaultdict(lambda: defaultdict(list))  # {entity: {label: [models]}}

    for temp in TEMP_ORDER:
        model_name = f"{model}_temp_{temp}"
        try:
            # Get the LLM response for the current temperature setting
            llm_response = get_llm_response(filename, temp)
        except Exception as e:
            # Some cases the LLM response is hallucinated even in format so we
            # skip those cases and print an error message.
            # This is a safeguard against malformed JSON
            print(f"Error reading LLM response for {filename} at temp {temp}: {e}")
            continue
        # Check if the entity exists in the input text
        for annotation in llm_response:
            label = annotation["label"]
            entity = annotation["text"]
            # If it exists, add it to the label map without duplicates
            if compare_surface_form(input_text, entity):
                if model_name not in label_map[entity][label]:
                    label_map[entity][label].append(model_name)

    # Now we will calculate the scores for each entity and label
    for entity, label_dict in label_map.items():
        # Calculate scores for each label based on number of supporting models
        label_scores = {
            label: calculate_score(models) for label, models in label_dict.items()
        }
        if not label_scores:
            continue

        # Choose the label + text with the highest score for any conflicting label + text
        max_score = max(label_scores.values())
        top_labels = [lbl for lbl, score in label_scores.items() if score == max_score]

        # Break ties randomly if multiple labels share the top score
        best_label = random.choice(top_labels)

        # Add the final annotation with resolved label, models, and score
        final_annotations.append(
            {
                "label": best_label,
                "entity text": entity,
                "models": label_dict[best_label],
                "score": label_scores[best_label],
            }
        )
    return final_annotations


# ======================================================================================
# Main Logic
# ======================================================================================


def main():
    # Prompt user to enter filename (excluding .json extension)
    filename = input("Enter the filename to analyse (without .json): ").strip()

    # Aggregate annotations across temperatures for the selected model
    final = aggregate_annotations_for_file(filename, MODEL)

    # Save the aggregated results as a JSON file
    output_file = OUTPUT_DIR / f"stats/{filename}_consensus_{MODEL}_{PROMPT_TYPE}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4)

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(final)

    # Convert model list to a comma-separated string for readability
    df["models"] = df["models"].apply(lambda x: ", ".join(x))

    # Create a combined string of label + entity for plot axis labeling
    df["entity_label"] = df["label"] + ": " + df["entity text"]

    # Sort entities by their consensus score in descending order
    df_sorted = df.sort_values("score", ascending=False)

    # Define color mapping for labels (fallback to gray if unknown)
    colors = {
        "TEMP": "#ffb3ba",
        "SOFTN": "#ffffba",
        "SOFTV": "orange",
        "STIME": "#baffc9",
        "MOL": "#bae1ff",
        "FFM": "#cdb4db",
    }
    bar_colors = [colors.get(lbl, "#cccccc") for lbl in df_sorted["label"]]

    # Plot horizontal bar chart of consensus scores
    plt.figure(
        figsize=(10, 0.6 * len(df_sorted))
    )  # Adjust height based on number of entities
    plt.barh(df_sorted["entity_label"], df_sorted["score"], color=bar_colors)
    plt.xlabel("Consensus score (%)")
    plt.ylabel("Label: Entity text")
    plt.title("Consensus score for each entity")
    plt.gca().invert_yaxis()  # Show highest score at the top
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(f"{OUTPUT_DIR}/images/consensus_score_{filename}.png")


if __name__ == "__main__":
    main()
