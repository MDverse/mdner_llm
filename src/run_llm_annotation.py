# run_llm_annotation.py

import datetime
import json
import os
import time
import sys

import jinja2
from dotenv import load_dotenv
from groq import Groq, InternalServerError, RateLimitError
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from pathlib import Path

# ======================================================================================
# Configuration
# ======================================================================================

load_dotenv()  # this loads variables from .env into os.environ

# Folder where we have the json files to annotate
ANNOTATIONS_FOLDER = "annotations/"

# Number of texts to annotate
NUMBER_OF_TEXTS_TO_ANNOTATE = 1

# Folder where the prompt templates are stored
PROMPT_PATH = "prompt_templates/"

# Name of the prompts to then name the output folders
LIST_PROMPTS = ["zero_shot", "one_shot", "few_shot"]
# LIST_PROMPTS = ["few_shot_5", "few_shot_15", "few_shot_30"]

# INPUT: Determine which API to use
API_TYPE = input("Which API to use ('groq' or 'openai'): ")

# INPUT: Determine the output style
OUTPUT_STYLE = input("Which output style to use ('json' or 'labels'): ").strip().lower()
if OUTPUT_STYLE not in ["json", "labels"]:
    logger.error("Invalid output style. Choose 'json' or 'labels'.")
    sys.exit(1)

# Models to test depending on the API key
LIST_MODELS_GROQ = [
    "gemma2-9b-it",
    "mistral-saba-24b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "deepseek-r1-distill-llama-70b",
]

LIST_MODELS_OPENAI = [
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    # "gpt-4.1-nano-2025-04-14",
    # "gpt-4o-2024-11-20",
    # "o4-mini-2025-04-16",   # Model not to use for consensus scoring
    # "o3-2025-04-16",        # Model not to use for consensus scoring
    # "o3-mini-2025-01-31",   # Model not to use for consensus scoring
]

# Texts found in the prompt templates
# We want to avoid prompting the LLM to annotate these texts
FILES_TO_AVOID = [
    "zenodo_6970327.json",
    "zenodo_30904.json",
    "zenodo_7192724.json",
    "zenodo_1346073.json",
    "zenodo_1488094.json",
]

# Temperatures to use for consensus scoring
TEMPERATURES = [0.2, 0.6, 1.0, 1.4, 1.8]
CONSENSUS_SCORING = (
    input("Do you want to use consensus scoring (model temperatures)? (yes/no): ")
    .strip()
    .lower()
)
if CONSENSUS_SCORING not in ["yes", "no"]:
    logger.error("Invalid input. Choose 'yes' or 'no'.")
    sys.exit(1)
# If consensus scoring is enabled, we will use the temperatures
if CONSENSUS_SCORING == "yes":
    CONSENSUS_SCORING = True
else:
    CONSENSUS_SCORING = False


class SubResponse(BaseModel):
    """SubResponse class to define the structure of each response
    in the structured JSON response.
    This class contains a label and the text associated with that label.
    Example:
    {"label": "MOL", "text": "water"}
    """

    label: str
    text: str


class GlobalResponse(BaseModel):
    """GlobalResponse class to define the structure of the
    structured JSON response.
    This class contains a list of SubResponse objects.
    Example:
    {
    "response":
        [
        {"label": "MOL", "text": "water"},
        {"label": "SOFTN", "text": "Python"}
        ]
    }
    """

    response: list[SubResponse]


# ======================================================================================
# Client and setup confirmation
# ======================================================================================


def get_api(api_type: str):
    """
    Returns the API client based on the provided API type.
    """
    if api_type == "openai":
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif api_type == "groq":
        return Groq(api_key=os.environ.get("GROQ_API_KEY"))
    else:
        raise ValueError("Invalid API type. Choose 'openai' or 'groq'.")


# Initialize the API client
client = get_api(API_TYPE)

# log the current setup : api type, models, prompts, # of texts, consensus : confirm
logger.info(f"Using API: {API_TYPE}")
logger.info(f"Models: {LIST_MODELS_GROQ if API_TYPE == 'groq' else LIST_MODELS_OPENAI}")
logger.info(f"Prompts: {LIST_PROMPTS}")
logger.info(f"Number of texts to annotate: {NUMBER_OF_TEXTS_TO_ANNOTATE}")
logger.info(f"Output style: {OUTPUT_STYLE}")
logger.info(f"Consensus scoring: {CONSENSUS_SCORING}")

# Is this the correct config? (yes/no)
confirm_setup = input("Is this the correct config? (yes/no): ").strip().lower()
if confirm_setup != "yes":
    logger.error("Configuration not confirmed. Exiting...")
    logger.error("Modify config in '../src/run_llm_annotation.py'")
    sys.exit(1)


# ======================================================================================
# Create output folders with current date and time
# ======================================================================================

# Create timestamped base output directory
date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_BASE = f"llm_outputs/{date_and_time}/"

# Define output subfolders
OUTPUT_FOLDERS = {
    "annotations": os.path.join(OUTPUT_BASE, "annotations"),
    "stats": os.path.join(OUTPUT_BASE, "stats"),
    "images": os.path.join(OUTPUT_BASE, "images"),
}

# Create output directories if they don't exist
for folder in OUTPUT_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

# Select model list based on API type
LIST_MODELS = LIST_MODELS_OPENAI if API_TYPE == "openai" else LIST_MODELS_GROQ

# Create subdirectories for each prompt/model/temperature combination
for prompt in LIST_PROMPTS:
    prompt_folder = os.path.join(OUTPUT_FOLDERS["annotations"], prompt)
    os.makedirs(prompt_folder, exist_ok=True)
    for model in LIST_MODELS:
        os.makedirs(os.path.join(prompt_folder, model), exist_ok=True)
        # If consensus scoring is enabled, create subdirectories for each temperature
        if CONSENSUS_SCORING:
            for temperature in TEMPERATURES:
                temperature_str = str(temperature)
                if "." in temperature_str:
                    temperature_str = temperature_str.replace(".", "_")
                os.makedirs(
                    os.path.join(prompt_folder, model, temperature_str), exist_ok=True
                )

# Save confirmed configuration to a file
# Log file format
logger.add(
    os.path.join(OUTPUT_BASE, f"{Path(__file__).stem}.log"),
    mode="w",
    format="{time:YYYY-MM-DDTHH:mm:ss} | <lvl>{level:<8} | {message}</lvl>",
    level="DEBUG",
)
# Log the configuration to the file
logger.info(f"Using API: {API_TYPE}")
logger.info(f"Models: {LIST_MODELS_GROQ if API_TYPE == 'groq' else LIST_MODELS_OPENAI}")
logger.info(f"Prompts: {LIST_PROMPTS}")
logger.info(f"Number of texts to annotate: {NUMBER_OF_TEXTS_TO_ANNOTATE}")
logger.info(f"Output style: {OUTPUT_STYLE}")
logger.info(f"Consensus scoring: {CONSENSUS_SCORING}")
if CONSENSUS_SCORING:
    logger.info(f"Temperatures: {TEMPERATURES}")
else:
    logger.info("Temperatures: 1.0\n")

# ======================================================================================
# Helper Functions
# ======================================================================================


def process_json_file(json_file: str) -> tuple:
    """Process the JSON file and return the text to annotate and the entities.
    (This file is the ground-truth/input text).

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        tuple: Returns a tuple containing:
            - The text to annotate/input text (str)
            - The entities of the text (list): A list of entities in the format
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    annotation_entry = data["annotations"][0]
    return annotation_entry[0], annotation_entry[1]["entities"]


def load_and_render_prompt(template_path: str, text_to_annotate: str) -> str:
    """Load the prompt template from the specified path and
    render it with the provided text.

    Args:
        template_path (str): _description_
        text_to_annotate (str): _description_

    Returns:
        str: _description_
    """
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()
    template = jinja2.Template(template_content)
    return template.render(text_to_annotate=text_to_annotate)


def labels_text_response(
    model_name: str,
    prompt_used: str,
    temp: float,
) -> str:
    """LLM response using XML-style labels format.
    This function sends a chat completion request to the LLM with the specified
    model, prompt, and temperature.

    Args:
        model_name (str): model to use for the LLM response
        prompt_used (str): full rendered prompt to send to the LLM
        temp (float): temperature to use for the LLM response

    Returns:
        str: The response of the LLM that is a txt response.
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_used}],
        model=model_name,
        temperature=temp,
    )
    return response


def structured_json_response(
    model_name: str,
    prompt_used: str,
    temp: float,
) -> str:
    """LLM response using structured JSON format.
    This function sends a chat completion request to the LLM with the specified
    model, prompt, and temperature. The response is expected to be in a structured
    JSON format defined by the GlobalResponse model.

    Args:
        model_name (str): model to use for the LLM response
        prompt_used (str): full rendered prompt to send to the LLM
        temp (float): temperature to use for the LLM response

    Returns:
        str: The response of the LLM that is a structured JSON response.
        The response is parsed into a GlobalResponse object.
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_used}],
        model=model_name,
        temperature=temp,
    )
    return response


def chat_with_template(
    template_path: str, model_name: str, temp: float, text_to_annotate: str
) -> str:
    """Chat with the model using the specified prompt and text to annotate.
    Handles rate limits and retries.

    Args:
        template_path (str): Path to the prompt template file.
        model_name (str): Model to use for the LLM prompting.
        temp (float): Temperature to use for the LLM response.
        text_to_annotate (str): Input text to annotate by the LLM

    Returns:
        str: The response of the LLM that is a txt response or a structured JSON response.
    """
    delay = 1
    max_retries = 10

    prompt_rendered = load_and_render_prompt(template_path, text_to_annotate)

    for attempt in range(max_retries + 1):
        response = None
        try:
            # Depending on the output style, we communicate with the model differently
            # That is why we have two functions to handle the response
            if OUTPUT_STYLE == "labels":
                response = labels_text_response(
                    model_name=model_name, prompt_used=prompt_rendered, temp=temp
                )
            elif OUTPUT_STYLE == "json":
                response = structured_json_response(
                    model_name=model_name, prompt_used=prompt_rendered, temp=temp
                )
            content = response.choices[0].message.content
            usage = response.usage

            # Extract the three token counts
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Bundle them in a dict
            usage_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            return content, usage_dict

        # Handle specific exceptions for rate limits and server errors
        except RateLimitError as err:
            status = getattr(err, "status_code", None)
            if status in (498, 499, 429) and attempt < max_retries:
                logger.warning(
                    f"{status} error for model {model_name}, retrying in {delay}s..."
                )
                time.sleep(delay)
                delay *= 5
                continue
            else:
                logger.error(f"Error from model {model_name}: {err}")
                raise
        except InternalServerError as err:
            status = getattr(err, "status_code", None)
            if status in (503, 500, 502) and attempt < max_retries:
                logger.warning(
                    f"{status} error for model {model}, retrying in {delay}s..."
                )
                time.sleep(delay)
                delay *= 5
                continue
            else:
                logger.error(f"Error from model {model_name}: {err}")
                raise


def save_response_as_json(response_text: str, output_path: str) -> None:
    """Save the LLM response text as a JSON file.

    Args:
        response_text (str): LLM response text to save.
        output_path (str): Path to save the response as a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response_text, f, ensure_ascii=False, indent=2)


# ======================================================================================
# Main Logic
# ======================================================================================


def main():
    logger.info(f"Starting LLM annotation process at {date_and_time}")
    # Count the number of texts to annotate to respect the limit
    number_texts = 0

    for filename in os.listdir(ANNOTATIONS_FOLDER):
        if number_texts >= NUMBER_OF_TEXTS_TO_ANNOTATE:
            break

        # Check if the file is a JSON file and has the correct format "_"
        # Grab file
        if (
            filename.endswith(".json")
            and filename.count("_") == 1
            and filename not in FILES_TO_AVOID
        ):
            number_texts += 1
            logger.info(f"Processing file {number_texts}: {filename}...")
            input_path = os.path.join(ANNOTATIONS_FOLDER, filename)
            input_text, _ = process_json_file(input_path)

            # Grab prompt
            for prompt in LIST_PROMPTS:
                logger.info(f"Testing prompt: {prompt} ------")
                prompt_folder = os.path.join(OUTPUT_FOLDERS["annotations"], prompt)

                # Grab model
                for model in LIST_MODELS:
                    logger.info(f"Testing model: {model}")
                    output_model_folder = os.path.join(prompt_folder, model)

                    # If we are doing consensus scoring, we also test out various temps
                    if CONSENSUS_SCORING:
                        # Grab a temperature
                        for temperature in TEMPERATURES:
                            temperature_str = str(temperature).replace(".", "_")
                            output_temp_folder = os.path.join(
                                output_model_folder, temperature_str
                            )

                            logger.info(f"Testing temperature: {temperature}")
                            # Chat with the model
                            response, usage = chat_with_template(
                                template_path=os.path.join(
                                    PROMPT_PATH, f"{prompt}.txt"
                                ),
                                model_name=model,
                                temp=temperature,
                                text_to_annotate=input_text,
                            )

                            # Save the response
                            output_path = os.path.join(output_temp_folder, filename)
                            data = {
                                "model": model,
                                "text_to_annotate": input_text,
                                "response": response,
                                "usage": usage,
                            }

                            save_response_as_json(data, output_path)

                    # If we are not doing consensus scoring, we just test the prompt and
                    # model without temperatures
                    else:
                        # Chat with the model
                        response, usage = chat_with_template(
                            template_path=os.path.join(PROMPT_PATH, f"{prompt}.txt"),
                            model_name=model,
                            temp=1,
                            text_to_annotate=input_text,
                        )

                        # Save the response
                        output_path = os.path.join(output_model_folder, filename)
                        data = {
                            "model": model,
                            "text_to_annotate": input_text,
                            "response": response,
                            "usage": usage,
                        }

                        save_response_as_json(data, output_path)

    logger.success("Annotation process completed.")


if __name__ == "__main__":
    main()
