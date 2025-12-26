"""Utility functions for the annotation JSON validation process."""

# Import necessary libraries
import json
import os
import re
import textwrap
import unicodedata
from pathlib import Path

import instructor
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from instructor.core import (
    InstructorRetryException,
)
from instructor.core import (
    ValidationError as InstructorValidationError,
)
from instructor.core.client import Instructor
from instructor.exceptions import (
    ValidationError as InstructorValidationError,  # noqa: F811
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from loguru import logger
from openai.types.chat import ChatCompletion
from pydantic import ValidationError
from pydantic import ValidationError as PydanticValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_core import ValidationError as CoreValidationError
from spacy import displacy
from tqdm import tqdm

from models.pydantic_output_models import ListOfEntities, ListOfEntitiesPositions


# FUNCTIONS
def check_json_validity(json_string: str) -> bool:
    """Check if the given string is a valid JSON.

    Parameters
    ----------
    json_string : str
        The string to check.
    """
    try:
        json.loads(json_string)
        # print("Valid JSON ✅")
        return True
    except json.JSONDecodeError:
        # print("Invalid JSON ❌")
        return False


def normalize_text(text: str) -> str:
    """Normalize text by removing special characters and converting to lowercase.

    Parameters
    ----------
    text : str
        The text to normalize.

    Returns
    -------
    str
        The normalized text.
    """
    # Normalize unicode characters
    text_normalized = unicodedata.normalize("NFKD", text)
    # Convert to lowercase
    text_normalized = text_normalized.lower()
    # Remove extra whitespace
    text_normalized = re.sub(r"\s+", " ", text_normalized)
    # Strip leading and trailing whitespace
    text_normalized = text_normalized.strip()
    return text_normalized


def find_hallucinated_entities(
    annotations: ListOfEntities, original_text: str
) -> list[dict[int, str]]:
    """Identify entities in the annotation that are not present in the original text.

    Parameters
    ----------
    annotations : ListOfEntities
        The annotation output to check.
    original_text : str
        The original text to compare against.

    Returns
    -------
    list[dict[int, str]]
        A list of dictionaries containing the index and text of hallucinated entities.
    """
    hallucinated_entities = []
    # Normalize the original text
    text_normalized = normalize_text(original_text)
    for i, annotation in enumerate(annotations):
        if annotation is None:
            continue
        # Initialize a temporary list to store hallucinated entities for this annotation
        # Check each entity in the annotation
        temp_list = [entity.text for entity in annotation.entities
                     if entity.text.lower() not in text_normalized]
        # If there are hallucinated entities, add them to the result list
        if temp_list != []:
            hallucinated_entities.append({i: temp_list})
    return hallucinated_entities


def report_hallucinated_entities(
    hallucinated_entities: dict[int, list[str]], original_text: str
) -> None:
    """Report hallucinated entities found in the annotation output.

    Parameters
    ----------
    hallucinated_entities : Dict[int, List[str]]
        A dictionary containing the index and a list of hallucinated entities.
    original_text : str
        The original text to compare against.
    """
    print("\n" + "=" * 80)
    print("📝 Original text:")
    print("=" * 80)
    wrapped_text = textwrap.fill(original_text, width=120)
    print(wrapped_text)

    if hallucinated_entities != []:
        print("\n" + "=" * 80)
        print(
            f"⚠️ Hallucinated entities detected for {len(hallucinated_entities)} \
            annotations:"
        )
        print("=" * 80)
        # Get each dictionary representing hallucinated entities
        for hallucinated_entity in hallucinated_entities:
            # Get the list of hallucinated texts from each dictionary
            for index, texts in hallucinated_entity.items():
                print(f"Annotation {index} :")
                for text in texts:
                    print(f"  - {text}")
                print()
    else:
        print("✅ No hallucinated entities found.")


def is_annotation_in_text(response: ListOfEntities, original_text: str) -> bool:
    """Validate the content of the annotation output against the original text.

    Parameters
    ----------
    response : ListOfEntities
        The annotation output to validate.
    original_text : str
        The original text to compare against.

    Returns
    -------
    bool
        True if the entities in the response are present in the original text,
        False otherwise.
    """
    # Normalize the original text
    text_normalized = normalize_text(original_text)

    # Check if all entity texts are present in the original text
    return all(entity.text.lower() in text_normalized for entity in response.entities)


def validate_annotation_output_format(
    response: ChatCompletion | ListOfEntities,
) -> ListOfEntities | None:
    """Validate the annotation output against the ListOfEntities schema.

    Parameters
    ----------
    response : Union[ChatCompletion, ListOfEntities]
        The response from the LLM model to validate.

    Returns
    -------
    bool
        True if the response is valid according to the ListOfEntities schema,
        False otherwise.
    """
    # It's already a ListOfEntities instance
    if isinstance(response, ListOfEntities):
        return response
    if isinstance(response, ChatCompletion):
        # Extract the content from the ChatCompletion response
        response_str = response.choices[0].message.content
        try:
            # Validate the response string against the ListOfEntities schema
            parsed_response = ListOfEntities.model_validate_json(response_str)
            return parsed_response
        except ValidationError:
            return None


def run_annotation_stats(
    text_to_annotate: str,
    model_name: str,
    client,
    num_iterations: int = 100,
    *,
    validation: bool = False,
) -> tuple[list[ListOfEntities | ChatCompletion], int, int]:
    """
    Run multiple annotation attempts on the given text, with/out a validation schema.

    Parameters
    ----------
    text_to_annotate : str
        The text to annotate.
    model_name : str
        The name of the LLM model to use.
    client : instructor.core.client.Instructor
        The LLM client to use (either Groq , OpenAI or OpenRouter).
    num_iterations : int, optional
        Number of annotation attempts to run, by default 100
    validation : bool, optional
        Whether to use validation schema, by default False

    Returns
    -------
    tuple[list[ListOfEntities | ChatCompletion], int, int]:
        The responses and he number of valid and invalid responses.
    """
    valid_count = 0
    invalid_count = 0
    list_of_entities = []

    desc = (
        f"Running annotations {'with' if validation else 'without'} validation schema"
    )
    for _ in tqdm(range(num_iterations), desc=desc):
        response = annotate(text_to_annotate, model_name, client, validation=validation)
        validated_output = validate_annotation_output_format(response)
        list_of_entities.append(validated_output)
        if validated_output:
            if is_annotation_in_text(validated_output, text_to_annotate):
                valid_count += 1
        else:
            invalid_count += 1

    return list_of_entities, valid_count, invalid_count


def compare_annotation_validation(
    text_to_annotate: str,
    model_name: str,
    client,
    num_iterations: int = 100,
) -> dict[str, dict[str, int | list[ListOfEntities | ChatCompletion]]]:
    """
    Run annotation tests both with and without validation schema and print results.

    Parameters
    ----------
    text_to_annotate : str
        The text to annotate.
    model_name : str
        The name of the LLM model to use.
    client : instructor.core.client.Instructor
        The LLM client to use (either Groq , OpenAI or OpenRouter).
    num_iterations : int, optional
        Number of annotation attempts to run, by default 100

    Returns
    -------
    Dict[str, Dict[str, Union[int, List[ListOfEntities|ChatCompletion]]]]
        A dictionary summarizing the results with and without validation schema. As:
        {
            "without_validation": {
                "valid": int,
                "invalid": int,
                "examples": List[ListOfEntities|ChatCompletion]
            },
            "with_validation": {
                "valid": int,
                "invalid": int,
                "examples": List[ListOfEntities|ChatCompletion]
            }
        }
    """
    # Without validation
    list_of_entities_wo, valid_wo, invalid_wo = run_annotation_stats(
        text_to_annotate, model_name, client, num_iterations, validation=False
    )
    # With validation
    list_of_entities_w, valid_w, invalid_w = run_annotation_stats(
        text_to_annotate, model_name, client, num_iterations, validation=True
    )

    print("\n" + "=" * 80)
    print(f"📝 Input text:\n{textwrap.fill(text_to_annotate, width=120)}")
    print("=" * 80 + "\n")
    print("\n" + "=" * 80)
    print(f"📊 Summary after {num_iterations} runs with {model_name}:")
    print("=" * 80 + "\n")
    print(
        f"❌ Without validation schema : {valid_wo / num_iterations * 100:.1f}% \
        valid responses"
    )
    print(
        f"✅ With validation schema    : {valid_w / num_iterations * 100:.1f}% \
        valid responses"
    )

    return {
        "without_validation": {
            "valid": valid_wo,
            "invalid": invalid_wo,
            "examples": list_of_entities_wo,
        },
        "with_validation": {
            "valid": valid_w,
            "invalid": invalid_w,
            "examples": list_of_entities_w,
        },
    }


def assign_all_instructor_clients(
    open_router_models: list[str]
) -> dict[str, Instructor]:
    """
    Assign a client to each model.

    Returns
    -------
    Dict[str, Instructor]
        Dictionary mapping model names to Instructor clients.
    """
    load_dotenv()
    models_with_providers = dict.fromkeys(open_router_models, "openrouter")
    return {
        model: instructor.from_provider(
            f"{provider}/{model}", mode=instructor.Mode.JSON
        )
        for model, provider in models_with_providers.items()
    }


def assign_all_llamaindex_clients(
    open_router_models: list[str]
) -> dict[str, OpenAI | OpenRouter]:
    """
    Assign a LlamaIndex client to each model.

    Returns
    -------
    Dict[str, OpenAI | OpenRouter]
        Dictionary mapping model names to LlamaIndex OpenAI | OpenRouter clients.
    """
    load_dotenv()
    clients = {}
    api_key = os.getenv("OPENROUTER_API_KEY")
    for model_name in open_router_models:
        llm = OpenRouter(
            model=model_name,
            api_key=api_key
        )
        clients[model_name] = llm.as_structured_llm(output_cls=ListOfEntities)

    return clients


def assign_all_pydanticai_clients(
    open_router_models: list[str]
) -> dict[str, OpenAIChatModel]:
    """
    Assign a PydanticAI for each PydanticAI model.

    Returns
    -------
    Dict[str, OpenAIChatModel]
        Dictionary mapping model names to PydanticAI clients.
    """
    load_dotenv()
    clients = {}
    api_key = os.getenv("OPENROUTER_API_KEY")
    for model_name in open_router_models:
        llm = OpenAIChatModel(model_name, provider=OpenRouterProvider(api_key=api_key))
        clients[model_name] = llm

    return clients


def annotate(
    text: str,
    model: str,
    client: Instructor | OpenAI | OpenRouter | OpenAIChatModel,
    tag_prompt: str,
    validator: str = "instructor",
    max_retries: int = 3,
    prompt_json_path: Path = Path("prompts/json_few_shot.txt"),
    prompt_positions_path: Path = Path("prompts/json_with_positions_few_shot.txt"),
    *,
    validation: bool = True
) -> ChatCompletion | str | ListOfEntities | ListOfEntitiesPositions:
    """Annotate the given text using the specified model.

    If validation, the output will be validated against the GlobalResponse schema.

    Parameters
    ----------
    text : str
        The text to annotate.
    model : str
        The name of the LLM model to use.
    client : Union[Instructor | OpenAI | OpenRouter | OpenAIChatModel]
        The LLM client to use (either from Instructor, llamaindex or pydantic_ai).
    validator: str, optional
        The name of the output validator package between "instructor", "llamaindex",
        "pydanticai" (Default is "instructor").
    max_retries : int, optional
        Maximum number of retries for the API call in case of failure, by default 3.
    prompt_json_path: Path
        The path to retrieve prompt to annotate only entities in JSON format.
    prompt_positions_path: Path
        The path to retrieve prompt to annotate entities+positions in JSON format.
    validation : bool, optional
        Whether to validate the output against the schema, by default True.

    Returns
    -------
    Union[ListOfEntities,ChatCompletion]
        The response from the LLM model, either validated or raw output.
    """
    # Set response model and retries based on validation flag
    if validation:
        if tag_prompt == "json":
            response_model = ListOfEntities
        else:
            response_model = ListOfEntitiesPositions
    else:
        response_model = None
        max_retries = 0

    # Set prompt based on positions of the start and end or without
    prompt_json = prompt_json_path.read_text(encoding="utf-8")
    prompt_positions = prompt_positions_path.read_text(encoding="utf-8")
    prompt = prompt_json if tag_prompt == "json" else prompt_positions

    result = None
    # Query the LLM client for annotation
    if validator == "instructor":
        try:
            result = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                        "content": "Extract entities as structured JSON."},
                    {"role": "user",
                        "content": f"{prompt}\nThe text to annotate:\n{text}"}
                ],
                response_model=response_model,
                max_retries=max_retries,
            )
        except InstructorRetryException as e:
            logger.warning(
                f"    ⚠️ Validated annotation failed after {e.n_attempts} attempts."
            )
            # logger.warning(f"Total usage: {e.total_usage.total_tokens} tokens")
            return str(e.last_completion)

        except InstructorValidationError as e:
            # logger.error(e.errors)
            return str(e.raw_output)

    elif validator == "llamaindex":
        try:
            if model.startswith("openai"):
                output_model = (
                    ListOfEntities if tag_prompt == "json"
                    else ListOfEntitiesPositions
                )
                client = OpenAI(model=model.split("/")[1])
                client = client.as_structured_llm(output_cls=output_model)

            response = client.complete(f"{prompt}\nThe text to annotate:\n{text}")
            result = response.raw
        except (PydanticValidationError, CoreValidationError, ValueError) as e:
            return str(e)

    elif validator == "pydanticai":
        agent = Agent(
            model=client,
            output_type=ListOfEntities,
            retries=max_retries,
            system_prompt=("Extract entities as structured JSON."),
        )
        try:
            response = agent.run_sync(f"{prompt}\nThe text to annotate:\n{text}")
            result = response.output
        except (
            PydanticValidationError,
            CoreValidationError,
            UnexpectedModelBehavior,
        ) as e:
            return str(e)

    return result


def annotate_forced_validation(
    text: str,
    model: str,
    client: Instructor,
    prompt: str,
    max_retries: int = 10,
    *,
    validation: bool = True
) -> dict:
    """Annotate the given text using the specified model.

    If validation is True, the output will be validated against the GlobalResponse.

    Parameters
    ----------
    text : str
        The text to annotate.
    model : str
        The name of the LLM model to use.
    client : Union[instructor.core.client.Instructor | OpenAI]
        The LLM client to use (either Groq or OpenAI).
    prompt: str :
        The prompt template to use for llm.
    validation : bool, optional
        Whether to validate the output against the schema, by default True
    max_retries : int, optional
        Maximum number of retries for the API call in case of failure, by default 35

    Returns
    -------
    dict
        The annotated text as a structured JSON.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            result = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract entities as structured JSON.",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\nThe text to annotate:\n{text}",
                    },
                ],
                response_model=ListOfEntities if validation else None,
                max_retries=3 if validation else 0,
            )

            if validation:
                return result.model_dump_json(indent=2)
            else:
                raw_content = result.choices[0].message.content
                try:
                    parsed_json = json.loads(raw_content)
                    return json.dumps(parsed_json, indent=2)
                except json.JSONDecodeError:
                    return raw_content

        except InstructorValidationError as e:
            attempt += 1
            print(f"🔍 Instructor Validation Error on attempt {attempt}/{max_retries}:")
            print(e.errors)
            if attempt >= max_retries:
                print("Max retries reached. Returning raw output.")
                return str(e.raw_output)
            else:
                print("Retrying...")


def visualize_entities(
    text: str, entities: ChatCompletion | ListOfEntities | str
) -> None:
    """Visualize the extracted entities in a readable format.

    Parameters
    ----------
    text : str
        The original text from which entities were extracted.
    entities : Union[ChatCompletion, ListOfEntities, str]
        The extracted entities to visualize.
    """
    # Define colors for each entity type
    colors = {
        "TEMP": "#ffb3ba",
        "SOFTNAME": "#ffffba",
        "SOFTVERS": "#orange",
        "STIME": "#baffc9",
        "MOL": "#bae1ff",
        "FFM": "#cdb4db",
    }
    options = {"colors": colors}
    ents = []
    # If entities is a ChatCompletion
    if isinstance(entities, ChatCompletion):
        content = json.loads(entities.choices[0].message.content)
        entity_list = content["entities"]
        print(f"Entity list from ChatCompletion: {entity_list}")
        for ent in entity_list:
            # Escape special regex characters like +, ., etc.
            pattern = re.escape(ent["text"])
            # Find all occurrences (case-insensitive)
            matches = [
                {"start": m.start(), "end": m.end(), "label": ent["label"]}
                for m in re.finditer(pattern, text, flags=re.IGNORECASE)
            ]
            ents.extend(matches)
    # If entities is a ListOfEntities
    elif hasattr(entities, "entities"):  # or isinstance(entities, ListOfEntities)
        entity_list = entities.entities
        for ent in entity_list:
            pattern = re.escape(ent.text)
            matches = [
                {"start": m.start(), "end": m.end(), "label": ent["label"]}
                for m in re.finditer(pattern, text, flags=re.IGNORECASE)
            ]
            ents.extend(matches)
    # If entities is a string (raw JSON)
    elif isinstance(entities, str):
        content = json.loads(entities)
        entity_list = content["entities"]
        for ent in entity_list:
            pattern = re.escape(ent["text"])
            matches = [
                {"start": m.start(), "end": m.end(), "label": ent["label"]}
                for m in re.finditer(pattern, text, flags=re.IGNORECASE)
            ]
            ents.extend(matches)
    # Prepare the data for displacy
    spacy_format = {"text": text, "ents": ents}
    displacy.render(spacy_format, style="ent", manual=True, options=options)


def remove_entity_annotation_file(file_name: str, entities_to_remove: list) -> None:
    """
    Remove specific entities from a formatted annotation JSON file.

    Parameters
    ----------
    file_name : str
        Name of the JSON file located in the formatted annotations directory.
    entities_to_remove : list
        A list of tuples of the form (label, text) specifying which entities
        should be removed. Example: [("MOL", "water"), ("TEMP", "37°C")]
    """
    file_path = f"../data/formated_annotations/{file_name}"

    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)
        original_count = len(data["entities"])

        data["entities"] = [
            ent
            for ent in data["entities"]
            if (ent["label"], ent["text"]) not in entities_to_remove
        ]
        print(data["entities"])

        removed_count = original_count - len(data["entities"])
        print(f"{removed_count} entité(s) supprimée(s) du fichier {file_name}")

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def find_entity_positions(raw_text: str, entity_text: str) -> list[tuple[int, int]]:
    """Find all occurrences of an entity text inside the raw annotation text.

    This function scans the raw text and returns every (start, end) character
    index pair where the entity text appears. It supports repeated occurrences.

    Parameters
    ----------
    raw_text : str
        The full text in which to search for occurrences.
    entity_text : str
        The substring corresponding to the entity that should be located.

    Returns
    -------
    list[tuple[int, int]]
        A list of (start, end) positions for each occurrence of the entity text.
        Returns an empty list if the text is not found.
    """
    positions = []
    start_idx = 0

    # Search for all occurrences iteratively
    while True:
        start = raw_text.find(entity_text, start_idx)
        if start == -1:
            break  # no more occurrences

        end = start + len(entity_text)
        positions.append((start, end))

        # Move search index forward to avoid infinite loops
        start_idx = end

    return positions


def add_entity_annotation_file(file_name: str, new_entities: list):
    """Add new entities to an existing formatted annotation file.

    This function loads an annotation file, finds all occurrences of new entity
    texts inside the raw text, and appends corresponding entity dictionaries to
    the "entities" list. It supports inserting multiple labels and occurrences
    per label.

    Parameters
    ----------
    file_name : str
        Name of the formatted annotation JSON file.
    new_entities : list
        A list of (label, text) tuples representing the entities to insert.
        Example: [("MOL", "water"), ("TEMP", "37°C")]
    """
    # Load the annotation file
    file_path = f"../data/formated_annotations/{file_name}"
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    raw_text = data["raw_text"]
    for label, text in new_entities:
        positions = find_entity_positions(raw_text, text)

        for start, end in positions:
            entity_dict = {"label": label, "text": text, "start": start, "end": end}

            if entity_dict not in data["entities"]:
                data["entities"].append(entity_dict)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def run_annotations(
    text_to_annotate: str,
    model_name: str,
    client,
    validator: str = "instructor",
    num_iterations: int = 100,
    *,
    validation: bool = False,
) -> list[ListOfEntities | ChatCompletion]:
    """
    Run multiple annotation attempts on the given text, with/without validation schema.

    Parameters
    ----------
    text_to_annotate : str
        The text to annotate.
    model_name : str
        The name of the LLM model to use.
    client : instructor.core.client.Instructor
        The LLM client to use (either OpenAI or OpenRouter).
    validator: str = "instructor"
        The name of the output validator package between "instructor", "llamaindex",
        "pydanticai" (Default is "instructor").
    num_iterations : int, optional
        Number of annotation attempts to run, by default 100
    validation : bool, optional
        Whether to use validation schema, by default False

    Returns
    -------
    list[ListOfEntities | ChatCompletion]:
        List of the LLM response for the annotations of the same text with same prompt.
    """
    list_of_responses = []

    desc = f"Running annotations {f'with {validator}' if validation else 'without'} \
    validation schema..."
    logger.debug(f"{'🟢' if validation else '🔴'}{desc}")
    for _ in tqdm(
        range(num_iterations),
        desc=desc,
        colour="blue",
        ncols=200,
        unit="annotation",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]",
    ):
        response = annotate(
            text_to_annotate, model_name, client, validator, validation=validation
        )
        list_of_responses.append(response)

    # logger.success(f"Completed {len(list_of_responses)} annotations successfully!\n")
    return list_of_responses


def convert_annotations(response, text_to_annotate):
    """
    Convert custom entity list to spaCy displaCy format.

    If response is ListOfEntities → compute spans by locating text occurrences.
    If response is ListOfEntitiesPositions → use provided start/end positions.

    Parameters
    ----------
    response : ListOfEntities | ListOfEntitiesPositions

    Returns
    -------
    List[dict]  (spaCy displaCy manual format)
    """
    ents = []

    # ---------- CASE 1 : response is ListOfEntitiesPositions ----------
    # Already has (start, end, label)
    if isinstance(response, ListOfEntitiesPositions):
        ents.extend({
                "start": ent.start,
                "end": ent.end,
                "label": ent.label
            } for ent in response.entities)
        return [{"text": text_to_annotate, "ents": ents}]

    # ---------- CASE 2 : response is ListOfEntities ----------
    # We must find spans in TEXT_TO_ANNOTATE
    if isinstance(response, ListOfEntities):
        text_lower = text_to_annotate.lower()
        consumed = [False] * len(text_to_annotate)

        for entity in response.entities:
            span_text = entity.text
            span_lower = span_text.lower()

            start = -1
            search_pos = 0

            while True:
                start = text_lower.find(span_lower, search_pos)
                if start == -1:
                    break

                end = start + len(span_text)

                # avoid overlap
                if not any(consumed[start:end]):
                    for i in range(start, end):
                        consumed[i] = True

                    ents.append({
                        "start": start,
                        "end": end,
                        "label": entity.label
                    })
                    break
                else:
                    search_pos = start + 1

            if start == -1:
                print(f"⚠️ Warning: entity '{span_text}' not found in text.")

        return [{"text": text_to_annotate.replace("\n", " "), "ents": ents}]


def visualize_llm_annotation(response: ListOfEntities | ListOfEntitiesPositions,
                             text_to_annotate: str):
    """
    Visualize named entities from LLM annotations using spaCy's displaCy.

    Parameters
    ----------
    response (ListOfEntities | ListOfEntitiesPositions):
        The annotated entities returned by the LLM.
    text_to_annotate (str):
        The original text on which entities were predicted.
    """
    colors = {
        "TEMP": "#ffb3ba",
        "SOFTNAME": "#ffffba",
        "SOFTVERS": "#ffffe4",
        "STIME": "#baffc9",
        "MOL": "#bae1ff",
        "FFM": "#cdb4db",
    }
    options = {"colors": colors}
    print("=" * 80)
    print("🧐 VISUALIZATION OF ENTITIES ")
    print("=" * 80)
    converted_data = convert_annotations(response, text_to_annotate)
    displacy.render(converted_data, style="ent", manual=True, options=options)
    print()


def plot_top_entities(df, top_k=10, class_name="Class"):
    """
    Plot the top-k entities from a DataFrame as a bar chart with counts above bars.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns 'entity' and 'count'.
    top_k : int, optional
        Number of top entities to display (default is 10).
    class_name : str, optional
        Name of the class to use in the title (default is "Class").
    """
    # Sort by count descending
    top_df = df.sort_values("count", ascending=False).head(top_k)
    n = len(top_df)
    cmap = plt.get_cmap("viridis", n)
    colors = [cmap(i) for i in range(n)]

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_df["entity"], top_df["count"], color=colors, edgecolor="black")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(
        f"Top {top_k} entities for class {class_name} (Total:{len(df)})",
        fontsize=14,
        weight="bold"
    )
    plt.xlabel("Entity", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(top_df["count"]) * 0.01,  # slightly above the bar
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold"
        )

    plt.tight_layout()
    plt.show()
