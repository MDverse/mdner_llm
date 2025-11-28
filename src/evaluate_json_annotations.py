"""Evaluate LLM outputs for several models.

We ask several models :
- GPT-4 d'OpenAI
- Gemini de Google
- kimik2 de MoonshotAI
- Qwen de QwenAI
- llama-3.1-8b-instruct de meta
- llama-3.3-70b-instruct de meta

to annotate a text related to molecular dynamics with entities : molecule, simulation time, force field, software name, software version and temperature with the same prompt
It save the records annotations for each test (no_validation, validation_instructor, validation_llamaindex, validation_pydanticai) with is_json_valid, have_no_hallucination, is_annotation_valid tags
and lastly it save a xlsx file to summarize the statistics for each model to benchmark.



Usage :
=======
    uv run src/evaluate_json_annotations.py [--log] [--out-path]

Arguments:
==========
    --log : (optional)
        Enable logging to a file.
    --out-path : (optional)
        File path to save the evaluation results. Default is "results/json_evaluation_stats/{timestamp}.xlsx".

Example:
========
    uv run src/evaluate_json_annotations.py --log
"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "AGPL-3.0 license"
__date__ = "2025"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import os
import json
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Union, List

import instructor
from instructor.core.client import Instructor
from instructor.core import (
    InstructorRetryException,
    ValidationError as InstructorValidationError,
)
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai import Agent
from pydantic import ValidationError as PydanticValidationError
from pydantic_core import ValidationError as CoreValidationError

# UTILITY IMPORTS
from utils import (
    ListOfEntities,
    PROMPT,
    validate_annotation_output_format,
    is_annotation_in_text,
)


# CONSTANTS
OUTPUT_DIR = "results/json_evaluation_stats"
# To adapt on your needs :
NB_ITERATIONS = 5
TEXT_TO_ANNOTATE = """Improved Coarse-Grained Modeling of Cholesterol-Containing Lipid Bilayers
Cholesterol trafficking, which is an essential function in mammalian cells, is intimately connected to molecular-scale interactions through cholesterol modulation of membrane structure and dynamics and interaction with membrane receptors. Since these effects of cholesterol occur on micro- to millisecond time scales, it is essential to develop accurate coarse-grained simulation models that can reach these time scales. Cholesterol has been shown experimentally to thicken the membrane and increase phospholipid tail order between 0 and 40% cholesterol, above which these effects plateau or slightly decrease. Here, we showed that the published MARTINI coarse-grained force-field for phospholipid (POPC) and cholesterol fails to capture these effects. Using reference atomistic simulations, we systematically modified POPC and cholesterol bonded parameters in MARTINI to improve its performance. We showed that the corrections to pseudobond angles between glycerol and the lipid tails and around the oleoyl double bond particle (the angle-corrected model ) slightly improves the agreement of MARTINI with experimentally measured thermal, elastic, and dynamic properties of POPC membranes. The angle-corrected model improves prediction of the thickening and ordering effects up to 40% cholesterol but overestimates these effects at higher cholesterol concentration. In accordance with prior work that showed the cholesterol rough face methyl groups are important for limiting cholesterol self-association, we revised the coarse-grained representation of these methyl groups to better match cholesterol-cholesterol radial distribution functions from atomistic simulations. In addition, by using a finer-grained representation of the branched cholesterol tail than MARTINI, we improved predictions of lipid tail order and bilayer thickness across a wide range of concentrations. Finally, transferability testing shows that a model incorporating our revised parameters into DOPC outperforms other CG models in a DOPC/cholesterol simulation series, which further argues for its efficacy and generalizability. These results argue for the importance of systematic optimization for coarse-graining biologically important molecules like cholesterol with complicated molecular structure.
"""
MODELS_OPENAI = [
    "o3-mini-2025-01-31",
    "gpt-4o-mini-2024-07-18",
    "gpt-5-mini-2025-08-07",
]
MODELS_OPENROUTER = [
    "meta-llama/llama-3.1-8b-instruct",
    # "meta-llama/llama-3-70b-instruct",
    "moonshotai/kimi-k2-thinking",
    "google/gemini-2.5-flash",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-chat-v3-0324",
]


# FUNCTIONS
def parse_arguments() -> Tuple[bool, str, str]:
    """Parse command line arguments.

    Returns:
    --------
    log : bool
        Whether to enable logging to a file.
    out_path : str
        The output folder path for the evaluation and annotation results.
    file_name : str
        The output ending of file names for the evaluation and annotation results.
    """
    logger.info("Starting to parse command-line arguments...")
    parser = argparse.ArgumentParser(
        description="Evaluate LLM json output annotations."
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Enable logging to a file.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=f"{OUTPUT_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        help="Output file path for the evaluation and annotation results.",
    )

    args = parser.parse_args()
    # retrieve output directory
    folder_out_path = os.path.dirname(args.out_path)
    file_name = os.path.basename(args.out_path)

    logger.debug(f"Logger: '{args.log}'")
    logger.debug(f"Output folder path: '{folder_out_path}'")
    logger.debug(f"Output ending of file names: '{file_name}'")

    logger.success("Parsed arguments sucessfully!\n")
    return args.log, folder_out_path, file_name


def assign_all_instructor_clients() -> Dict[str, Instructor]:
    """
    Assign a client to each model.

    Returns:
    --------
    Dict[str, Instructor]
        Dictionary mapping model names to Instructor clients.
    """
    load_dotenv()
    models_with_providers = {m: "openai" for m in MODELS_OPENAI}
    models_with_providers.update({m: "openrouter" for m in MODELS_OPENROUTER})
    return {
        model: instructor.from_provider(
            f"{provider}/{model}", mode=instructor.Mode.JSON
        )
        for model, provider in models_with_providers.items()
    }


def assign_all_llamaindex_clients() -> Dict[str, OpenAI | OpenRouter]:
    """
    Assign a LlamaIndex client to each model.

    Returns
    -------
    Dict[str, OpenAI | OpenRouter]
        Dictionary mapping model names to LlamaIndex OpenAI | OpenRouter clients.
    """
    load_dotenv()
    clients = {}
    for model_name in MODELS_OPENROUTER:
        llm = OpenRouter(
            model=model_name,
        )
        clients[model_name] = llm.as_structured_llm(output_cls=ListOfEntities)

    for model_name in MODELS_OPENAI:
        llm = OpenAI(
            model=model_name,
        )
        clients[model_name] = llm.as_structured_llm(output_cls=ListOfEntities)

    return clients


def assign_all_pydanticai_clients() -> Dict[str, OpenAIChatModel]:
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
    for model_name in MODELS_OPENROUTER:
        llm = OpenAIChatModel(model_name, provider=OpenRouterProvider(api_key=api_key))
        clients[model_name] = llm

    for model_name in MODELS_OPENAI:
        llm = OpenAIChatModel(model_name)
        clients[model_name] = llm

    return clients


def annotate(
    text: str,
    model: str,
    client: Union[Instructor | OpenAI | OpenRouter | OpenAIChatModel],
    validator: str = "instructor",
    validation: bool = True,
    max_retries: int = 3,
) -> Union[ChatCompletion, str]:
    """Annotate the given text using the specified model.
    If validation is True, the output will be validated against the GlobalResponse schema.

    Parameters
    ----------
    text : str
        The text to annotate.
    model : str
        The name of the LLM model to use.
    client : Union[Instructor | OpenAI | OpenRouter | OpenAIChatModel]
        The LLM client to use (either from Instructor, llamaindex or pydantic_ai).
    validator: str, optional
        The name of the output validator package between "instructor", "llamaindex", "pydanticai" (Default is "instructor").
    validation : bool, optional
        Whether to validate the output against the schema, by default True
    max_retries : int, optional
        Maximum number of retries for the API call in case of failure, by default 3

    Returns
    -------
    Union[ListOfEntities,ChatCompletion]
        The response from the LLM model, either validated or raw output.
    """
    # Set response model and retries based on validation flag
    if validation:
        response_model = ListOfEntities
    else:
        response_model = None
        max_retries = 0

    result = None
    # Query the LLM client for annotation
    if validator == "instructor":
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
                        "content": f"{PROMPT}\nThe text to annotate:\n{text}",
                    },
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
        input_msg = ChatMessage.from_str(f"{PROMPT}\nThe text to annotate:\n{text}")
        try:
            response = client.chat([input_msg])
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
            response = agent.run_sync(f"{PROMPT}\nThe text to annotate:\n{text}")
            result = response.output
        except (
            PydanticValidationError,
            CoreValidationError,
            UnexpectedModelBehavior,
        ) as e:
            return str(e)

    return result


def run_annotations(
    text_to_annotate: str,
    model_name: str,
    client,
    validator: str = "instructor",
    num_iterations: int = 100,
    validation: bool = False,
) -> List[ListOfEntities | ChatCompletion]:
    """
    Runs multiple annotation attempts on the given text, with or without validation schema.

    Parameters
    ----------
    text_to_annotate : str
        The text to annotate.
    model_name : str
        The name of the LLM model to use.
    client : instructor.core.client.Instructor
        The LLM client to use (either OpenAI or OpenRouter).
    validator: str = "instructor"
        The name of the output validator package between "instructor", "llamaindex", "pydanticai" (Default is "instructor").
    num_iterations : int, optional
        Number of annotation attempts to run, by default 100
    validation : bool, optional
        Whether to use validation schema, by default False

    Returns:
    --------
    List[ChatCompletion | str]:
        List of the LLM response for the annotations of the same text with the same prompt.
    """
    list_of_responses = []

    desc = f"Running annotations {f'with {validator}' if validation else 'without'} validation schema..."
    logger.debug(f"{'🟢' if validation else '🔴'}{desc}")

    for _ in tqdm(
        range(num_iterations),
        desc=desc,
        colour="blue",
        ncols=200,
        unit="annotation",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        response = annotate(
            text_to_annotate, model_name, client, validator, validation=validation
        )
        list_of_responses.append(response)

    # logger.success(f"Completed {len(list_of_responses)} annotations successfully!\n")
    return list_of_responses


def run_annotation_format_validation(
    resp_not_validated: List[Union[ListOfEntities, ChatCompletion]],
) -> Tuple[float, list, list]:
    """
    Validate a list of annotation responses to ensure they are in proper JSON format.

    Parameters:
    -----------
    resp_not_validated (List[Union[ListOfEntities, ChatCompletion]]):
        List of annotation responses to validate.

    Returns:
    --------
    Tuple[float, list, list]:
        - prc_validation (float): Percentage of responses in valid format.
        - valid_resp (list): List of validated responses.
        - unvalid_resp (list): List of responses that failed validation.
    """
    # logger.debug("Validating annotations for JSON format...")
    valid_count = 0
    valid_resp = []
    unvalid_resp = []

    for resp in resp_not_validated:
        validated = validate_annotation_output_format(resp)
        if validated:
            valid_resp.append(validated)
            valid_count += 1
        else:
            unvalid_resp.append(resp)

    prc_validation = round((valid_count / len(resp_not_validated)) * 100, 1)
    logger.debug(
        f"{valid_count}/{len(resp_not_validated)} annotations ({prc_validation}%) are in valid JSON format."
    )
    return prc_validation, valid_resp, unvalid_resp


def run_annotation_halucination_validation(
    resp_format_valid: List[Union[ListOfEntities, ChatCompletion]],
) -> Tuple[float, list, list]:
    """
    Validate the content of annotation responses to ensure they match the expected text.

    This function checks each response in `resp_format_valid` to determine whether
    the annotation content is present in the target text `TEXT_TO_ANNOTATE`. It
    returns the percentage of valid annotations along with lists of valid and invalid responses.

    Parameters:
    -----------
    resp_format_valid (List[Union[ListOfEntities, ChatCompletion]]):
        List of annotation responses that have already been validated for format.

    Returns:
    --------
    Tuple[float, list, list]:
        - prc_validation (float): Percentage of responses with valid content.
        - valid_resp (list): List of responses with valid content.
        - unvalid_resp (list): List of responses with invalid content.
    """
    # logger.debug("Starting content validation of annotations...")
    valid_count = 0
    valid_resp = []
    unvalid_resp = []

    for resp in resp_format_valid:
        if is_annotation_in_text(resp, TEXT_TO_ANNOTATE):
            valid_resp.append(resp)
            valid_count += 1
        else:
            unvalid_resp.append(resp)

    prc_validation = round((valid_count / NB_ITERATIONS) * 100, 1)
    logger.debug(
        f"{valid_count}/{NB_ITERATIONS} annotations ({prc_validation}%) have no hallucinated entities."
    )
    return prc_validation, valid_resp, unvalid_resp


def save_annotation_records(
    model_name: str,
    text_to_annotate: str,
    resp_format_unvalid: List[str],
    resp_content_unvalid: List[str],
    resp_content_valid: List[str],
    folder_out_path: str,
    parquet_file_name: str,
) -> None:
    """
    Build a detailed annotation record dataset and save it as a Parquet file.

    This function consolidates all responses into a structured format with these fields:
    - model : name of the model used
    - response : the raw response text
    - is_format_valid : whether the response has a valid format
    - is_content_valid : whether the content passed hallucination/content validation

    Parameters
    ----------
    model_name : str
        Name of the model that generated the responses.
    text_to_annotate: str
        The input text that the instructor model will annotate.
    resp_format_unvalid : List[str]
        Responses that failed the format validation.
    resp_content_unvalid : List[str]
        Responses that passed the format validation but failed content validation.
    resp_content_valid : List[str]
        Responses that passed both format and content validation.
    folder_out_path : str
        The output folder path for the evaluation and annotation results.
    parquet_file_name : str
        File name where the detailed results should be saved in Parquet format.
    """
    # logger.debug("Building annotation records for saving...")
    records = []

    def serialize_response(resp):
        if isinstance(resp, str):
            return resp
        try:
            # Convert ChatCompletion to JSON string if needed
            return json.dumps(resp.__dict__, default=str)
        except Exception:
            return str(resp)

    # Add all format-invalid responses
    for r in resp_format_unvalid:
        records.append(
            {
                "model": model_name,
                "response": serialize_response(r),
                "is_format_valid": False,
                "is_content_valid": False,
            }
        )

    # Add format-valid but content-invalid responses
    for r in resp_content_unvalid:
        records.append(
            {
                "model": model_name,
                "response": serialize_response(r),
                "is_format_valid": True,
                "is_content_valid": False,
            }
        )

    # Add fully valid responses
    for r in resp_content_valid:
        records.append(
            {
                "model": model_name,
                "response": serialize_response(r),
                "is_format_valid": True,
                "is_content_valid": True,
            }
        )

    df = pd.DataFrame(records)
    path = os.path.join(folder_out_path, parquet_file_name.replace(".xlsx", ".parquet"))
    total_annotations = len(records)

    table = pa.Table.from_pandas(df)
    metadata = {
        "text_annotated": text_to_annotate.encode("utf-8"),
        "model": model_name.encode("utf-8"),
        "total_annotations": str(total_annotations).encode("utf-8"),
    }
    existing_metadata = table.schema.metadata or {}
    merged_metadata = {**existing_metadata, **metadata}
    table = table.replace_schema_metadata(merged_metadata)

    try:
        pq.write_table(table, path)
        logger.success(
            f"{total_annotations} annotation records saved into {path} successfully!\n"
        )
    except Exception as e:
        logger.error(f"Failed to save annotation records to {path}: {e}")


def evaluate_and_save_annotations(
    text_to_annotate: str,
    model_name: str,
    client,
    n_iterations: int,
    folder_out_path: str,
    parquet_file_name: str,
    validator: str = "instructor",
    validation: bool = False,
) -> Tuple[float, float]:
    """
    Run a complete annotation pipeline:
    1. Generate raw annotations with an instructor model.
    2. Validate the format of each generated annotation.
    3. Validate the content (hallucination detection) for responses with valid format.
    4. Save detailed results for each response into a Parquet file.

    Parameters
    ----------
    text_to_annotate : str
        The input text that the instructor model will annotate.
    model_name : str
        Name of the model used for annotations.
    client : object
        Client object used by run_annotations_with_{validator}().
    n_iterations : int
        Number of annotation attempts to generate.
    folder_out_path : str
        The output folder path for the evaluation and annotation results.
    parquet_file_name : str
        File name where the detailed results should be saved in Parquet format.
    validator: str, optional
        The name of the output validator package between "instructor", "llamaindex", "pydanticai" (Default is "instructor").
    validation : bool, optional
        Whether to apply validation during annotation generation, by default False.

    Returns
    -------
    pr_valid_format_resp_nv : float
        Percentage of responses with valid format.
    pr_valid_content_resp_nv : float
        Percentage of format-valid responses that also passed content validation.
    """
    # 1. Run annotations generation
    resp_not_validated = run_annotations(
        text_to_annotate,
        model_name,
        client,
        validator,
        n_iterations,
        validation=validation,
    )

    # 2. Validate annotations format
    pr_valid_format_resp_nv, resp_format_valid, resp_format_unvalid = (
        run_annotation_format_validation(resp_not_validated)
    )

    # 3. Validate annotations content
    pr_valid_content_resp_nv, resp_content_valid, resp_content_unvalid = (
        run_annotation_halucination_validation(resp_format_valid)
    )

    # 4. Save full annotations evaluation
    save_annotation_records(
        model_name,
        text_to_annotate,
        resp_format_unvalid,
        resp_content_unvalid,
        resp_content_valid,
        folder_out_path,
        parquet_file_name,
    )

    return pr_valid_format_resp_nv, pr_valid_content_resp_nv


def evaluate_json_annotations(folder_out_path: str, file_name: str) -> None:
    """
    Evaluate JSON annotation quality across several LLM models and validation strategies.

    Parameters:
    -----------
    folder_out_path : str
        The output folder path for the scraped data.
    file_name : str
        The output file name for the scraped data.
    """
    logger.info("Starting evaluation of JSON annotation outputs...")
    all_records = []

    # Ensure output folder exists
    Path(folder_out_path).mkdir(parents=True, exist_ok=True)

    # assign each models to instructor/llamaindex/pydanticai clients
    instructor_clients = assign_all_instructor_clients()
    llama_clients = assign_all_llamaindex_clients()
    py_clients = assign_all_pydanticai_clients()

    # for each models :
    for model_name in MODELS_OPENAI + MODELS_OPENROUTER:
        logger.info(
            f"======================== 🤖 Evaluating model: {model_name} ========================"
        )
        provider = "OpenAI" if model_name in MODELS_OPENAI else "OpenRouter"

        # ------------------------------------------------------
        # 1. Annotation without validation
        # ------------------------------------------------------
        full_resp_path_nv = f"{model_name.split('/')[-1]}_full_annotations_nv_{file_name.replace('.parquet', '.xlsx')}"
        pr_valid_format_resp_nv, pr_valid_content_resp_nv = (
            evaluate_and_save_annotations(
                TEXT_TO_ANNOTATE,
                model_name,
                instructor_clients[model_name],
                NB_ITERATIONS,
                folder_out_path,
                full_resp_path_nv,
                validation=False,
            )
        )

        # ------------------------------------------------------
        # 2. Annotation with INSTRUCTOR validation
        # ------------------------------------------------------
        full_resp_path_inst = f"{model_name.split('/')[-1]}_full_annotations_instructor_val_{file_name.replace('.parquet', '.xlsx')}"
        pr_valid_format_resp_instructor, pr_valid_content_resp_instructor = (
            evaluate_and_save_annotations(
                TEXT_TO_ANNOTATE,
                model_name,
                instructor_clients[model_name],
                NB_ITERATIONS,
                folder_out_path,
                full_resp_path_inst,
                validator="instructor",
                validation=True,
            )
        )

        # ------------------------------------------------------
        # 3. Annotation with LLAMAINDEX validation
        # ------------------------------------------------------
        full_resp_path_llama = f"{model_name.split('/')[-1]}_full_annotations_llamaindex_val_{file_name.replace('.parquet', '.xlsx')}"
        pr_valid_format_resp_llama, pr_valid_content_resp_llama = (
            evaluate_and_save_annotations(
                TEXT_TO_ANNOTATE,
                model_name,
                llama_clients[model_name],
                NB_ITERATIONS,
                folder_out_path,
                full_resp_path_llama,
                validator="llamaindex",
                validation=True,
            )
        )

        # ------------------------------------------------------
        # 4. Annotation with PYDANTICAI validation
        # ------------------------------------------------------
        full_resp_path_py = f"{model_name.split('/')[-1]}_full_annotations_pydanticai_val_{file_name.replace('.parquet', '.xlsx')}"
        pr_valid_format_resp_py, pr_valid_content_resp_py = (
            evaluate_and_save_annotations(
                TEXT_TO_ANNOTATE,
                model_name,
                py_clients[model_name],
                NB_ITERATIONS,
                folder_out_path,
                full_resp_path_py,
                validator="pydanticai",
                validation=True,
            )
        )

        all_records.append(
            {
                "Model (Provider)": f"{model_name} ({provider})",
                "JSON without format validation - Respect output format (%)": pr_valid_format_resp_nv,
                "JSON without format validation - No hallucination (%)": pr_valid_content_resp_nv,
                "JSON + Instructor - Respect output format (%)": pr_valid_format_resp_instructor,
                "JSON + Instructor - No hallucination (%)": pr_valid_content_resp_instructor,
                "JSON + LlamaIndex - Respect output format (%)": pr_valid_format_resp_llama,
                "JSON + LlamaIndex - No hallucination (%)": pr_valid_content_resp_llama,
                "JSON + PydanticAI - Respect output format (%)": pr_valid_format_resp_py,
                "JSON + PydanticAI - No hallucination (%)": pr_valid_content_resp_py,
            }
        )

    df_simple = pd.DataFrame(all_records)
    multi_columns = pd.MultiIndex.from_tuples(
        [
            ("JSON without format validation", "Respect output format (%)"),
            ("JSON without format validation", "No hallucination (%)"),
            ("JSON + Instructor", "Respect output format (%)"),
            ("JSON + Instructor", "No hallucination (%)"),
            ("JSON + LlamaIndex", "Respect output format (%)"),
            ("JSON + LlamaIndex", "No hallucination (%)"),
            ("JSON + PydanticAI", "Respect output format (%)"),
            ("JSON + PydanticAI", "No hallucination (%)"),
        ]
    )
    df_results = pd.DataFrame(
        df_simple.drop(columns=["Model (Provider)"]).values,
        columns=multi_columns,
        index=df_simple["Model (Provider)"],
    )
    path = os.path.join(folder_out_path, file_name)
    df_results.to_excel(path, index=True)
    logger.success(f"Evaluation stats saved to: {path} successfully!")


# MAIN PROGRAM
if __name__ == "__main__":
    # Parse arguments
    log, folder_out_path, file_name = parse_arguments()
    # Configure logging
    if log:
        log_folder = Path("logs")
        log_folder.mkdir(parents=True, exist_ok=True)
        logger.add(log_folder / "evaluate_json_annotations_{time:YYYY-MM-DD}.log")

    # Evaluate all models
    evaluate_json_annotations(folder_out_path, file_name)
