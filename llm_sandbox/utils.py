"""Utility functions for the annotation JSON validation process."""

# Import necessary libraries
import json
import textwrap
from typing import Dict, Tuple, Union, List
from tqdm import tqdm
import unicodedata

import re
from spacy import displacy
import instructor
from instructor.exceptions import ValidationError as InstructorValidationError, InstructorRetryException
from pydantic import BaseModel, Field, ValidationError
from openai.types.chat import ChatCompletion


# CONSTANTS
PROMPT = """
You are given a scientific abstract or dataset description related to molecular dynamics simulations.
Your task is to identify and extract specific named entities relevant to simulation setup and analysis.
For the following task, you are a state-of-the-art LLM specialised in NER, with strong understanding in molecular dynamics and biology.
You need to simply extract entities, identify their type, and return them in a list of dictionnaries (one dictionnary for one entity, marked by its label and the text associated). Do not invent entities or mash up words. Use only what is in the original text. You are only allowed to use the entities below:

Entity Labels to Identify:
SOFTNAME: Software used in the simulation or analysis (e.g., Gromacs, AMBER, VMD, CHARMM-GUI)
SOFTVERS: Version number associated with the software (e.g., v. 2016.4, 5.0.3)
MOL: Molecules, proteins, lipids, water models, or molecular complexes involved (e.g., DPPC, water, GTP, KRas4B)
STIME: Duration of the simulation (e.g., 50 ns, 200ns, 5 µs)
TEMP: Temperature used in the simulation (e.g., 300 K, 288K, 358K)
FFM: Force fields used in the simulation (e.g., Charmm36, AMBER, MARTINI, TIP3P)

Avoid adding descriptors to the entities (e.g. "hydrated sodium cholide" should be annotated as "sodium chloride"), be precise, and avoid annotating large vague concepts as entities (e.g. "machine learning" or "molecular dynamics")

Expected Output Format:
{"response": [{"label": "MOL", "text": "cholesterol"}, {"label": "FFM", "text": "MARTINI"}, {"label": "MOL", "text": "POPC"}, {"label": "MOL", "text": "glycerol"}, {"label": "SOFTNAME", "text": "Gromacs"}]}


Input text:
POPC Ulmschneider OPLS Verlet Group\nMD simulation trajectory and related files for fully hydrated POPC bilayer run with Verlet and Group schemes. The Ulmschneider force field for POPC was used with Gromacs 5.0.3 [1,2]. Conditions: T 298.15, 128 POPC molecules, 5120 tip3p waters. 200ns trajectory (preceded by 5ns NPT equillibration). Starting structure was obtained from CHARMM-GUI [3]. This data is ran for the nmrlipids.blospot.fi project. More details from nmrlipids.blospot.fi and https://removed [1] J.P. Ulmschneider & M.B. Ulmschneider, United Atom Lipid Parameters for Combination with the Optimized Potentials for Liquid Simulations All-Atom Force Field, JCTC 2009, 5 (7), 1803 1813 [2] https://removed [3] https://removed

Output text:
{"entities" : [ {"label": "MOL", "text": "waters"}, {"label": "FFM", "text": "OPLS"}, {"label": "MOL", "text": "POPC"}, {"label": "SOFTNAME", "text": "Gromacs"}, {"label": "SOFTNAME", "text": "CHARMM-GUI"}, {"label": "SOFTVERS", "text": "5.0.3"}, {"label": "TEMP", "text": "298.15"}, {"label": "STIME", "text": "200ns"}, {"label": "STIME", "text": "5ns"}]}

Input text:
Interaction of the inhibitory peptides ShK and HmK with the voltage-gated potassium channel KV1.3: Role of conformational dynamics\nABSTRACT: Peptide toxins that adopt the ShK fold can inhibit the voltage-gated potassium channel KV1.3 with IC50 values in the pM range, and are therefore potential leads for drugs targeting autoimmune and neuroinflammatory diseases. NMR relaxation measurements and pressure-dependent NMR have shown that, despite being cross-linked by disulfide bonds, ShK itself is flexible in solution. This flexibility affects the local structure around the pharmacophore for KV1.3 channel blockade and, in particular, the relative orientation of the key Lys and Tyr side chains (Lys22 and Tyr23 in ShK), and has implications for the design of KV1.3 inhibitors. In this study, we have performed molecular dynamics (MD) simulations on ShK and a close homolog, HmK, in order to probe the conformational space occupied by the Lys and Tyr residues, and docked the different conformations with a recently determined cryo-EM structure of the KV1.3 channel. Although ShK and HmK have 60% sequence identity, their dynamic behaviors are quite different, with ShK sampling a broad range of conformations over the course of a 5 μs MD simulation, while HmK is relatively rigid. We also investigated the importance of conformational dynamics, in particular the distance between the side chains of the key dyad Lys22 and Tyr23, for binding to KV1.3. Although these peptides have quite different dynamics, the dyad in both adopts a similar configuration upon binding, revealing a conformational selection upon binding to KV1.3 in the case of ShK . Intriguingly, the more flexible peptide, ShK , binds with nearly 300-fold higher affinity than HmK .

Output text:
{"entities" : [{"label": "MOL", "text": "ShK"}, {"label": "STIME", "text": "5 μs"}, {"label": "MOL", "text": "HmK"}, {"label": "MOL", "text": "KV1.3"}]}


Input text:
Assessment of mutation probabilities of KRAS G12 missense mutants and their long-time scale dynamics by atomistic molecular simulations and Markov state modeling: Datasets.\nDatasets related to the publication [1]. Including: KRAS G12X mutations derived from COSMIC v.79 [http://cancer.sanger.ac.uk/cosmic/] (KRAS G12X mut COSMICv79..xlsx) RMSFs (300-2000ns) of GDP-systems (300 2000rmsf GDP systems RAW AVG SE.xlsx) RMSFs (300-2000ns) of GTP-systems (300 2000RMSF GTP systems RAW AVG SE.xlsx) PyInteraph analysis data for salt-bridges and hydrophobic clusters (.dat files for each system in the PyInteraph data.zip-file) Backbone trajectories for each system (residues 4-164; frames for every 1ns). Last number (e.g. 1) refers to the replica of the simulated system. backbone 4-164.gro/.pdb/.tpr -files (resid 4-164) [1] Pantsar T et al. Assessment of mutation probabilities of KRAS G12 missense mutants and their long-time scale dynamics by atomistic molecular simulations and Markov state modeling. PLoS Comput Biol Submitted (2018)

Output text:
{"entities" : [ {"label": "MOL", "text": "KRAS G12"}, {"label": "STIME", "text": "300-2000ns"}, {"label": "SOFTNAME", "text": "GROMACS"}, {"label": "MOL", "text": "KRAS G12X"}, {"label": "MOL", "text": "GDP"}, {"label": "SOFTNAME", "text": "NAMD"}, {"label": "SOFTNAME", "text": "PyInteraph"}, {"label": "MOL", "text": "GTP"}, {"label": "STIME", "text": "1ns"}]}


Input text:
Automated protein-protein structure prediction of the T cell receptor-peptide major histocompatibility complex\nThe T Cell Receptor (TCR) recognition of a peptide-major histocompatibility complex (pMHC) is a crucial component of the adaptive immune response. The identification of TCR-pMHC pairs is a significant bottleneck in the implementation of TCR immunotherapies and may be augmented by computational methodologies that accelerate the rate of TCR discovery. The ability to computationally design TCRs to a target pMHC will require an automated integration of next-generation sequencing, homology modeling, molecular dynamics (MD), and TCR ranking. We present a generic pipeline to evaluate patient-specific, sequence-based TCRs to a target pMHC. The most expressed TCRs from 16 colorectal cancer patients are homology modeled to target the CEA peptide using Modeller and ColabFold. Then, these TCR-pMHC structures are compared by performing an automated molecular dynamics equilibration. We find that Colabfold generates starting configurations that require, on average, an ~2.5X reduction in simulation time to equilibrate TCR-pMHC structures compared to Modeller. In addition, there are differences between equilibrated structures generated by Modeller and ColabFold. Moreover, we identify TCR ranking criteria that may be used to prioritize TCRs for evaluation of in vitro immunogenicity.

Output text:
{"entities": [{"label": "MOL", "text": "T cell receptor"}, {"label": "MOL", "text": "T Cell Receptor"}, {"label": "MOL", "text": "TCR"}, {"label": "MOL", "text": "peptide-major histocompatibility complex"}, {"label": "MOL", "text": "pMHC"}, {"label": "MOL", "text": "TCRs"},{"label": "SOFTNAME", "text": "Modeller"}, {"label": "SOFTNAME", "text": "ColabFold"}]}


Input text:
Molecular dynamics simulations of lipid bilayers containing POPC and POPS (various mixtures) with ECC-lipids force field, and Na+ (K+) counterions\nClassical molecular dynamics simulations of various mixtures of POPC:POPS lipid bilayers in water solution with only Na+ counterions (or with K+ counterions when noted with " KCl" suffix). ECC-lipids force field parameters used for lipids, SPC/E water model and ECC-ions, all parameters available at https://removed lipids simulations performed with Gromacs 2018.0 (*.xtc files) simulation length 1000 ns 1 microsecond temperature 298 K Gromacs simulation setting is in the file npt lipid bilayer.mdp

Output text:
{ "entities" : [{"label": "MOL", "text": "POPC"}, {"label": "MOL", "text": "POPS"}, {"label": "TEMP", "text": "298 K"}, {"label": "MOL", "text": "Na+"}, {"label": "MOL", "text": "water"}, {"label": "SOFTVERS", "text": "2018.0"}, {"label": "FFM", "text": "ECC-lipids"}, {"label": "FFM", "text": "SPC/E"}, {"label": "STIME", "text": "1000 ns"}, {"label": "STIME", "text": "1 microsecond"}, {"label": "MOL", "text": "K+"}, {"label": "FFM", "text": "ECC-ions"}, {"label": "SOFTNAME", "text": "Gromacs"}]}
"""


# CLASSES
# Base entity
class Entity(BaseModel):
    """Base class for all extracted entities."""
    # ... is used to indicate that these fields are required
    label: str = Field(..., description="Short label identifying the entity type")
    text: str = Field(..., description="Extracted text content")


# Subcklasses for each entity type of labels
class Molecule(Entity):
    label: str = Field("MOL", description="Label for molecule entities")


class SimulationTime(Entity):
    label: str = Field("STIME", description="Label for simulation time entities")


class ForceField(Entity):
    label: str = Field("FFM", description="Label for force field entities")


class Temperature(Entity):
    label: str = Field("TEMP", description="Label for temperature entities")


class SoftwareName(Entity):
    label: str = Field("SOFTNAME", description="Label for software name entities")


class SoftwareVersion(Entity):
    label: str = Field("SOFTVERS", description="Label for software version entities")


# Container for the full response
class ListOfEntities(BaseModel):
    """Structured list of all extracted entities."""
    entities: List[
        Union[Molecule, SimulationTime, ForceField, Temperature, SoftwareName, SoftwareVersion]
    ] = Field(..., description="List of recognized entities extracted from text")


# FUNCTIONS
def check_json_validity(json_string: str) -> None:
    """Check if the given string is a valid JSON.
    
    Parameters
    ----------
    json_string : str
        The string to check.
    """
    try:
        json.loads(json_string)
        #print("Valid JSON ✅")
        return True
    except json.JSONDecodeError:
        #print("Invalid JSON ❌")
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


def find_hallucinated_entities(annotations: ListOfEntities, original_text: str) -> List[Dict[int, str]]:
    """Identify entities in the annotation output that are not present in the original text.

    Parameters
    ----------
    response : ListOfEntities
        The annotation output to check.
    original_text : str
        The original text to compare against.

    Returns
    -------
    List[Dict[int, str]]
        A list of dictionaries containing the index and text of hallucinated entities.
    """
    hallucinated_entities = []
    # Normalize the original text
    text_normalized = normalize_text(original_text)
    for i, annotation in enumerate(annotations): 
        if annotation is None:
            continue
        # Initialize a temporary list to store hallucinated entities for this annotation
        temp_list = []
        # Check each entity in the annotation
        for entity in annotation.entities:
            if entity.text.lower() not in text_normalized:
                temp_list.append(entity.text)
        # If there are hallucinated entities, add them to the result list
        if temp_list != []:
            hallucinated_entities.append({i: temp_list})
    return hallucinated_entities


def report_hallucinated_entities(hallucinated_entities: Dict[int, List[str]], original_text: str) -> None:
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
        print(f"⚠️ Hallucinated entities detected for {len(hallucinated_entities)} annotations:")
        print("=" * 80)
        # Get each dictionary representing hallucinated entities
        for hallucinated_entities in hallucinated_entities:
            # Get the list of hallucinated texts from each dictionary
            for index, texts in hallucinated_entities.items():
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
        True if the entities in the response are present in the original text, False otherwise.
    """
    # Normalize the original text
    text_normalized = normalize_text(original_text)

    # Check if all entity texts are present in the original text
    for entity in response.entities:
        if entity.text.lower() not in text_normalized:
            #print(f"Entity text '{entity.text}' not found in original text.")
            return False
    return True


def validate_annotation_output_format(response: Union[ChatCompletion, ListOfEntities]) -> Union[ListOfEntities, None]:
    """Validate the annotation output against the ListOfEntities schema.
    
    Parameters
    ----------
    response : Union[ChatCompletion, ListOfEntities]
        The response from the LLM model to validate.

    Returns
    -------
    bool
        True if the response is valid according to the ListOfEntities schema, False otherwise.
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
        except ValidationError as e:
            return None


def annotate(
    text: str,
    model: str,
    client: instructor.core.client.Instructor,
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
    client : Union[instructor.core.client.Instructor | OpenAI]
        The LLM client to use (either Groq or OpenAI).
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
    
    try:
        # Query the LLM client for annotation
        result = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Extract entities as structured JSON."},
                {"role": "user", "content": f"{PROMPT}\nThe text to annotate:\n{text}"}
            ],
            response_model=response_model,
            max_retries=max_retries
        )
        return result
    
    except InstructorRetryException as e:
        print(f"    ⚠️ Validated annotation failed after {e.n_attempts} attempts.")
        #print(f"Total usage: {e.total_usage.total_tokens} tokens")
        return str(e.last_completion)

    except InstructorValidationError as e:
        #print(e.errors)
        return str(e.raw_output)


def annotate_forced_validation(
    text: str,
    model: str,
    client: instructor.core.client.Instructor,
    validation: bool = True,
    max_retries: int = 10,
) -> dict:    
    """Annotate the given text using the specified model.
    If validation is True, the output will be validated against the GlobalResponse schema.

    Parameters
    ----------
    text : str
        The text to annotate.
    model : str
        The name of the LLM model to use.
    client : Union[instructor.core.client.Instructor | OpenAI]
        The LLM client to use (either Groq or OpenAI).
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
                    {"role": "system", "content": "Extract entities as structured JSON."},
                    {"role": "user", "content": f"{PROMPT}\nThe text to annotate:\n{text}"}
                ],
                response_model=ListOfEntities if validation else None,
                max_retries=3 if validation else 0
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


def run_annotation_stats(
    text_to_annotate: str,
    model_name: str,
    client,
    num_iterations: int = 100,
    validation: bool = False,
) -> Tuple[List[ListOfEntities|ChatCompletion], int, int]:
    """
    Runs multiple annotation attempts on the given text, with or without validation schema.
    Returns the number of valid and invalid responses.

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
    """
    valid_count = 0
    invalid_count = 0
    list_of_entities = []

    desc = f"Running annotations {'with' if validation else 'without'} validation schema"
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
) -> Dict[str, Dict[str, Union[int, List[ListOfEntities|ChatCompletion]]]]:
    """
    Wrapper that runs annotation tests both with and without validation schema,
    and prints a summary.

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
        A dictionary summarizing the results with and without validation schema. As follows:
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

    print("\n" + "="*80)
    print(f"📝 Input text:\n{textwrap.fill(text_to_annotate, width=120)}")
    print("=" * 80 + "\n")
    print("\n" + "="*80)
    print(f"📊 Summary after {num_iterations} runs with {model_name}:")
    print("=" * 80 + "\n")
    print(f"❌ Without validation schema : {valid_wo / num_iterations * 100:.1f}% valid responses")
    print(f"✅ With validation schema    : {valid_w / num_iterations * 100:.1f}% valid responses")

    return {
        "without_validation": {"valid": valid_wo, "invalid": invalid_wo, "examples": list_of_entities_wo},
        "with_validation": {"valid": valid_w, "invalid": invalid_w, "examples": list_of_entities_w},
    }


def visualize_entities(text: str, entities: Union[ChatCompletion, ListOfEntities, str]) -> None:
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
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                ents.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": ent["label"]
                })
    # If entities is a ListOfEntities 
    elif hasattr(entities, "entities"): # or isinstance(entities, ListOfEntities)
        entity_list = entities.entities
        for ent in entity_list:
            pattern = re.escape(ent.text)
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                ents.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": ent.label
                })
    # If entities is a string (raw JSON)
    elif isinstance(entities, str):
        content = json.loads(entities)
        entity_list = content["entities"]
        for ent in entity_list:
            pattern = re.escape(ent["text"])
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                ents.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": ent["label"]
                })
    
    # Prepare the data for displacy
    spacy_format = {"text": text, "ents": ents}
    displacy.render(spacy_format, style="ent", manual=True, options=options)