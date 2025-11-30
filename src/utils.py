"""Utility functions for the annotation JSON validation process."""

# Import necessary libraries
import json
import textwrap
from typing import Dict, Tuple, Union, List
from tqdm import tqdm
import unicodedata

import re
import instructor
from instructor.core import InstructorRetryException
from instructor.exceptions import ValidationError as InstructorValidationError
from pydantic import BaseModel, Field, ValidationError
from openai.types.chat import ChatCompletion
from llama_index.core.llms import ChatMessage


# CONSTANTS
PROMPT_JSON = """
You are given a scientific abstract or dataset description related to molecular dynamics simulations.
Your task is to identify and extract specific named entities relevant to simulation setup and analysis.
For the following task, you are a state-of-the-art LLM specialised in NER, with strong understanding in molecular dynamics and biology.
You need to simply extract entities, identify their type, and return them in a list of dictionnaries (one dictionnary for one entity, marked by its label and the text associated). Do not invent entities or mash up words. Do not merge, normalize, or reformat entities in any way—keep them exactly as written in the text.  
 

You are only allowed to use the entities below:
Entity Labels to Identify:
SOFTNAME: Software used in the simulation or analysis (e.g., Gromacs, AMBER, VMD, CHARMM-GUI)
SOFTVERS: Version number associated with the software (e.g., v. 2016.4, 5.0.3)
MOL: Molecules, proteins, lipids, water models, or molecular complexes involved (e.g., DPPC, water, GTP, KRas4B)
STIME: Duration of the simulation (e.g., 50 ns, 200ns, 5 µs)
TEMP: Temperature used in the simulation (e.g., 300 K, 288K, 358K)
FFM: Force fields used in the simulation (e.g., Charmm36, AMBER, MARTINI, TIP3P)

Avoid adding descriptors to the entities (e.g. "hydrated sodium cholide" should be annotated as "sodium chloride"), be precise, and avoid annotating large vague concepts as entities (e.g. "machine learning" or "molecular dynamics")
Also, pay close attention to letter casing: annotate the same word multiple times if it appears with different capitalization (e.g. "Sodium chloride" and "sodium chloride" must be treated as separate entities).

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
"""


PROMPT_POSITIONS = """
You are given a scientific abstract or dataset description related to molecular dynamics simulations.
Your task is to identify and extract specific named entities relevant to simulation setup and analysis.
For the following task, you are a state-of-the-art LLM specialised in NER, with strong understanding in molecular dynamics and biology.

Your job is to extract entities exactly as written in the text and return, for each entity:
- its label
- its text span (the exact substring, without modification)
- its character-level start index (inclusive)
- its character-level end index (exclusive)

The character positions must correspond to the exact indices in the input text.  
Do not alter, normalize, lowercase, uppercase, merge, or split entities.  
Do not invent entities or infer additional information.

Allowed Entity Labels:
- SOFTNAME : Software used in the simulation or analysis (e.g., Gromacs, AMBER)
- SOFTVERS : Version numbers (e.g., v.2016.4, 5.0.3)
- MOL      : Molecules, proteins, lipids, water models, complexes (e.g., DPPC, water, KRas4B)
- STIME    : Simulation durations (e.g., 50 ns, 5 µs)
- TEMP     : Simulation temperatures (e.g., 300 K)
- FFM      : Force fields (e.g., Charmm36, AMBER, MARTINI)

Important rules:
- Keep entities exactly as they appear (case-sensitive).
- Annotate repeated strings separately if they appear multiple times.
- Do not annotate vague concepts or long sentences.
- Only annotate valid entities from the allowed list.

Expected Output Format:
{"entities": [
  {"label": "MOL", "text": "cholesterol", "start": 123, "end": 134},
  {"label": "FFM", "text": "MARTINI", "start": 200, "end": 207}
]}

The output MUST be valid JSON.

Input text:
Short molecular dynamics of a peptide inside a pure DMPC membrane\n1 ns of molecular dynamics simulation of a 19-residue peptide inside a pure DMPC membrane, performed at the NPT ensemble - 1 atm at 310K using Berendsen (semi-isotropic) and V-rescale for pressure and temperature coupling, respectively, with 1.6 ps and 0.1ps as coupling constants. Van der Waals interactions were treated using the Verlet algorithm with a 1 nm cutoff. Electrostatics where treated with particle-mesh Ewald, with a 1 nm cutoff for the real space calculations. Both the peptide and DMPC were parameterized using the GROMOS 54A7, while SPC was used for water. LINCS restraints were used on all the bonds. Peptide sequence is AAAQAAQAQWAQRQATWQA. This sequence was not taken from any real world examples, as this is a test system created for MDAnalysis datasets. The peptide was set to have be -helical secondary structure, and was inserted manually in the membrane before minimization and equilibration. More on MDAnalysis: https://www.mdanalysis.org/

Output text: {"entities": [{"label": "MOL", "text": "DMPC", "start": 52, "end": 56},
        {"label": "STIME", "text": "1 ns", "start": 66, "end": 70},
        {"label": "MOL", "text": "DMPC", "start": 142, "end": 146},
        {"label": "TEMP", "text": "310K", "start": 198, "end": 202},
        {"label": "MOL", "text": "DMPC", "start": 563, "end": 567},
        {"label": "FFM", "text": "GROMOS 54A7", "start": 597, "end": 608},
        {"label": "FFM", "text": "SPC", "start": 616, "end": 619},
        {"label": "MOL", "text": "water", "start": 633, "end": 638},
        {"label": "SOFTNAME", "text": "LINCS", "start": 640, "end": 645},
        {"label": "MOL", "text": "AAAQAAQAQWAQRQATWQA", "start": 705, "end": 724}]}


Input text:
Unbiased simulation of Alanine Dipeptide in gas phase\n87 microsecond long unbiased Molecular Dynamics (MD) trajectory of alanine dipeptide in gas phase (traj comp.xtc). Temperature   300 K. Force Field: AMBER99SB-ILDN. There are 30+ back and forth transitions between the C 7eq and the C 7ax state. Simulations were performed using GROMACS 2021.4. The .tpr file is provided for reproduction. Details of the simulation parameters are accessible from the md.log file. The phi and psi torsion angles and the internal energy is printed in the COLVAR file at 2 ps interval. Alanine dipeptide is often used as a model system to test new simulation methods. We hope that sharing our long unbiased trajectory will help other research groups to compare the accuracy of the results obtained from new methods. This trajectory was generated as a part of our recent publication below. Please cite the following paper when using this trajectory: 1. Ray, Dhiman, Narjes Ansari, Valerio Rizzi, Michele Invernizzi, and Michele Parrinello. \"Rare event kinetics from adaptive bias enhanced sampling.\" Journal of Chemical Theory and Computation (2022). https://doi.org/10.1021/acs.jctc.2c00806

Output text: {"entities": [{"label": "MOL","text": "Alanine Dipeptide","start": 23,"end": 40},
        {"label": "STIME","text": "87 microsecond","start": 54,"end": 68},
        {"label": "MOL","text": "alanine dipeptide","start": 121,"end": 138},
        {"label": "TEMP","text": "300 K.","start": 183,"end": 189},
        {"label": "FFM","text": "AMBER99SB-ILDN","start": 203,"end": 217},
        {"label": "SOFTNAME","text": "GROMACS","start": 332,"end": 339},
        {"label": "SOFTVERS","text": "2021.4","start": 340,"end": 346},
        {"label": "SOFTNAME","text": "COLVAR","start": 539,"end": 545},
        {"label": "MOL","text": "Alanine dipeptide","start": 569,"end": 586}]}


Input text:
GROMOS 43A1-S3 POPE Simulations (versions 1 and 2) 313 K (NOTE: anisotropic pressure coupling)\nTwo GROMOS 43A1-S3 POPE bilayer simulations performed using GROMACS 4.0.7 for 200 ns with different starting velocities. Simulations were performed with the standard 43A1-S3 settings: a 1.0 nm cut-off with PME for the Coulombic interactions and a twin-range 1.0/1.6 nm cut-off for the van der Waals interactions. These simulations were performed at 313 K with a 128 lipid bilayer and used anisotropic pressure coupling. The full trajectories are provided bar the initial 100 ns. The starting structure was made through the conversion of an equilibrated GROMOS 43A1-S3 POPC membrane.

Output text: {"entities": [{"label": "FFM","text": "GROMOS 43A1-S3","start": 0,"end": 14},
        {"label": "MOL","text": "POPE","start": 15,"end": 19},
        {"label": "TEMP","text": "313 K","start": 51,"end": 56},
        {"label": "FFM","text": "GROMOS 43A1-S3","start": 99,"end": 113},
        {"label": "MOL","text": "POPE","start": 114,"end": 118},
        {"label": "SOFTNAME","text": "GROMACS","start": 155,"end": 162},
        {"label": "SOFTVERS","text": "4.0.7","start": 163,"end": 168},
        {"label": "STIME","text": "200 ns","start": 173,"end": 179},
        {"label": "TEMP","text": "313 K","start": 444,"end": 449},
        {"label": "STIME","text": "100 ns","start": 566,"end": 572},
        {"label": "FFM","text": "GROMOS 43A1-S3","start": 648,"end": 662},
        {"label": "MOL","text": "POPC","start": 663,"end": 667}]}
"""


# CLASSES
# Base entity
class Entity(BaseModel):
    """Base class for all extracted entities."""

    # ... is used to indicate that these fields are required
    label: str = Field(..., description="Short label identifying the entity type")
    text: str = Field(..., description="Extracted text content")


# Subclasses for each entity type of labels
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
        Union[
            Molecule,
            SimulationTime,
            ForceField,
            Temperature,
            SoftwareName,
            SoftwareVersion,
        ]
    ] = Field(..., description="List of recognized entities extracted from text")


# Base entity with positions
class EntityWithPosition(Entity):
    """Entity with text span positions."""
    start: int = Field(..., description="Start index of the entity in the text")
    end: int = Field(..., description="End index of the entity in the text")


# Subclasses for each entity type of labels
class MoleculePosition(EntityWithPosition):
    label: str = Field("MOL")


class SimulationTimePosition(EntityWithPosition):
    label: str = Field("STIME")


class ForceFieldPosition(EntityWithPosition):
    label: str = Field("FFM")


class TemperaturePosition(EntityWithPosition):
    label: str = Field("TEMP")


class SoftwareNamePosition(EntityWithPosition):
    label: str = Field("SOFTNAME")


class SoftwareVersionPosition(EntityWithPosition):
    label: str = Field("SOFTVERS")


# Container for the full response 
class ListOfEntitiesPositions(BaseModel):
    """Structured list of all extracted entities with character positions."""
    entities: List[
        Union[
            MoleculePosition,
            SimulationTimePosition,
            ForceFieldPosition,
            TemperaturePosition,
            SoftwareNamePosition,
            SoftwareVersionPosition
        ]
    ] = Field(..., description="List of entities with positions")


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
) -> List[Dict[int, str]]:
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


def report_hallucinated_entities(
    hallucinated_entities: Dict[int, List[str]], original_text: str
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
            f"⚠️ Hallucinated entities detected for {len(hallucinated_entities)} annotations:"
        )
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
            # print(f"Entity text '{entity.text}' not found in original text.")
            return False
    return True


def validate_annotation_output_format(
    response: Union[ChatCompletion, ListOfEntities],
) -> Union[ListOfEntities, None]:
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
    client : Union[instructor.core.client.Instructor | OpenAI]
        The LLM client to use (either Groq or OpenAI).
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

    try:
        result = None
        # Query the LLM client for annotation
        if validator == "instructor":
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

        elif validator == "llamaindex":
            input_msg = ChatMessage.from_str(f"{PROMPT}\nThe text to annotate:\n{text}")
            response = client.chat([input_msg])
            result = response.raw

        elif validator == "pydanticai":
            agent = Agent(
                model=model,
                output_type=ListOfEntities,
                system_prompt=("Extract entities as structured JSON."),
            )
            response = agent.run_sync(f"{PROMPT}\nThe text to annotate:\n{text}")
            result = response.output

        return result

    except InstructorRetryException as e:
        print(f"    ⚠️ Validated annotation failed after {e.n_attempts} attempts.")
        # print(f"Total usage: {e.total_usage.total_tokens} tokens")
        return str(e.last_completion)

    except InstructorValidationError as e:
        # print(e.errors)
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
                    {
                        "role": "system",
                        "content": "Extract entities as structured JSON.",
                    },
                    {
                        "role": "user",
                        "content": f"{PROMPT}\nThe text to annotate:\n{text}",
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


def run_annotation_stats(
    text_to_annotate: str,
    model_name: str,
    client,
    num_iterations: int = 100,
    validation: bool = False,
) -> Tuple[List[ListOfEntities | ChatCompletion], int, int]:
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
) -> Dict[str, Dict[str, Union[int, List[ListOfEntities | ChatCompletion]]]]:
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

    print("\n" + "=" * 80)
    print(f"📝 Input text:\n{textwrap.fill(text_to_annotate, width=120)}")
    print("=" * 80 + "\n")
    print("\n" + "=" * 80)
    print(f"📊 Summary after {num_iterations} runs with {model_name}:")
    print("=" * 80 + "\n")
    print(
        f"❌ Without validation schema : {valid_wo / num_iterations * 100:.1f}% valid responses"
    )
    print(
        f"✅ With validation schema    : {valid_w / num_iterations * 100:.1f}% valid responses"
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


def visualize_entities(
    text: str, entities: Union[ChatCompletion, ListOfEntities, str]
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
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                ents.append(
                    {"start": match.start(), "end": match.end(), "label": ent["label"]}
                )
    # If entities is a ListOfEntities
    elif hasattr(entities, "entities"):  # or isinstance(entities, ListOfEntities)
        entity_list = entities.entities
        for ent in entity_list:
            pattern = re.escape(ent.text)
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                ents.append(
                    {"start": match.start(), "end": match.end(), "label": ent.label}
                )
    # If entities is a string (raw JSON)
    elif isinstance(entities, str):
        content = json.loads(entities)
        entity_list = content["entities"]
        for ent in entity_list:
            pattern = re.escape(ent["text"])
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                ents.append(
                    {"start": match.start(), "end": match.end(), "label": ent["label"]}
                )

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

    with open(file_path, "r", encoding="utf-8") as file:
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

    Returns
    -------
    None
        The function updates and rewrites the JSON file.
    """
    # Load the annotation file
    file_path = f"../data/formated_annotations/{file_name}"
    with open(file_path, "r", encoding="utf-8") as file:
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


def read_parquet_summary(parquet_path: str) -> dict:
    """
    Read and return summary metadata stored in a Parquet file
    created by save_annotation_records().

    Parameters
    ----------
    parquet_path : str
        Path to the Parquet file.

    Returns
    -------
    dict
        A dictionary containing extracted metadata fields
        (summary, model, total_annotations, and any other stored metadata).
    """

    try:
        metadata = pq.read_metadata(parquet_path).metadata

        if metadata is None:
            return {"error": "No metadata found in this Parquet file."}

        # Decode UTF-8 encoded metadata
        decoded = {
            key.decode("utf-8"): metadata[key].decode("utf-8") for key in metadata
        }
        print(f"Annotation results from {parquet_path}:")
        print(f"Total annotations: {decoded['total_annotations']}")
        print(f"LLM Model: {decoded['model']}")
        print(f"Text annotated: {decoded['text_annotated']}")
        return decoded

    except Exception as e:
        return {"error": f"Failed to read metadata: {e}"}



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
        range(NB_ITERATIONS),
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


def evaluate_and_save_annotations(
    text_to_annotate: str,
    model_name: str,
    client,
    out_path : str,
    validator: str = "instructor",
    validation: bool = False,
) -> Tuple[float, float, float]:
    """
    Run a complete annotation pipeline:
    1. Generate raw annotations with an LLM.
    2. Validate the format of each generated annotation.
    3. Validate the content (hallucination detection) for responses with valid format.
    4. Validate the content (correct annotations like groundtruth) for responses with no hallucinated entities.
    5. Save detailed results for each response into a Parquet

    Parameters
    ----------
    text_to_annotate : str
        The input text that the instructor model will annotate.
    model_name : str
        Name of the model used for annotations.
    client : object
        Client object used by run_annotations_with_{validator}().
    out_path : str
        The full output path for the evaluation and annotation results.
    validator: str, optional
        The name of the output validator package between "instructor", "llamaindex", "pydanticai" (Default is "instructor").
    validation : bool, optional
        Whether to apply validation during annotation generation, by default False.

    Returns
    -------
    pr_valid_format_resp : float
        Percentage of responses with valid format.
    pr_valid_content_resp : float
        Percentage of format-valid responses that also passed content validation.
    pr_correct_answers : float
        Percentage of responses that is correct, cad que la réponse est la même que le groundtruth.
    """
    # 1. Run annotations generation
    response = run_annotations(
        text_to_annotate,
        model_name,
        client,
        validator,
        validation=validation
    )

    # 2. Validate annotations format
    pr_valid_format_resp, resp_format_valid, resp_format_unvalid = (
        run_annotation_format_validation(response)
    )

    # 3. Validate annotations content
    pr_valid_content_resp, resp_content_valid, resp_content_unvalid = (
        run_annotation_halucination_validation(resp_format_valid)
    )

     # 4. Validate annotations content (correct like grountruth)
    pr_correct_answers, resp_correct_answers, resp_uncorrect_answers = (
        run_annotation_groundtruth_validation(resp_content_valid)
    )

    # 5. Save full annotations evaluation
    save_annotation_records(
        model_name,
        text_to_annotate,
        resp_format_unvalid,
        resp_content_unvalid,
        resp_correct_answers,
        resp_uncorrect_answers,
        out_path
    )

    return pr_valid_format_resp, pr_valid_content_resp, pr_correct_answers


def save_annotation_records(
    model_name: str,
    text_to_annotate: str,
    resp_format_unvalid: List[str],
    resp_content_unvalid: List[str],
    resp_correct_answers: List[str],
    resp_uncorrect_answers: List[str],
    out_path : str,
) -> None:
    """
    Build a detailed annotation record dataset and save it as a Parquet file.

    This function consolidates all responses into a structured format with these fields:
    - model : name of the model used
    - response : the raw response text
    - is_format_valid : whether the response has a valid format
    - is_content_valid : whether the content passed hallucination/content validation
    - is_correct : whether the response is the same as the annoation groundtruth

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
    resp_correct_answers : List[str]
        Responses that passed format, content and groundtruth validation.
    resp_uncorrect_answers : List[str]
        Responses that passed the content validation but failed groundtruth validation.
    out_path : str
        The output path for the evaluation and annotation results.
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
    
    def add_records(responses, is_format_valid, is_content_valid, is_correct):
        for r in responses:
            records.append({
                "model": model_name,
                "text_to_annotate": text_to_annotate,
                "response": serialize_response(r),
                "is_format_valid": is_format_valid,
                "is_content_valid": is_content_valid,
                "is_correct": is_correct,
            })

    # Add all format-invalid responses
    add_records(resp_format_unvalid, False, False, False)

    # Add format-valid but content-invalid responses
    add_records(resp_content_unvalid, True, False, False)

    # Add format-valid, content valid but incorrect responses
    add_records(resp_uncorrect_answers, True, True, False)

    # Add fully valid responses
    add_records(resp_correct_answers, True, True, True)

    df = pd.DataFrame(records)
    path = Path(str(out_path).replace(".xlsx", ".parquet"))
    total_annotations = len(records)
    try:
        df.to_parquet(path, index=False)
        logger.success(f"{total_annotations} annotation records saved into {out_path} successfully!\n")
    except Exception as e:
        logger.error(f"Failed to save annotation records to {out_path}: {e}")



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


def is_same_as_groundtruth(resp, groundtruth):
    """
    Strict comparison between the entities in a model response
    and the entities in the groundtruth.

    Rules:
    - response must contain a key "entities"
    - entities must match exactly:
        * same number of entities
        * no missing or extra entities
        * order does not matter
        * each entity must be identical in all fields
        * extracted text must be strictly equal
    """

    # --- Convert resp to dict if it's a JSON string ---
    if isinstance(resp, str):
        try:
            resp = json.loads(resp)
        except Exception:
            return False

    # --- Convert groundtruth to dict if it's a JSON string ---
    if isinstance(groundtruth, str):
        try:
            groundtruth = json.loads(groundtruth)
        except Exception:
            return False

    # --- Entities must exist ---
    if not isinstance(resp, dict) or "entities" not in resp:
        return False
    if not isinstance(groundtruth, dict) or "entities" not in groundtruth:
        return False

    resp_entities = resp["entities"]
    gt_entities = groundtruth["entities"]

    # --- Both must be lists ---
    if not isinstance(resp_entities, list) or not isinstance(gt_entities, list):
        return False

    # --- Same number of entities ---
    if len(resp_entities) != len(gt_entities):
        return False

    # --- Strict comparison ignoring order ---
    # We sort entities by all their fields to allow order-independent comparison
    try:
        resp_sorted = sorted(resp_entities, key=lambda x: json.dumps(x, sort_keys=True))
        gt_sorted = sorted(gt_entities, key=lambda x: json.dumps(x, sort_keys=True))
    except Exception:
        return False

    # --- Compare each entity strictly ---
    for r, g in zip(resp_sorted, gt_sorted):
        # must have same keys
        if set(r.keys()) != set(g.keys()):
            return False
        # must have same values, including exact extracted text
        for k in r:
            if r[k] != g[k]:
                return False

    return True


def run_annotation_groundtruth_validation(
    resp_content_valid: List[Union[ListOfEntities, ChatCompletion]]
) -> Tuple[float, list, list]:
    """
    Validate responses that passed hallucination/content checks by comparing
    them to the annotation groundtruth.

    Parameters
    ----------
    resp_content_valid : List[Union[ListOfEntities, ChatCompletion]]
        List of responses that passed format and hallucination validation.

    Returns
    -------
    Tuple[float, list, list]
        - pr_correct (float): Percentage of responses matching the groundtruth.
        - correct_resp (list): Responses that match the groundtruth.
        - incorrect_resp (list): Responses valid but not equal to the groundtruth.
    """

    correct_resp = []
    incorrect_resp = []
    correct_count = 0

    for response in resp_content_valid:
        if isinstance(response, ListOfEntities):
            resp_json = {"entities": [{"label": e.label, "text": e.text} for e in response.entities]}
        if isinstance(response, ChatCompletion):
            # Extract the content from the ChatCompletion response
            resp_str = response.choices[0].message.content
            try:
                resp_json = json.loads(resp_str)
            except json.JSONDecodeError:
                logger.warning(f"We : {resp_str}")
                resp_json = {}
        
        print(f"resp : {resp_json}")
        print(f"GROUNDTRUTH : {GROUNDTRUTH_JSON}")

        if is_same_as_groundtruth(resp_json, GROUNDTRUTH_JSON):
            correct_resp.append(response)
            correct_count += 1
        else:
            incorrect_resp.append(response)

    # Avoid division by zero
    total = len(resp_content_valid)
    pr_correct = round((correct_count / total * 100), 1) if total > 0 else 0.0

    logger.debug(
        f"{correct_count}/{total} valid annotations ({pr_correct}%) match the groundtruth."
    )

    return pr_correct, correct_resp, incorrect_resp
