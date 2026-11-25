"""Classify molecules entities (`MOL`) of JSON annotations."""

import json
import os
import re
import time
import urllib.parse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

import click
import httpx
import instructor
import loguru
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI

from mdner_llm.annotations.build_entity_inventory import collect_entities
from mdner_llm.common import clean_text
from mdner_llm.logger import create_logger
from mdner_llm.models.molecule_category import Category, MoleculeCategory

load_dotenv()

# All category names, in the sort order of the database.
CATEGORIES = get_args(Category)
# One client for all the requests: the connections are reused (faster).
HTTP_CLIENT = httpx.Client(timeout=60, follow_redirects=True)
# URLs of the APIs of the databases used to classify molecules.
API_OLS = "https://www.ebi.ac.uk/ols4/api"
API_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
API_UNIPROT = "https://rest.uniprot.org/uniprotkb"
API_PDB = "https://data.rcsb.org/rest/v1/core/entry"
# Web page of a database entry, filled with its ID.
DATABASE_ENTRY_URLS = {
    "RCSB PDB": "https://www.rcsb.org/structure/{}",
    "UniProt": "https://www.uniprot.org/uniprotkb/{}/entry",
    "ChEBI": "https://www.ebi.ac.uk/chebi/{}",
    "PubChem": "https://pubchem.ncbi.nlm.nih.gov/compound/{}",
}
# Patterns on the name, the definition and the ancestors of a ChEBI term
# Lipids come before ions: ChEBI files many phospholipids under their
# cationic form (e.g. DPPC is a "quaternary ammonium ion").
CHEBI_PATTERNS = {
    "Nucleic acid": re.compile(r"nucleic acid|oligonucleotide|\b[dr]na\b"),
    "Protein": re.compile(
        # Not followed by "inhibitor": a "protein kinase inhibitor" is a drug.
        r"\b(?:protein|(?:poly|oligo|di)?peptide)s?\b(?! (?:\w+ )?inhibitor)"
        # Any single word ending in -ase (peroxidase), but not "protease inhibitor".
        r"|^[a-z]+ase$"
    ),
    "Sugar": re.compile(
        # "derivative" is excluded: ATP is a "monosaccharide derivative".
        r"\b(?:mono|di|oligo|poly)saccharides?\b(?! derivative)|^carbohydrates?$"
        # Any single word ending in -ose (aldose, pyranose), but not "udp-d-glucose".
        r"|^[a-z]+ose$"
    ),
    "Lipid": re.compile(
        r"lipid|fatty acid|glycerophospho|phosphocholine|steroid|sterol"
        r"|sphingo|ceramide|glyceride"
    ),
    "Material, polymer": re.compile(r"polymer|macromolecule|elemental carbon|nanotube"),
    "Ion": re.compile(
        # All the ions of ChEBI have the ancestor "ion".
        r"^ion$|\bsalt$"
        # Neutral atoms, for ions written without their charge (Na, Ca, Cl).
        r"|alkali metal atom|alkaline earth metal atom|^halogen$"
    ),
}
# Regexes to detect identifiers and sequences, case ignored.
PDB_REGEX = re.compile(r"^[1-9][a-z0-9]{3}$", flags=re.IGNORECASE)
UNIPROT_REGEX = re.compile(
    r"[opq][0-9][a-z0-9]{3}[0-9]|[a-nr-z][0-9]([a-z][a-z0-9]{2}[0-9]){1,2}",
    flags=re.IGNORECASE,
)
SEQUENCE_REGEXES = [
    re.compile(r"^5'-[acgtu]+-3'$", flags=re.IGNORECASE),  # DNA / RNA
    re.compile(
        r"^(?![ACGTU]+$)[ACDEFGHIKLMNPQRSTVWY]{9,}(?:-NH2)?$"
    ),  # peptide (9+ amino acids, optional amidation (e.g. RKIRRKFKK-NH2)
    re.compile(r"^[\[(][A-Za-z]{2,}[\])]\d+$"),  # repeated motif, e.g. (GVPGV)7
]


def group_mol_entities(
    df_entities: pd.DataFrame,
    database: dict,
    logger: "loguru.Logger" = loguru.logger,
) -> pd.DataFrame:
    """Group the MOL entities of `collect_entities` by normalized name.

    Returns
    -------
    pd.DataFrame
        Indexed by normalized name (spelling of the database, or else the most
        frequent one), with the columns: nb_mentions and json_files (sorted names
        of the annotation files mentioning the entity).
    """
    molecules = df_entities[df_entities["category"] == "MOL"].copy()
    # The case is ignored for long names only: "Cholesterol" is "cholesterol",
    # but "AIR" is not "air".
    molecules["group"] = [
        name if len(name) <= 5 else name.lower()
        for name in molecules["entity_normalized"]
    ]
    # Long names of the database, in lower case -> as written in the database.
    # Example: {"alanine dipeptide": "Alanine Dipeptide"}
    db_names = {name.lower(): name for name in database if len(name) > 5}
    rows = []
    # Each group contains all the mentions of one entity.
    for group, mentions in molecules.groupby("group"):
        # Spellings of the group: "cholesterol" (27 mentions), "Cholesterol" (2).
        spellings = mentions["entity_normalized"]
        # Keep the most frequent for the searches: "cholesterol".
        most_frequent_spelling = spellings.mode()[0]
        # If the database already has this molecule, keep its spelling instead.
        name = db_names.get(group, most_frequent_spelling)
        rows.append(
            {
                "entity_normalized": name,
                "nb_mentions": len(mentions),
                # Each file is listed once, in alphabetical order.
                "json_files": sorted(mentions["json_file"].unique()),
            }
        )
    logger.success(f"Found {len(rows)} unique MOL entities.")
    return pd.DataFrame(rows).set_index("entity_normalized")


def load_json_database(path: Path, logger: "loguru.Logger" = loguru.logger) -> dict:
    """Load the JSON database, keyed by normalized name.

    Returns
    -------
    dict
        Molecules keyed by normalized name (empty if the file does not exist).

    Raises
    ------
    OSError
        If the file cannot be opened.
    ValueError
        If the file is not valid JSON or if a category is unknown.
    """
    logger.info(f"Loading the JSON database from {path}.")
    # First run: the database does not exist yet.
    if not path.exists():
        return {}
    try:
        db_molecules = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error(f"Cannot read the database {path}: {exc}")
        # Stop here: an empty database would overwrite the file when saved.
        raise
    database = {}
    for molecule in db_molecules:
        # Detect typos in the categories edited by hand.
        if molecule["category"] not in CATEGORIES:
            msg = f"Unknown category '{molecule['category']}' for '{molecule['name']}'."
            raise ValueError(msg)
        # Molecules saved before this field existed have no justification.
        molecule.setdefault("justification", "")
        # Transform the list of dictionaries into a dictionary keyed by normalized name.
        # Before: [{"name": "POPC", "category": "Lipid"}]
        # After: {"POPC": {"name": "POPC", "category": "Lipid"}}
        database[clean_text(molecule["name"])] = molecule
    if database:
        first_mol = next(iter(database))
        logger.info(f"First example: {first_mol} -> {database[first_mol]['category']}")
    logger.success(f"Collected {len(database)} molecules existing in the database.")
    return database


def make_molecule(
    name: str, category: str = "Undefined", source: str = "", db_id: str = ""
) -> dict:
    """Build the database entry of a molecule (automatic curation).

    Returns
    -------
    dict
        Molecule with name, category, database_source, database_id,
        database_url, curation and justification (of the LLM).
    """
    # PubChem IDs are stored as "CID:5491283" but the URL only takes the number.
    url = (
        DATABASE_ENTRY_URLS[source].format(db_id.removeprefix("CID:")) if db_id else ""
    )
    return {
        "name": name,
        "category": category,
        "database_source": source,
        "database_id": db_id,
        "database_url": url,
        "curation": "automatic",
        # Only filled by the LLM.
        "justification": "",
    }


def query_uniprot(query: str) -> list:
    """Search UniProtKB.

    Returns
    -------
    list
        Up to 10 UniProt entries.
    """
    params = {
        "query": query,
        "fields": "accession,protein_name,gene_names,organism_name",
        "size": 10,
    }
    return (get_json(f"{API_UNIPROT}/search", params) or {}).get("results", [])


def classify_with_regex(names: list, database: dict, logger: "loguru.Logger") -> list:
    """Classify the names that look like an identifier or a sequence.

    Returns
    -------
    list
        Molecules that are still not classified.
    """
    remaining = []
    for name in names:
        code = name.upper()
        try:
            # Is PDB ID ?
            # A PDB code needs a letter (1000 is a number) and must exist on RCSB.
            if (
                PDB_REGEX.match(name)
                and re.search(r"[A-Za-z]", name)
                and get_json(f"{API_PDB}/{code}")
            ):
                database[name] = make_molecule(name, "PDB ID", "RCSB PDB", code)
            # Is UniProt ID ?
            # Search the accession without its isoform suffix (P01116-2 -> P01116).
            elif UNIPROT_REGEX.match(name) and query_uniprot(
                f"accession:{code.split('-')[0]}"
            ):
                database[name] = make_molecule(name, "UniProt ID", "UniProt", code)
            # Is a sequence (DNA, RNA, peptide, repeated motif) ?
            # DNA / RNA: 5'-ACGT-3' (case ignored)
            # Peptide: 9+ amino acids, optional amidation (RKIRRKFKK-NH2)
            # Repeated motif: within parentheses, followed by a number (GVPGV)7
            elif any(regex.match(name) for regex in SEQUENCE_REGEXES):
                database[name] = make_molecule(name, "Sequence", "Regex")
            # Not classified by regex: keep it for the next steps.
            else:
                remaining.append(name)
        except RuntimeError as exc:
            logger.error(f"'{name}': {exc}")
            remaining.append(name)
    logger.info(f"Classified by regex: {len(names) - len(remaining)}/{len(names)}.")
    return remaining


def get_json(url: str, params: dict | None = None) -> dict | None:
    """GET a JSON resource, retrying on rate limits and server errors.

    Returns
    -------
    dict | None
        Decoded JSON, or None when the resource does not exist.

    Raises
    ------
    RuntimeError
        If the database still fails after all retries.
    """
    for attempt in range(5):
        # PubChem allows 5 requests per second.
        if url.startswith(API_PUBCHEM):
            time.sleep(0.25)
        try:
            response = HTTP_CLIENT.get(url, params=params)
        except httpx.HTTPError:
            response = None
        if response is not None and response.status_code == 200:
            return response.json()
        # Network error, rate limit or server error: wait longer each time.
        if response is None or response.status_code in {429, 500, 502, 503, 504}:
            time.sleep(2**attempt)
            continue
        # Any other status (404...): the resource does not exist.
        return None
    msg = f"Database unreachable: {url}"
    raise RuntimeError(msg)


def get_chebi_category(chebi_id: str, name: str, definition: str = "") -> str:
    """Map a ChEBI term to a category.

    Returns
    -------
    str
        Category of the first pattern found, "Small molecule" when none matches.
    """
    # 1. Search in the name of the term, 2. then in its definition.
    # Example of name: "mannose" ends in -ose -> Sugar.
    # Example of definition: "poly(ethylene glycol)" (CHEBI:46793)
    # "A polymer composed of repeating ethyleneoxy units." -> Material, polymer.
    for text in (name, definition):
        for category, pattern in CHEBI_PATTERNS.items():
            if pattern.search(text.lower()):
                return category
    # 3. Get the ancestors of the term (its families), and search in their names.
    # Example: GLP-1 (CHEBI:749689) has no definition, and no pattern matches its name.
    # With ancestors: "polypeptide", "peptide"... -> Protein.
    # The tree of ancestors can be seen on the OLS website:
    # https://www.ebi.ac.uk/ols4/ontologies/chebi/classes?obo_id=CHEBI:749689
    url = f"{API_OLS}/ontologies/chebi/hierarchicalAncestors"
    data = get_json(url, {"id": chebi_id, "size": 1000}) or {}
    ancestors = data.get("_embedded", {}).get("terms", [])
    for category, pattern in CHEBI_PATTERNS.items():
        for ancestor in ancestors:
            if pattern.search(ancestor["label"].lower()):
                return category
    # Assign "Small molecule" when none of the patterns matches.
    # Example: caffeine (CHEBI:27732)
    return "Small molecule"


def search_chebi(name: str) -> tuple | None:
    """Find a ChEBI term whose label or synonym is `name`.

    The case is ignored, except for short upper-case synonyms (AIR is not "air").

    Returns
    -------
    tuple | None
        (category, ChEBI ID, label of the term), or None.
    """
    params = {
        "q": name.lower(),
        "ontology": "chebi",
        "exact": "true",
        "rows": 10,
        "fieldList": "obo_id,label,synonym,description",
    }
    data = get_json(f"{API_OLS}/search", params) or {}
    for term in data.get("response", {}).get("docs", []):
        chebi_id, name_in_chebi = term.get("obo_id", ""), term.get("label", "")
        # The search also returns terms imported from other ontologies.
        # Example: "has role" -> RO:0000087 (not a ChEBI ID, skipped).
        if not chebi_id.startswith("CHEBI:"):
            continue
        # Gather all the names of the term: its main name, then its synonyms.
        # Example: ["L-alanyl-L-glutamic acid", "AE", "Ala-Glu", ...]
        chebi_names = [name_in_chebi, *term.get("synonym", [])]
        for chebi_name in chebi_names:
            cleaned = clean_text(chebi_name)
            # Short acronyms must have the same case: "AIR" is not "air".
            # AIR is found (CHEBI:138560) but "air" is not.
            if len(chebi_name) <= 5 and chebi_name.isupper():
                is_same = cleaned == name
            # For the other names, the case is ignored: "Caffeine" is "caffeine".
            else:
                is_same = cleaned.lower() == name.lower()
            if is_same:
                # The definition is a list of sentences (empty if there is none).
                definition = " ".join(term.get("description", []))
                category = get_chebi_category(chebi_id, name_in_chebi, definition)
                return category, chebi_id, name_in_chebi
    return None


def search_uniprot(name: str) -> tuple | None:
    """Find a reviewed protein with `name` as gene symbol or protein name.

    The case is ignored for long names (same rule as `group_mol_entities`).
    For short names, only the case of the first letter is ignored.

    Returns
    -------
    tuple | None
        (category, UniProt accession, "protein name (organism)"), or None.
    """
    # Too short to be specific, or would break the quoted query.
    if len(name) < 3 or '"' in name:
        return None
    # Search the genes first, then the proteins.
    queries = [
        f'(gene_exact:"{name}") AND (reviewed:true)',  # TLR4
        f'(protein_name:"{name}") AND (reviewed:true)',  # Acetylcholinesterase
    ]
    for query in queries:
        for entry in query_uniprot(query):
            # Get all the names of the entry: genes, full and short protein names.
            # In the JSON, they are all written as: "value": "TLR4"
            entry_text = json.dumps(entry, ensure_ascii=False)
            uniprot_names = re.findall(r'"value": "(.*?)"', entry_text)
            for uniprot_name in uniprot_names:
                # UniProt also returns close names, so check that it is the same one.
                # Rejected: "lipase" is not "Gastric triacylglycerol lipase".
                cleaned = clean_text(uniprot_name)
                # Short names: only the case of the first letter is ignored, because
                # a protein and its gene can differ by this letter.
                # Accepted: "RecA" is the gene recA.
                # Accepted: "p53" is the gene P53 (synonym of TP53).
                # Rejected: the lipid "POPC" is not the gene popC.
                is_same = (
                    name[0].lower() == cleaned[:1].lower() and name[1:] == cleaned[1:]
                )
                # Long names: the case is ignored.
                # Accepted: "acetylcholinesterase" is "Acetylcholinesterase".
                if len(name) > 5 and name.lower() == cleaned.lower():
                    is_same = True
                if is_same:
                    # Label of the entry, shown to the LLM.
                    # Example: "Toll-like receptor 4 (Homo sapiens)"
                    description = entry.get("proteinDescription", {})
                    full_name = description.get("recommendedName", {}).get(
                        "fullName", {}
                    )
                    organism = entry.get("organism", {}).get("scientificName", "")
                    label = f"{full_name.get('value', '')} ({organism})"
                    return "Protein", entry["primaryAccession"], label
    return None


def search_pubchem(name: str) -> tuple | None:
    """Find a PubChem compound with `name` as synonym.

    Returns
    -------
    tuple | None
        (category, PubChem CID, title of the compound), or None.
    """
    quoted = urllib.parse.quote(name.lower(), safe="")
    data = get_json(f"{API_PUBCHEM}/compound/name/{quoted}/cids/JSON")
    if not data:
        return None
    cid = data["IdentifierList"]["CID"][0]
    cid_url = f"{API_PUBCHEM}/compound/cid/{cid}"
    data = get_json(f"{cid_url}/property/Title/JSON") or {}
    compound = data.get("PropertyTable", {}).get("Properties", [{}])[0]
    data = get_json(f"{cid_url}/synonyms/JSON") or {}
    info = data.get("InformationList", {}).get("Information", [{}])[0]
    # A ChEBI cross-reference is listed among the synonyms ("CHEBI:...").
    chebi_ids = [s for s in info.get("Synonym", []) if s.startswith("CHEBI:")]
    # Without ChEBI, PubChem does not tell the category: the LLM will choose it.
    # Example: Iturin C is a peptide, GM3 ganglioside is a lipid.
    category = "Undefined"
    if chebi_ids:
        category = get_chebi_category(chebi_ids[0], compound.get("Title", ""))
    return category, f"CID:{cid}", compound.get("Title", "")


def classify_with_databases(
    names: list, database: dict, logger: "loguru.Logger"
) -> dict:
    """Classify the names with ChEBI, then UniProt, then PubChem.

    Returns
    -------
    dict
        Label of the database entry of each name found, to show to the LLM
        (e.g. {"AE": "L-alanyl-L-glutamic acid"}).
    """
    searches = {
        "ChEBI": search_chebi,
        "UniProt": search_uniprot,
        "PubChem": search_pubchem,
    }
    database_matches = {}
    for index, name in enumerate(names, start=1):
        try:
            for source, search in searches.items():
                found = search(name)
                if found:
                    category, db_id, database_matches[name] = found
                    database[name] = make_molecule(name, category, source, db_id)
                    logger.debug(
                        f"[{index}/{len(names)}] {name} -> {category} ({source})"
                    )
                    # The first database that answers wins.
                    break
        except RuntimeError as exc:
            logger.error(f"'{name}': {exc}")
    logger.info(f"Classified by databases: {len(database_matches)}/{len(names)}.")
    return database_matches


def describe_datasets(files: list, datasets: dict) -> str:
    """Describe the datasets mentioning a molecule, to add to the LLM prompt.

    Returns
    -------
    str
        Name, URL and text of each dataset, separated by "---".
    """
    descriptions = []
    for file in files:
        dataset_name = Path(file).stem
        url = datasets[file].get("url", "")
        text = datasets[file].get("raw_text", "")
        descriptions.append(
            f"Dataset: {dataset_name}\nURL: {url}\nDescription:\n{text}"
        )
    return "\n\n".join(descriptions)


def classify_with_llm(
    names: list,
    database: dict,
    database_matches: dict,
    unique_mol_entities: pd.DataFrame,
    datasets: dict,
    llm_name: str,
    prompt_path: Path,
    logger: "loguru.Logger",
) -> None:
    """Step 3: classify the names with an LLM, given the datasets mentioning them.

    The names in `database_matches` were found in a database: the LLM confirms
    or rejects this match. The molecules are added to `database`.
    """
    client = instructor.from_openai(
        OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        ),
        mode=instructor.Mode.JSON,
    )
    # Prompt with three blanks to fill for each molecule:
    # {{ entity }}, {{ candidate }} and {{ datasets }}.
    template = Template(prompt_path.read_text(encoding="utf-8"))
    nb_classified = 0
    for molecule in names:
        # Database match to confirm, if any.
        if molecule in database_matches:
            entry = database[molecule]
            source = entry["database_source"]
            database_id = entry["database_id"]
            category = entry["category"]
            label = database_matches[molecule]
            candidate = f"{source} [ID: {database_id}]: {label} (category: {category})"
        else:
            candidate = "None"
        # Fill the blanks of the prompt.
        files = unique_mol_entities.loc[molecule, "json_files"]
        prompt = template.render(
            entity=molecule,
            candidate=candidate,
            datasets=describe_datasets(files, datasets),
        )
        # Ask the LLM. If it fails, go to the next molecule.
        try:
            answer = client.chat.completions.create(
                model=llm_name,
                response_model=MoleculeCategory,
                max_retries=3,
                temperature=0,  # try to get the same answer each time
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"LLM failed for '{molecule}': {exc}")
            continue
        nb_classified += 1
        logger.debug(f"{molecule} -> {answer.category}")
        logger.debug(f"Justification: {answer.justification}")
        if molecule in database_matches and answer.candidate_matches:
            # Match confirmed: keep the database ID, with the category of the LLM.
            database[molecule]["category"] = answer.category
            database[molecule]["database_source"] += f" (checked by LLM ({llm_name}))"
        else:
            # No match, or match rejected: only keep the answer of the LLM.
            database[molecule] = make_molecule(
                molecule, answer.category, f"LLM ({llm_name})"
            )
        # Keep the explanation of the LLM, for the manual review.
        database[molecule]["justification"] = answer.justification
    logger.info(f"Classified by LLM ({llm_name}): {nb_classified}/{len(names)}.")


def save_database(
    database: dict, unique_mol_entities: pd.DataFrame, datasets: dict, path: Path
) -> None:
    """Write the JSON database.

    Fields updated at each run, to help the manual review:
    - "nb_mentions": number of mentions in the annotation files.
    - "datasets": annotation files mentioning the entity, with their URL.
    """
    # Sort by category (order of CATEGORIES), then by name.
    # Example: proteins from A to Z, then lipids from A to Z...
    molecules = []
    for category in CATEGORIES:
        for name in sorted(database):
            if database[name]["category"] != category:
                continue
            # Molecules of entities no longer in the annotations have no mention.
            nb_mentions, files = 0, []
            if name in unique_mol_entities.index:
                # Convert the NumPy integer, which JSON cannot store.
                nb_mentions = int(unique_mol_entities.loc[name, "nb_mentions"])
                files = unique_mol_entities.loc[name, "json_files"]
            molecules.append(
                {
                    **database[name],
                    "nb_mentions": nb_mentions,
                    "datasets": {file: datasets[file].get("url") for file in files},
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(molecules, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def classify_mol_entities(
    annotations_path: Path,
    db_json_path: Path,
    llm_name: str | None = None,
    prompt_path: Path | None = None,
    logger: "loguru.Logger" = loguru.logger,
) -> dict:
    """Add MOL entities to the JSON database and classify them.

    The JSON database at `db_json_path` is created or updated. Without
    `llm_name`, no LLM fallback is used.

    Returns
    -------
    dict
        Database molecules keyed by normalized name.
    """
    # Load the existing database (or create an empty one).
    database = load_json_database(db_json_path, logger)
    # Collect the MOL entities of the annotation files,
    # and group them by normalized name (case ignored for long names only).
    # Example: "POPC" and "popc" are two different entities.
    # Example: "Cholesterol" and "cholesterol" are the same.
    # Example: "β-amylase" and "beta-amylase" are the same.
    unique_mol_entities = group_mol_entities(
        collect_entities(annotations_path, logger), database, logger
    )
    # Load each annotation file once, to get the URL and the text of its dataset.
    # Example: {"zenodo_573274.json": { "raw_text": "...", "url": "https://..."}}
    json_files = unique_mol_entities["json_files"].explode().unique()
    datasets = {}
    for file in json_files:
        content = (annotations_path / file).read_text(encoding="utf-8")
        datasets[file] = json.loads(content)
    # Select the entities to classify.
    short_names = set()
    to_classify = []
    for name in unique_mol_entities.index:
        # Short names (AE, DPPC, p53) are ambiguous: the LLM checks them.
        if len(name) <= 5 and any(char.isalpha() for char in name):
            short_names.add(name)
        # New entities are handled as "Undefined" ones.
        molecule = database.get(name, make_molecule(name, category="Undefined"))
        # Short name found in a database but not checked by the LLM yet.
        to_check = (
            llm_name
            and name in short_names
            and molecule["database_source"] in {"ChEBI", "UniProt", "PubChem"}
        )
        # Only classify the entity if it is new or "Undefined", or if it is a
        # short name found in a database but not checked by the LLM yet.
        if molecule["curation"] != "manual" and (
            molecule["category"] == "Undefined" or to_check
        ):
            to_classify.append(name)
    logger.info(f"Classifying {len(to_classify)} new entities...")
    try:
        # Step 1: names that look like an identifier or a sequence.
        remaining = classify_with_regex(to_classify, database, logger)
        # Step 2: public databases (ChEBI, UniProt, PubChem).
        database_matches = classify_with_databases(remaining, database, logger)
        # Step 3: LLM, for the names not found, the short names to check and the
        # names found in PubChem without a category.
        if llm_name:
            for_llm = [
                name
                for name in remaining
                if name not in database_matches
                or name in short_names
                or database[name]["category"] == "Undefined"
            ]
            classify_with_llm(
                for_llm,
                database,
                database_matches,
                unique_mol_entities,
                datasets,
                llm_name,
                prompt_path,
                logger,
            )
    finally:
        # Save even after an interruption, so that nothing is lost.
        for name in to_classify:
            # If the entity is still missing from the database, add it as "Undefined".
            database.setdefault(name, make_molecule(name, category="Undefined"))
        save_database(database, unique_mol_entities, datasets, db_json_path)
    # Log entities to check by hand, with the links of their datasets.
    logger.warning("Entities to check by hand:")
    for name, files in unique_mol_entities["json_files"].items():
        if database[name]["category"] == "Undefined":
            urls = ", ".join(datasets[f].get("url") or Path(f).stem for f in files)
            logger.warning(f"{name} -> Undefined")
            logger.debug(f"URLs: {urls}")
    counts = Counter(database[name]["category"] for name in unique_mol_entities.index)
    logger.info(f"Molecule categories found in {annotations_path}:")
    # Sort by decreasing count.
    for category, count in counts.most_common():
        logger.info(f"{category:<20} {count}")
    logger.success(f"Saved {len(database)} molecules in: {db_json_path} successfully!")
    return database


def summarize_mol_categories(
    df_entities: pd.DataFrame, database_path: Path, n_examples: int = 3
) -> pd.DataFrame:
    """Summarize the groundtruth MOL entities per category.

    `df_entities` is the output of `collect_entities`. The categories are
    read from the database built by `classify_mol_entities` (no database or LLM
    query). Entities missing from the database are counted as "Undefined".
    `n_examples` is the number of most-mentioned names shown as examples.

    Returns
    -------
    pd.DataFrame
        Columns: Category, Mentions, Share of MOL, Unique mentions, Texts,
        Examples. Sorted by decreasing mentions.
    """
    database = load_json_database(database_path)
    # One row per unique molecule, with its number of mentions and its files.
    mol_entities = group_mol_entities(df_entities, database)
    # Find the category of each molecule in the database.
    categories = []
    for name in mol_entities.index:
        molecule = database.get(name, {})
        categories.append(molecule.get("category", "Undefined"))
    mol_entities["category"] = categories
    # Total number of mentions, to compute the share of each category.
    total_mentions = mol_entities["nb_mentions"].sum()
    # Build one row of the summary per category.
    rows = []
    for category, molecules in mol_entities.groupby("category"):
        # Most-mentioned molecules first, to use them as examples.
        molecules = molecules.sort_values("nb_mentions", ascending=False)
        nb_mentions = molecules["nb_mentions"].sum()
        # One row per file, so that a file shared by several molecules of the
        # category is counted once.
        nb_texts = molecules["json_files"].explode().nunique()
        examples = molecules.index[:n_examples]
        rows.append(
            {
                "Category": category,
                "Mentions": nb_mentions,
                "Share of MOL": nb_mentions / total_mentions,
                "Unique mentions": len(molecules),
                "Texts": nb_texts,
                "Examples": ", ".join(examples),
            }
        )
    # Most-mentioned categories first.
    summary = pd.DataFrame(rows).sort_values("Mentions", ascending=False)
    return summary.reset_index(drop=True)


@click.command()
@click.option(
    "--annotations-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    show_default=True,
    help="Folder containing the JSON annotation files.",
)
@click.option(
    "--db",
    "db_json_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="JSON database of the MOL entities (created or updated).",
)
@click.option(
    "--llm",
    "llm_name",
    type=str,
    default=None,
    help="OpenRouter model used as fallback (e.g. google/gemma-4-31b-it). "
    "No LLM fallback when not given.",
)
@click.option(
    "--prompt-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Prompt template of the LLM fallback.",
)
def run_main_from_cli(
    annotations_path: Path, db_json_path: Path, llm_name: str | None, prompt_path: Path
) -> None:
    """Classify the groundtruth MOL entities into molecular categories."""
    # Set up logger.
    logpath = f"logs/molecule_classification/{Path(annotations_path).name}"
    logpath += f"_{datetime.now(UTC).strftime('%Y-%m-%d')}.log"
    logger = create_logger(logpath=logpath)
    # Run the classification.
    classify_mol_entities(
        annotations_path,
        db_json_path,
        llm_name,
        prompt_path,
        logger=logger,
    )


if __name__ == "__main__":
    run_main_from_cli()
