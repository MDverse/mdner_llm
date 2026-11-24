"""Classify MOL entities of the groundtruth into molecular categories.

All MOL entities of the annotation files are stored in a JSON database, one
molecule per entity (grouped by normalized name):

    {
        "name": "POPC",
        "category": "Lipid",
        "database_source": "ChEBI",
        "database_id": "CHEBI:74669",
        "database_url": "https://www.ebi.ac.uk/chebi/CHEBI:74669",
        "curation": "automatic"
    }

Entities without a category have the category "Undefined". Molecules with
curation "manual" are edited by hand and never changed by this script.
Molecules already categorized are not queried again: only new entities and
"Undefined" ones are (re)classified.

Classification steps, each one on the entities left by the previous one:

1. Regex, on the shape of the name: PDB ID (checked on RCSB), UniProt accession
   (checked on UniProt), DNA / RNA / peptide sequence.
2. Public databases (the first one that answers wins):
   - ChEBI (EBI OLS): exact label or synonym; the category is read from the
     label of the term, or else from its definition, or else from the labels of
     its ancestors.
   - UniProt: gene symbol (same case, so POPC does not match the gene popC)
     or exact protein name -> Protein, peptide.
   - PubChem compound: ChEBI cross-reference if any, else formula and charge.
3. LLM (OpenRouter model given with --llm), given the descriptions of the
   datasets mentioning the entity. Its answer is restricted to the categories.
   It classifies:
   - the entities not found in step 2;
   - the short names (at most 5 characters: AE, DPPC, p53) found in step 2,
     because they are ambiguous (AE is also the dipeptide Ala-Glu in ChEBI).
     If the LLM confirms the match, the database ID is kept (source "ChEBI
     (checked by LLM)"...), otherwise the LLM category is used without ID.
"""

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

API_OLS = "https://www.ebi.ac.uk/ols4/api"
API_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
API_UNIPROT = "https://rest.uniprot.org/uniprotkb"
API_PDB = "https://data.rcsb.org/rest/v1/core/entry"
# One client for all the requests: the connections are reused (faster).
HTTP_CLIENT = httpx.Client(timeout=60, follow_redirects=True)

# Web page of a database entry, filled with its ID.
ENTRY_URLS = {
    "RCSB PDB": "https://www.rcsb.org/structure/{}",
    "UniProt": "https://www.uniprot.org/uniprotkb/{}/entry",
    "ChEBI": "https://www.ebi.ac.uk/chebi/{}",
    "PubChem": "https://pubchem.ncbi.nlm.nih.gov/compound/{}",
}

# Patterns on the name, the definition and the ancestors of a ChEBI term, checked in
# order.
# Lipids come before ions: ChEBI files many phospholipids under their
# cationic form (e.g. DPPC is a "quaternary ammonium ion").
CHEBI_PATTERNS = {
    "Nucleic acid": re.compile(r"nucleic acid|oligonucleotide|\b[dr]na\b"),
    "Protein, peptide": re.compile(r"\b(?:protein|(?:poly|oligo|di)?peptide)s?\b"),
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
        r"\bsalt$|monoatomic (?:cation|anion|ion)|inorganic (?:cation|anion|ion)"
        r"|^ion$|alkali metal atom|alkaline earth metal atom|^halogen$"
    ),
}

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
    df_entities: pd.DataFrame, logger: "loguru.Logger" = loguru.logger
) -> pd.DataFrame:
    """Group the MOL entities of `collect_entities` by normalized name.

    Returns
    -------
    pd.DataFrame
        Indexed by normalized name (case kept), with the columns: nb_mentions
        and json_files (sorted names of the annotation files mentioning the
        entity).
    """
    molecules = df_entities[df_entities["category"] == "MOL"]
    rows = []
    # Each group contains all the mentions of one entity. The case matters:
    # POPC and popc are two entities.
    for name, mentions in molecules.groupby("entity_normalized"):
        rows.append(
            {
                "entity_normalized": clean_text(name),
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
        database_url and curation.
    """
    # PubChem IDs are stored as "CID:5491283" but the URL only takes the number.
    url = ENTRY_URLS[source].format(db_id.removeprefix("CID:")) if db_id else ""
    return {
        "name": name,
        "category": category,
        "database_source": source,
        "database_id": db_id,
        "database_url": url,
        "curation": "automatic",
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

    The patterns are searched in three texts, and the first match wins:

    1. the name of the term (no request);
    2. its definition (no request, already given by the search);
    3. the names of its ancestors (one request).

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
    # Example: "L-alanyl-L-glutamic acid" (CHEBI:61565)
    # ancestors: "peptide", "dipeptide", "oligopeptide"... -> Protein, peptide.
    # The tree of ancestors can be seen on the OLS website:
    # https://www.ebi.ac.uk/ols4/ontologies/chebi/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FCHEBI_61565
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

    Returns
    -------
    tuple | None
        (category, UniProt accession, "protein name (organism)"), or None.
    """
    # Too short to be specific, or would break the quoted query.
    if len(name) < 3 or '"' in name:
        return None
    hits = []
    # Gene symbols are written with capitals (TLR4, RecA) and one word.
    if " " not in name and name.lower() != name:
        for hit in query_uniprot(f'(gene_exact:"{name}") AND (reviewed:true)'):
            symbols = [
                value.get("value", "")
                for gene in hit.get("genes", [])
                for value in [gene.get("geneName", {}), *gene.get("synonyms", [])]
            ]
            # Accepted: TLR4/TLR4, RecA/recA, ChAT/CHAT. Rejected: POPC/popC.
            if any(
                name in {symbol, symbol[:1].upper() + symbol[1:]}
                or (symbol.isupper() and name.upper() == symbol)
                for symbol in symbols
            ):
                hits.append(hit)
    # No gene: search the full and short names (Acetylcholinesterase, AChE).
    if not hits:
        query = f'(protein_name:"{name.lower()}") AND (reviewed:true)'
        for hit in query_uniprot(query):
            description = hit.get("proteinDescription", {})
            proteins = [
                clean_text(value.get("value", "")).lower()
                for names in [
                    description.get("recommendedName", {}),
                    *description.get("alternativeNames", []),
                ]
                for value in [names.get("fullName", {}), *names.get("shortNames", [])]
            ]
            if name.lower() in proteins:
                hits.append(hit)
    if not hits:
        return None
    # Keep the first match. Its label is shown to the LLM.
    hit = hits[0]
    description = hit.get("proteinDescription", {}).get("recommendedName", {})
    protein = description.get("fullName", {}).get("value", "")
    organism = hit.get("organism", {}).get("scientificName", "")
    return "Protein, peptide", hit["primaryAccession"], f"{protein} ({organism})"


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
    data = get_json(f"{cid_url}/property/MolecularFormula,Charge,Title/JSON") or {}
    compound = data.get("PropertyTable", {}).get("Properties", [{}])[0]
    data = get_json(f"{cid_url}/synonyms/JSON") or {}
    info = data.get("InformationList", {}).get("Information", [{}])[0]
    # A ChEBI cross-reference is listed among the synonyms ("CHEBI:...").
    chebi_ids = [s for s in info.get("Synonym", []) if s.startswith("CHEBI:")]
    if chebi_ids:
        category = get_chebi_category(chebi_ids[0], compound.get("Title", ""))
    # Without ChEBI: several components (Na.Cl) or a charge make an ion.
    elif "." in compound.get("MolecularFormula", "") or compound.get("Charge", 0) != 0:
        category = "Ion"
    else:
        category = "Small molecule"
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
    # The {{ }} of the template are filled for each entity.
    template = Template(prompt_path.read_text(encoding="utf-8"))
    nb_classified = 0
    for name in names:
        contexts = [
            f"Dataset: {Path(file).stem}\nURL: {datasets[file].get('url', '')}\n"
            f"Description:\n{datasets[file].get('raw_text', '')}"
            for file in unique_mol_entities.loc[name, "json_files"]
        ]
        # Database match to confirm, if any.
        candidate = "None"
        if name in database_matches:
            molecule = database[name]
            candidate = (
                f"{molecule['database_source']} {molecule['database_id']}: "
                f"{database_matches[name]} (category: {molecule['category']})"
            )
        prompt = template.render(
            entity=name, candidate=candidate, datasets="\n\n---\n\n".join(contexts)
        )
        try:
            answer = client.chat.completions.create(
                model=llm_name,
                response_model=MoleculeCategory,
                max_retries=3,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"LLM failed for '{name}': {exc}")
            continue
        nb_classified += 1
        if name in database_matches and answer.candidate_matches:
            # Keep the database ID, with the category chosen by the LLM.
            database[name]["category"] = answer.category
            database[name]["database_source"] += " (checked by LLM)"
        else:
            database[name] = make_molecule(name, answer.category, f"LLM ({llm_name})")
    logger.info(f"Classified by LLM: {nb_classified}/{len(names)}.")


def save_database(
    database: dict, unique_mol_entities: pd.DataFrame, datasets: dict, path: Path
) -> None:
    """Write the database as JSON, and as TSV to review it in a spreadsheet.

    Columns added to the TSV:
    - "verified": TRUE for manual molecules, FALSE otherwise.
    - "nb_mentions": number of mentions in the annotation files.
    - "nb_unique_mentions": number of datasets mentioning the entity.
    - "dataset_json_file_name": annotation files, one per line.
    - "dataset_url": links of the datasets, one per line.
    """
    # Sort by category (order of CATEGORIES), then by name.
    # Example: proteins from A to Z, then lipids from A to Z...
    molecules = {}
    for category in CATEGORIES:
        for name in sorted(database):
            if database[name]["category"] == category:
                molecules[name] = database[name]
    # Save the JSON database.
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(list(molecules.values()), indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")
    rows = []
    for name, molecule in molecules.items():
        # Molecules of entities no longer in the annotations have no mention.
        nb_mentions, files = 0, []
        if name in unique_mol_entities.index:
            nb_mentions = unique_mol_entities.loc[name, "nb_mentions"]
            files = unique_mol_entities.loc[name, "json_files"]
        rows.append(
            {
                "name": molecule["name"],
                "category": molecule["category"],
                "verified": "TRUE" if molecule["curation"] == "manual" else "FALSE",
                **molecule,
                "nb_mentions": nb_mentions,
                "nb_unique_mentions": len(files),
                "dataset_json_file_name": "\n".join(files),
                # Fall back on the file name when the dataset has no URL.
                "dataset_url": "\n".join(datasets[file].get("url") for file in files),
            }
        )
    # Save the TSV database (for manual review in a spreadsheet).
    pd.DataFrame(rows).to_csv(path.with_suffix(".tsv"), sep="\t", index=False)


def classify_mol_entities(
    annotations_path: Path,
    db_json_path: Path,
    llm_name: str | None = None,
    prompt_path: Path | None = None,
    logger: "loguru.Logger" = loguru.logger,
) -> dict:
    """Add MOL entities to the JSON database and classify them.

    The JSON database at `db_json_path` is created or updated, and exported
    as TSV next to it. Without `llm_name`, no LLM fallback is used.

    Returns
    -------
    dict
        Database molecules keyed by normalized name.
    """
    # Collect the MOL entities of the annotation files,
    # and group them by normalized name (case kept).
    # Example: "POPC" and "popc" are two different entities.
    # Example: "β-amylase" and "beta-amylase" are the same.
    unique_mol_entities = group_mol_entities(
        collect_entities(annotations_path, logger), logger
    )
    # Load each annotation file once, to get the URL and the text of its dataset.
    # Example: {"zenodo_573274.json": { "raw_text": "...", "url": "https://..."}}
    json_files = unique_mol_entities["json_files"].explode().unique()
    datasets = {}
    for file in json_files:
        content = (annotations_path / file).read_text(encoding="utf-8")
        datasets[file] = json.loads(content)
    # Load the existing database (or create an empty one).
    database = load_json_database(db_json_path, logger)
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
        # Step 3: LLM, for the names not found and the short names to check.
        if llm_name:
            for_llm = [
                name
                for name in remaining
                if name not in database_matches or name in short_names
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
            database.setdefault(name, make_molecule(name))
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
        Columns: Category, Mentions, Share of MOL, Unique names, Texts,
        Examples. Sorted by decreasing mentions.
    """
    mol_entities = group_mol_entities(df_entities)
    database = load_json_database(database_path)
    mol_entities["category"] = [
        database.get(key, {}).get("category", "Undefined") for key in mol_entities.index
    ]
    total_mentions = mol_entities["nb_mentions"].sum()
    summary = []
    for category, group in mol_entities.groupby("category"):
        group = group.sort_values("nb_mentions", ascending=False)
        summary.append(
            {
                "Category": category,
                "Mentions": int(group["nb_mentions"].sum()),
                "Share of MOL": group["nb_mentions"].sum() / total_mentions,
                "Unique names": len(group),
                # Number of annotation files with at least one entity of the category.
                "Texts": len(set().union(*group["json_files"])),
                "Examples": ", ".join(group.index[:n_examples]),
            }
        )
    return (
        pd.DataFrame(summary)
        .sort_values("Mentions", ascending=False)
        .reset_index(drop=True)
    )


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
