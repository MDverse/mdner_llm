"""Classify MOL entities into molecular families with public databases.

No list of molecule names is written here: each unique MOL entity of the
annotation files is looked up in public databases. The first step that
answers gives the family:

1. Shape of the name (regex, on the original case):
   - PDB code (e.g. 6LU7), checked against the RCSB PDB API.
   - UniProt accession (e.g. P01116-2), checked against the UniProt API.
   - Sequence: DNA / RNA (5'-CGAT-3'), peptide in upper-case one-letter
     code (GRPNLKLLLG...) or repeated motif ((GVPGV)7, [UG]12).
2. ChEBI (through the EBI Ontology Lookup Service): the name is the label
   or a synonym of a ChEBI term. The family is read from the labels of the
   term and of its ChEBI ancestors (lipid, saccharide, salt, polymer...).
3. UniProt gene symbol (e.g. EGFR, TLR4, RecA) -> Protein, peptide.
   The case must agree with the gene symbol, so that POPC does not match
   the bacterial gene popC.
4. PubChem compound: its ChEBI cross-reference, when present, gives the
   family as in step 2; otherwise the formula and charge decide between
   ion / salt and small molecule.
5. UniProt protein name (e.g. glutamine synthetase) -> Protein, peptide.
6. MeSH descriptor (NCBI E-utilities): the family is read from its tree
   numbers (D12 proteins, D08.811 enzymes, D10 lipids, D09 carbohydrates...).
7. PubChem substance (e.g. polymers), then as in step 4.
8. Steps 2, 4 and 7 again with the singular of a plural name (alkanes).
9. Reviewed UniProt protein whose name contains the entity as whole words
   (e.g. smoothened -> "Protein smoothened") -> Protein, peptide.

Names found nowhere are labelled "Unknown". Database answers are stored in a
JSON cache, so each name is only queried once.

The output table has one row per unique MOL entity with the columns:
Entity, Family, Mentions, Texts, Json files, Source, Database ID.
"""

import json
import re
import time
import urllib.parse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import click
import httpx
import loguru
import pandas as pd

from mdner_llm.annotations.build_entity_inventory import collect_entities
from mdner_llm.logger import create_logger

API_OLS = "https://www.ebi.ac.uk/ols4/api"
API_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
API_UNIPROT = "https://rest.uniprot.org/uniprotkb"
API_PDB = "https://data.rcsb.org/rest/v1/core/entry"
API_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Minimum delay between two requests to the same host (seconds), to stay
# below the rate limits of the services (PubChem: 5 requests per second).
MIN_REQUEST_INTERVAL = {
    "rest.uniprot.org": 0.5,
    "pubchem.ncbi.nlm.nih.gov": 0.25,
    "www.ebi.ac.uk": 0.1,
    "data.rcsb.org": 0.1,
    "eutils.ncbi.nlm.nih.gov": 0.4,
}

# Family labels, in display order.
PROTEIN = "Protein, peptide"
LIPID = "Lipid, sterol"
SMALL_MOLECULE = "Small molecule"
ION = "Ion, salt"
MATERIAL = "Material, polymer"
PDB_CODE = "PDB code"
SUGAR = "Sugar"
NUCLEIC_ACID = "Nucleic acid"
SEQUENCE = "Sequence"
UNIPROT_ID = "UniProt ID"
UNKNOWN = "Unknown"

FAMILY_ORDER = [
    PROTEIN,
    LIPID,
    SMALL_MOLECULE,
    ION,
    MATERIAL,
    PDB_CODE,
    SUGAR,
    NUCLEIC_ACID,
    SEQUENCE,
    UNIPROT_ID,
    UNKNOWN,
]

# Patterns on the labels of a ChEBI term and of its ancestors, checked in
# this order. Lipids come before ions: ChEBI files many phospholipids under
# their cationic form (e.g. DPPC is a "quaternary ammonium ion").
CHEBI_FAMILY_PATTERNS: list[tuple[str, re.Pattern]] = [
    (NUCLEIC_ACID, re.compile(r"nucleic acid|oligonucleotide|\b[dr]na\b")),
    (
        PROTEIN,
        re.compile(r"^(?:protein|(?:poly|oligo|di|tri|cyclic )?peptide)s?$"),
    ),
    (
        SUGAR,
        re.compile(
            r"^(?:mono|di|oligo|poly)saccharides?$|^(?:n-)?glycans?$|glycoside$"
            r"|^carbohydrates?$|^(?:hex|pent|ald|ket)ose$|furanose|pyranose"
        ),
    ),
    (
        LIPID,
        re.compile(
            r"lipid|fatty acid|fatty acyl|glycerophospho|phosphocholine"
            r"|phosphoethanolamine|steroid|sterol|sphingo|ceramide|ganglioside"
            r"|glyceride"
        ),
    ),
    (MATERIAL, re.compile(r"polymer|macromolecule|elemental carbon|nanotube")),
    (
        ION,
        re.compile(
            r"\bsalt\b|monoatomic (?:cation|anion|ion)|inorganic (?:cation|anion|ion)"
            r"|ammonium ion|guanidinium|iminium ion|^ion$"
            r"|alkali metal atom|alkaline earth metal atom|^halogen$"
        ),
    ),
]

# MeSH tree numbers (prefixes) that define a family, checked in this order.
# Other branches of the Chemicals and Drugs tree (D) give a small molecule.
MESH_FAMILY_TREES: list[tuple[str, str]] = [
    ("D13", NUCLEIC_ACID),  # Nucleic Acids, Nucleotides, and Nucleosides
    ("D12", PROTEIN),  # Amino Acids, Peptides, and Proteins
    ("D08.811", PROTEIN),  # Enzymes
    ("D10", LIPID),  # Lipids
    ("D09", SUGAR),  # Carbohydrates
    ("D05", MATERIAL),  # Macromolecular Substances (polymers)
    ("D25", MATERIAL),  # Biomedical and Dental Materials
]

# Elements found as free ions in simulation boxes (alkali and alkaline-earth
# metals, zinc, halogens): a one-element formula of these is an ion.
ION_ELEMENTS = {
    "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "F", "Cl", "Br", "I",
}  # fmt: skip

# Identifiers and sequences recognised by their shape (original case).
PDB_REGEX = re.compile(r"^[1-9][A-Za-z0-9]{3}$")
UNIPROT_REGEX = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
    r"(?:-\d+)?$",
    flags=re.IGNORECASE,
)
DNA_REGEX = re.compile(r"^5'-[ACGTacgt]+-3'$")
RNA_REGEX = re.compile(r"^5'-[ACGUacgu]+-3'$")
PEPTIDE_REGEX = re.compile(r"^(?![ACGTU]+$)[ACDEFGHIKLMNPQRSTVWY]{9,}(?:-NH2)?$")
REPEAT_REGEX = re.compile(r"^[\[(][A-Za-z]{2,}[\])]\d+$")

# Unicode punctuation mapped to ASCII before lookup.
_PUNCTUATION_MAP = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "\u2019": "'",  # right single quotation mark
        "`": "'",
        "\u2082": "2",  # subscript two
        "\u2083": "3",  # subscript three
        "\u00ad": None,  # soft hyphen
    }
)


def clean_text(text: str) -> str:
    """Clean an entity text, keeping its case.

    Returns
    -------
    str
        Text with unicode dashes, quotes and subscripts mapped to ASCII,
        soft hyphens removed and whitespace collapsed.
    """
    text = str(text).translate(_PUNCTUATION_MAP)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(text: str) -> str:
    """Normalize an entity text for grouping and case-insensitive lookups.

    Returns
    -------
    str
        Cleaned, lowercased text.
    """
    return clean_text(text).lower()


def is_short_acronym(value: str) -> bool:
    """Tell whether a synonym is a short upper-case acronym (AIR, P53).

    Returns
    -------
    bool
        True for acronyms of at most five characters.
    """
    return len(value) <= 5 and value.upper() == value and value.lower() != value


def same_case_as_gene(text: str, gene: str) -> bool:
    """Compare an entity text with a gene symbol, respecting the case.

    Accepted: TLR4 / TLR4, RecA / recA (protein spelling of a gene),
    ChAT / CHAT (upper-case gene, text with capitals).
    Rejected: POPC / popC, METH / meth.

    Returns
    -------
    bool
        True if the text is written as the gene symbol.
    """
    if text in (gene, gene[:1].upper() + gene[1:]):
        return True
    has_capital = any(char.isupper() for char in text)
    return gene.isupper() and has_capital and text.upper() == gene


class DatabaseClient:
    """Query public databases, with a JSON cache of the answers."""

    def __init__(
        self,
        cache_path: Path | None = None,
        timeout: float = 60,
        max_retries: int = 6,
        logger: "loguru.Logger" = loguru.logger,
    ) -> None:
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)
        self.max_retries = max_retries
        self.logger = logger
        self.cache_path = cache_path
        self.cache: dict[str, object] = {}
        self._lock = Lock()
        self._host_locks: dict[str, Lock] = {
            host: Lock() for host in MIN_REQUEST_INTERVAL
        }
        self._last_request: dict[str, float] = {}
        if cache_path is not None and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def save_cache(self) -> None:
        """Write the cache to disk."""
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            content = json.dumps(self.cache, indent=1, ensure_ascii=False)
        self.cache_path.write_text(content, encoding="utf-8")

    def get_json(self, url: str, params: dict | None = None) -> dict | None:
        """GET a JSON resource, retrying on rate limits and server errors.

        Returns
        -------
        dict | None
            Decoded JSON, or None when the resource does not exist.

        Raises
        ------
        RuntimeError
            If the database still fails after all retries. The answer is
            then not cached, so a later run tries again.
        """
        for attempt in range(self.max_retries):
            self._wait_for_host(url)
            try:
                response = self.client.get(url, params=params)
            except httpx.HTTPError as exc:
                self.logger.debug(f"Request failed ({exc}), retrying: {url}")
                time.sleep(2**attempt)
                continue
            if response.status_code == 200:
                return response.json()
            if response.status_code in (429, 500, 502, 503, 504):
                # Wait as long as the server asks (Retry-After), if it does.
                retry_after = response.headers.get("retry-after", "")
                delay = int(retry_after) if retry_after.isdigit() else 2**attempt
                self.logger.debug(
                    f"HTTP {response.status_code}, retrying in {delay} s: {url}"
                )
                time.sleep(delay)
                continue
            return None
        msg = f"Database unreachable after {self.max_retries} attempts: {url}"
        raise RuntimeError(msg)

    def _wait_for_host(self, url: str) -> None:
        """Sleep so that requests to one host respect its minimum interval."""
        host = urllib.parse.urlparse(url).netloc
        if host not in self._host_locks:
            return
        with self._host_locks[host]:
            elapsed = time.monotonic() - self._last_request.get(host, 0.0)
            time.sleep(max(0.0, MIN_REQUEST_INTERVAL[host] - elapsed))
            self._last_request[host] = time.monotonic()

    def cached(self, key: str, func: Callable[[], object]) -> object:
        """Return the cached answer for `key`, computing it if needed.

        Returns
        -------
        object
            Cached or freshly computed answer.
        """
        with self._lock:
            if key in self.cache:
                return self.cache[key]
        value = func()
        with self._lock:
            self.cache[key] = value
        return value

    # ------------------------------------------------------------------ PDB
    def pdb_exists(self, code: str) -> bool:
        """Check that a PDB entry exists.

        Returns
        -------
        bool
            True if RCSB knows the entry.
        """
        code = code.upper()
        return self.cached(
            f"pdb:{code}", lambda: self.get_json(f"{API_PDB}/{code}") is not None
        )

    # -------------------------------------------------------------- UniProt
    def uniprot_search(self, query: str, size: int = 10) -> list[dict]:
        """Search reviewed UniProt entries.

        Returns
        -------
        list[dict]
            Entries with accession, gene names and protein names.
        """

        def query_api() -> list[dict]:
            data = self.get_json(
                f"{API_UNIPROT}/search",
                params={
                    "query": f"({query}) AND (reviewed:true)",
                    "fields": "accession,protein_name,gene_names",
                    "size": size,
                },
            )
            return [
                {
                    "accession": hit["primaryAccession"],
                    "genes": _uniprot_gene_names(hit),
                    "proteins": _uniprot_protein_names(hit),
                }
                for hit in (data or {}).get("results", [])
            ]

        return self.cached(f"uniprot:{query}:{size}", query_api)

    def uniprot_accession_exists(self, accession: str) -> bool:
        """Check that a UniProt accession (or isoform) exists.

        Returns
        -------
        bool
            True if UniProt knows the accession.
        """
        base = accession.upper().split("-")[0]

        def query_api() -> bool:
            data = self.get_json(
                f"{API_UNIPROT}/search",
                params={"query": f"accession:{base}", "fields": "accession", "size": 1},
            )
            return bool(data and data.get("results"))

        return self.cached(f"uniprot_accession:{base}", query_api)

    def uniprot_gene(self, text: str) -> str | None:
        """Find a reviewed protein whose gene symbol is written as `text`.

        Returns
        -------
        str | None
            UniProt accession, or None.
        """
        # Gene symbols of proteins are written with capitals (TLR4, RecA):
        # an all-lowercase word (air, meth) is not looked up.
        if " " in text or len(text) < 3 or '"' in text or text.lower() == text:
            return None
        for hit in self.uniprot_search(f'gene_exact:"{text}"'):
            if any(same_case_as_gene(text, gene) for gene in hit["genes"]):
                return hit["accession"]
        return None

    def uniprot_protein_name(self, text: str, *, contained: bool = False) -> str | None:
        """Find a reviewed protein named `text` (case-insensitive).

        Parameters
        ----------
        text : str
            Entity text.
        contained : bool
            If True, the text may be only a part (whole words) of the
            protein name, e.g. "smoothened" in "Protein smoothened".

        Returns
        -------
        str | None
            UniProt accession, or None.
        """
        name = normalize_name(text)
        if len(name) < 4 or '"' in name:
            return None
        word_pattern = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
        for hit in self.uniprot_search(f'protein_name:"{name}"'):
            proteins = [normalize_name(protein) for protein in hit["proteins"]]
            if name in proteins or (
                contained and any(word_pattern.search(p) for p in proteins)
            ):
                return hit["accession"]
        return None

    # ---------------------------------------------------------------- ChEBI
    def chebi_term(self, text: str) -> dict | None:
        """Find the ChEBI term whose label or synonym is `text`.

        Matching is case-insensitive, except for short upper-case synonyms
        (e.g. "AIR" does not match the word "air").

        Returns
        -------
        dict | None
            {"id": "CHEBI:...", "label": ...}, or None.
        """
        name = normalize_name(text)

        def query_api() -> list[dict]:
            data = self.get_json(
                f"{API_OLS}/search",
                params={
                    "q": name,
                    "ontology": "chebi",
                    "exact": "true",
                    "rows": 10,
                    "fieldList": "obo_id,label,synonym",
                },
            )
            docs = (data or {}).get("response", {}).get("docs", [])
            return [
                {
                    "id": doc["obo_id"],
                    "label": doc.get("label", ""),
                    "names": [doc.get("label", ""), *doc.get("synonym", [])],
                }
                for doc in docs
                if doc.get("obo_id", "").startswith("CHEBI:")
            ]

        for term in self.cached(f"chebi:{name}", query_api):
            matches = [v for v in term["names"] if normalize_name(v) == name]
            if any(not is_short_acronym(v) or clean_text(v) == text for v in matches):
                return {"id": term["id"], "label": term["label"]}
        return None

    def chebi_ancestor_labels(self, chebi_id: str) -> list[str]:
        """List the labels of a ChEBI term and of its ancestors.

        Returns
        -------
        list[str]
            Lowercased labels, the term first.
        """

        def query_api() -> list[str]:
            iri = "http://purl.obolibrary.org/obo/" + chebi_id.replace(":", "_")
            encoded = urllib.parse.quote(urllib.parse.quote(iri, safe=""), safe="")
            term = self.get_json(f"{API_OLS}/ontologies/chebi/terms/{encoded}")
            ancestors = self.get_json(
                f"{API_OLS}/ontologies/chebi/terms/{encoded}/hierarchicalAncestors",
                params={"size": 1000},
            )
            labels = [(term or {}).get("label", "")]
            labels += [
                item["label"]
                for item in (ancestors or {}).get("_embedded", {}).get("terms", [])
            ]
            return [label.lower() for label in labels if label]

        return self.cached(f"chebi_ancestors:{chebi_id}", query_api)

    def chebi_family(self, chebi_id: str) -> str:
        """Map a ChEBI term to a family through its ancestors' labels.

        Returns
        -------
        str
            Family label; small molecule when no pattern matches.
        """
        labels = self.chebi_ancestor_labels(chebi_id)
        for family, pattern in CHEBI_FAMILY_PATTERNS:
            if any(pattern.search(label) for label in labels):
                return family
        return SMALL_MOLECULE

    # ----------------------------------------------------------------- MeSH
    def mesh_descriptor(self, text: str) -> dict | None:
        """Find the MeSH descriptor having `text` as heading or entry term.

        Returns
        -------
        dict | None
            {"id", "heading", "trees"}, or None.
        """
        name = normalize_name(text)
        if '"' in name:
            return None

        def query_api() -> dict | None:
            search = self.get_json(
                f"{API_EUTILS}/esearch.fcgi",
                params={"db": "mesh", "term": f'"{name}"', "retmode": "json"},
            )
            ids = (search or {}).get("esearchresult", {}).get("idlist", [])[:20]
            if not ids:
                return None
            summary = self.get_json(
                f"{API_EUTILS}/esummary.fcgi",
                params={"db": "mesh", "id": ",".join(ids), "retmode": "json"},
            )
            records = (summary or {}).get("result", {})
            for uid in ids:
                record = records.get(uid, {})
                terms = record.get("ds_meshterms", [])
                if record.get("ds_recordtype") != "descriptor" or name not in {
                    normalize_name(term) for term in terms
                }:
                    continue
                trees = [
                    link["treenum"]
                    for link in record.get("ds_idxlinks", [])
                    if link.get("treenum")
                ]
                return {
                    "id": record.get("ds_meshui", uid),
                    "heading": terms[0],
                    "trees": trees,
                }
            return None

        return self.cached(f"mesh:{name}", query_api)

    @staticmethod
    def mesh_family(trees: list[str]) -> str | None:
        """Map MeSH tree numbers to a family.

        Returns
        -------
        str | None
            Family label, or None outside the Chemicals and Drugs tree.
        """
        for prefix, family in MESH_FAMILY_TREES:
            if any(tree.startswith(prefix) for tree in trees):
                return family
        if any(tree.startswith("D") for tree in trees):
            return SMALL_MOLECULE
        return None

    # -------------------------------------------------------------- PubChem
    def pubchem_compound_by_cid(self, cid: int) -> dict | None:
        """Get the formula, charge and ChEBI cross-reference of a compound.

        Returns
        -------
        dict | None
            {"cid", "formula", "charge", "chebi_id"}, or None.
        """

        def query_api() -> dict | None:
            data = self.get_json(
                f"{API_PUBCHEM}/compound/cid/{cid}/property/MolecularFormula,Charge/JSON"
            )
            if not data:
                return None
            properties = data["PropertyTable"]["Properties"][0]
            synonyms = self.get_json(f"{API_PUBCHEM}/compound/cid/{cid}/synonyms/JSON")
            info = (synonyms or {}).get("InformationList", {}).get("Information", [{}])
            chebi_id = next(
                (s for s in info[0].get("Synonym", []) if s.startswith("CHEBI:")),
                None,
            )
            return {
                "cid": cid,
                "formula": properties.get("MolecularFormula", ""),
                "charge": properties.get("Charge", 0),
                "chebi_id": chebi_id,
            }

        return self.cached(f"pubchem_cid:{cid}", query_api)

    def pubchem_cid(self, text: str, record: str = "compound") -> int | None:
        """Resolve a name to a PubChem CID.

        Parameters
        ----------
        text : str
            Entity text.
        record : str
            "compound" (standardized structures) or "substance" (deposited
            records, e.g. polymers such as cellulose).

        Returns
        -------
        int | None
            First CID, or None.
        """
        name = normalize_name(text)

        def query_api() -> int | None:
            quoted = urllib.parse.quote(name, safe="")
            data = self.get_json(f"{API_PUBCHEM}/{record}/name/{quoted}/cids/JSON")
            if not data:
                return None
            if record == "compound":
                return data["IdentifierList"]["CID"][0]
            for info in data["InformationList"]["Information"]:
                if info.get("CID"):
                    return info["CID"][0]
            return None

        return self.cached(f"pubchem_{record}:{name}", query_api)

    def pubchem_family(self, cid: int) -> str:
        """Assign a family to a PubChem compound.

        Returns
        -------
        str
            Family from the ChEBI cross-reference, else from the formula
            and charge.
        """
        compound = self.pubchem_compound_by_cid(cid)
        if compound is None:
            return SMALL_MOLECULE
        if compound["chebi_id"]:
            return self.chebi_family(compound["chebi_id"])
        formula = compound["formula"]
        if "." in formula or compound["charge"] != 0 or formula in ION_ELEMENTS:
            return ION
        return SMALL_MOLECULE


def _uniprot_gene_names(hit: dict) -> list[str]:
    """Collect the gene names and gene synonyms of a UniProt entry.

    Returns
    -------
    list[str]
        Gene names, with their original case.
    """
    names = []
    for gene in hit.get("genes", []):
        if "geneName" in gene:
            names.append(gene["geneName"]["value"])
        names.extend(synonym["value"] for synonym in gene.get("synonyms", []))
    return names


def _uniprot_protein_names(hit: dict) -> list[str]:
    """Collect the full and short protein names of a UniProt entry.

    Returns
    -------
    list[str]
        Protein names.
    """
    description = hit.get("proteinDescription", {})
    candidates = [description.get("recommendedName", {})]
    candidates += description.get("alternativeNames", [])
    candidates += description.get("submissionNames", [])
    names = []
    for candidate in candidates:
        if "fullName" in candidate:
            names.append(candidate["fullName"]["value"])
        names.extend(short["value"] for short in candidate.get("shortNames", []))
    return names


def _result(family: str, source: str, db_id: str) -> dict[str, str]:
    """Format a classification result.

    Returns
    -------
    dict[str, str]
        {"Family", "Source", "Database ID"}.
    """
    return {"Family": family, "Source": source, "Database ID": db_id}


def classify_by_shape(text: str, db: DatabaseClient) -> dict[str, str] | None:
    """Classify identifiers and sequences from the shape of the text.

    Returns
    -------
    dict[str, str] | None
        {"Family", "Source", "Database ID"}, or None.
    """
    if PDB_REGEX.match(text) and re.search(r"[A-Za-z]", text) and db.pdb_exists(text):
        return _result(PDB_CODE, "RCSB PDB", text.upper())
    if UNIPROT_REGEX.match(text) and db.uniprot_accession_exists(text):
        return _result(UNIPROT_ID, "UniProt", text.upper())
    for regex, kind in (
        (DNA_REGEX, "DNA"),
        (RNA_REGEX, "RNA"),
        (PEPTIDE_REGEX, "peptide"),
        (REPEAT_REGEX, "repeat"),
    ):
        if regex.match(text):
            return _result(SEQUENCE, f"Regex ({kind})", "")
    return None


def classify_chemical(text: str, db: DatabaseClient) -> dict[str, str] | None:
    """Classify a name with ChEBI, then PubChem compounds and substances.

    Returns
    -------
    dict[str, str] | None
        {"Family", "Source", "Database ID"}, or None.
    """
    if (term := db.chebi_term(text)) is not None:
        return _result(
            db.chebi_family(term["id"]), f"ChEBI ({term['label']})", term["id"]
        )
    for record in ("compound", "substance"):
        if (cid := db.pubchem_cid(text, record=record)) is not None:
            return _result(db.pubchem_family(cid), f"PubChem {record}", f"CID:{cid}")
    return None


def classify_mol_entity(text: str, db: DatabaseClient) -> dict[str, str]:
    """Assign a molecular family to a MOL entity.

    Parameters
    ----------
    text : str
        Entity text, with its original case.
    db : DatabaseClient
        Database client.

    Returns
    -------
    dict[str, str]
        {"Family", "Source", "Database ID"}.
    """
    text = clean_text(text)
    # 1. Identifiers and sequences.
    if (result := classify_by_shape(text, db)) is not None:
        return result
    # 2. ChEBI.
    if (term := db.chebi_term(text)) is not None:
        return _result(
            db.chebi_family(term["id"]), f"ChEBI ({term['label']})", term["id"]
        )
    # 3. Gene symbol.
    if (accession := db.uniprot_gene(text)) is not None:
        return _result(PROTEIN, "UniProt gene", accession)
    # 4. PubChem compound.
    if (cid := db.pubchem_cid(text, record="compound")) is not None:
        return _result(db.pubchem_family(cid), "PubChem compound", f"CID:{cid}")
    # 5. Protein name.
    if (accession := db.uniprot_protein_name(text)) is not None:
        return _result(PROTEIN, "UniProt protein name", accession)
    # 6. MeSH descriptor.
    if (descriptor := db.mesh_descriptor(text)) is not None and (
        family := db.mesh_family(descriptor["trees"])
    ) is not None:
        return _result(family, f"MeSH ({descriptor['heading']})", descriptor["id"])
    # 7. PubChem substance.
    if (cid := db.pubchem_cid(text, record="substance")) is not None:
        return _result(db.pubchem_family(cid), "PubChem substance", f"CID:{cid}")
    # 8. Singular of a plural name.
    is_plural = len(text) > 4 and text.endswith("s") and not text.endswith("ss")
    if is_plural and (result := classify_chemical(text[:-1], db)) is not None:
        result["Source"] += " (singular)"
        return result
    # 9. Protein name containing the text.
    if (accession := db.uniprot_protein_name(text, contained=True)) is not None:
        return _result(PROTEIN, "UniProt protein name (partial)", accession)
    return _result(UNKNOWN, "", "")


def _safe_classify(
    text: str, db: DatabaseClient, logger: "loguru.Logger"
) -> dict[str, str]:
    """Classify one entity; a database failure gives "Unknown" for it only.

    Returns
    -------
    dict[str, str]
        {"Family", "Source", "Database ID"}.
    """
    try:
        return classify_mol_entity(text, db)
    except RuntimeError as exc:
        logger.error(f"'{text}' not classified: {exc}")
        return _result(UNKNOWN, "Database error", "")


def build_mol_family_table(
    df_entities: pd.DataFrame,
    cache_path: Path | None = None,
    n_workers: int = 4,
    logger: "loguru.Logger" = loguru.logger,
) -> pd.DataFrame:
    """Build one row per unique MOL entity with its family and counts.

    Parameters
    ----------
    df_entities : pd.DataFrame
        Entities from `collect_entities` (columns `entity`, `category`,
        `json_file`).
    cache_path : Path | None
        JSON cache of the database answers. No cache when None.
    n_workers : int
        Number of entities classified in parallel.
    logger : loguru.Logger
        Logger instance.

    Returns
    -------
    pd.DataFrame
        Columns: Entity, Family, Mentions, Texts, Json files, Source,
        Database ID. Sorted by family order then decreasing mentions.
    """
    df_mol = df_entities[df_entities["category"] == "MOL"].copy()
    df_mol["text_clean"] = df_mol["entity"].map(clean_text)
    df_mol["entity_normalized"] = df_mol["text_clean"].str.lower()
    df_mol_entities = (
        df_mol.groupby("entity_normalized")
        .agg(
            # Most frequent spelling, used for case-sensitive lookups.
            spelling=("text_clean", lambda texts: texts.value_counts().index[0]),
            Mentions=("json_file", "size"),
            Texts=("json_file", "nunique"),
            json_files=("json_file", lambda files: ", ".join(sorted(set(files)))),
        )
        .reset_index()
        .rename(columns={"entity_normalized": "Entity", "json_files": "Json files"})
    )
    spellings = df_mol_entities["spelling"].tolist()
    logger.info(f"Classifying {len(spellings)} unique MOL entities with databases.")
    db = DatabaseClient(cache_path=cache_path, logger=logger)
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                executor.map(lambda text: _safe_classify(text, db, logger), spellings)
            )
    finally:
        # Keep the answers already received, even if something failed.
        db.save_cache()
    df_mol_entities = pd.concat(
        [df_mol_entities, pd.DataFrame(results, index=df_mol_entities.index)], axis=1
    )
    unknown = df_mol_entities[df_mol_entities["Family"] == UNKNOWN]
    if not unknown.empty:
        logger.warning(
            f"{len(unknown)} MOL entities found in no database: "
            f"{', '.join(unknown['Entity'])}"
        )
    df_mol_entities["Family"] = pd.Categorical(
        df_mol_entities["Family"], categories=FAMILY_ORDER, ordered=True
    )
    df_mol_entities = df_mol_entities.sort_values(
        ["Family", "Mentions", "Entity"], ascending=[True, False, True]
    ).reset_index(drop=True)
    logger.success(
        f"Classified {len(df_mol_entities)} unique MOL entities "
        f"({df_mol_entities['Mentions'].sum()} mentions)."
    )
    return df_mol_entities[
        [
            "Entity",
            "Family",
            "Mentions",
            "Texts",
            "Json files",
            "Source",
            "Database ID",
        ]
    ]


def summarize_mol_families(
    df_mol_entities: pd.DataFrame,
    n_examples: int = 3,
) -> pd.DataFrame:
    """Summarize MOL entities per family.

    Parameters
    ----------
    df_mol_entities : pd.DataFrame
        Output of `build_mol_family_table`.
    n_examples : int
        Number of most-mentioned names shown as examples.

    Returns
    -------
    pd.DataFrame
        Columns: Family, Mentions, Share of MOL, Unique names, Texts, Examples.
        Sorted by decreasing mentions.
    """
    total_mentions = df_mol_entities["Mentions"].sum()
    rows = []
    for family, group in df_mol_entities.groupby("Family", observed=True):
        group = group.sort_values("Mentions", ascending=False)
        json_files = {
            json_file
            for files in group["Json files"]
            for json_file in files.split(", ")
            if json_file
        }
        rows.append(
            {
                "Family": family,
                "Mentions": int(group["Mentions"].sum()),
                "Share of MOL": group["Mentions"].sum() / total_mentions,
                "Unique names": len(group),
                "Texts": len(json_files),
                "Examples": ", ".join(group["Entity"].head(n_examples)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Mentions", ascending=False)
        .reset_index(drop=True)
    )


def classify_mol_entities(
    annotations_path: Path,
    out_path: Path | None = None,
    cache_path: Path | None = None,
    logger: "loguru.Logger" = loguru.logger,
) -> pd.DataFrame:
    """Collect MOL entities, classify them and optionally save the table.

    Parameters
    ----------
    annotations_path : Path
        Folder containing the JSON annotation files.
    out_path : Path | None
        TSV output path. Nothing is written when None.
    cache_path : Path | None
        JSON cache of the database answers. No cache when None.
    logger : loguru.Logger
        Logger instance.

    Returns
    -------
    pd.DataFrame
        Columns: Entity, Family, Mentions, Texts, Json files, Source,
        Database ID.
    """
    df_entities = collect_entities(annotations_path, logger=logger)
    df_mol_entities = build_mol_family_table(
        df_entities, cache_path=cache_path, logger=logger
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_mol_entities.to_csv(out_path, sep="\t", index=False)
        logger.success(f"Saved MOL families in: {out_path}")
    return df_mol_entities


@click.command()
@click.option(
    "--annotations-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Folder containing the list of JSON files with annotations.",
)
@click.option(
    "--out-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path of the output TSV file. If omitted, the table is not saved.",
)
@click.option(
    "--cache-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="JSON cache of the database answers. If omitted, no cache is used.",
)
def run_cli(
    annotations_path: Path, out_path: Path | None, cache_path: Path | None
) -> None:
    """Classify MOL entities into molecular families."""
    logger = create_logger()
    logger.info("Starting MOL classification.")
    df_mol_entities = classify_mol_entities(
        annotations_path, out_path=out_path, cache_path=cache_path, logger=logger
    )
    logger.info(f"\n{summarize_mol_families(df_mol_entities).to_string()}")
    logger.success("MOL classification completed successfully!")


if __name__ == "__main__":
    run_cli()
