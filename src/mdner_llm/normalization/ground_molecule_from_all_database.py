"""Script to normalize small molecule entities across various databases."""

import re
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

API_KEGG = "https://rest.kegg.jp"
API_CHEBI = "https://www.ebi.ac.uk/chebi/backend/api/public/es_search/"
API_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
API_PDB = "https://data.rcsb.org/rest/v1/core/entry/"
API_UNIPROT = "https://rest.uniprot.org/uniprotkb/"


def get_type(entry: str) -> str:
    """Determine the molecular entity type based on regex pattern.

    Parameters
    ----------
    entry : str
        The molecular identifier string to classify.

    Returns
    -------
    str
        The entity type: "PDB", "UNIPROT", "DNA", "RNA", "PROTEIN",
        or "SMALL_MOLECULE" for anything else.
    """
    entry = entry.replace("`", "'")
    entry = entry.replace("\u2019", "'")
    logger.info(entry)

    # PDB codes are 4 characters starting with a number
    if re.search(r"^[1-9]([a-z]|[1-9]){3}$", entry) is not None:
        return "PDB"

    # UniProt accession pattern matching
    if (
        re.search(
            r"[opq][0-9][a-z0-9]{3}[0-9]|[a-nr-z][0-9]([a-z][a-z0-9]{2}[0-9]){1,2}",
            entry,
        )
        is not None
    ):
        return "UNIPROT"

    # DNA sequence pattern (only a, t, c, g)
    if re.search(r"^5'-[atcg]+-3'$", entry) is not None:
        return "DNA"

    # RNA sequence pattern
    if re.search(r"^5'-[aucg]+-3'$", entry) is not None:
        return "RNA"

    # Amino acid sequence pattern
    if (
        re.search(r"^(?![ACGT]+$)[ACDEFGHIKLMNPQRSTVWY]{20,}$", entry) is not None
        and len(entry) > 4
    ):
        return "PROTEIN"

    return "SMALL_MOLECULE"


def call_pdb(code_pdb: str) -> dict:
    """Query the Protein Data Bank API for a given PDB code.

    Parameters
    ----------
    code_pdb : str
        4-character PDB identifier code.

    Returns
    -------
    dict
        Details retrieved from the PDB database (entry_id, pubmed_id, doi, name).
    """
    logger.info(f"Searching for `{code_pdb}` in PDB database...")
    try:
        response = httpx.get(f"{API_PDB}{code_pdb}", timeout=200)
        if response is not None and response.status_code == 200:
            results = response.json()
            logger.success(f"PDB grounding successful for `{code_pdb}`.")
            return {
                "entity_name": code_pdb,
                "type": "PDB",
                "id": results.get("rcsb_id", "Not Available"),
                "name": results.get("struct", {}).get("title", "Not Available"),
            }
        status = response.status_code if response is not None else "network error"
        logger.warning(
            f"Failed to ground `{code_pdb}` in PDB database (HTTP {status})."
        )
        return {
            "entity_name": code_pdb,
            "type": "PDB",
            "id": status,
            "name": status,
        }
    except httpx.RemoteProtocolError as e:
        logger.warning(f"RemoteProtocolError on {f'{API_PDB}{code_pdb}'}: {e}")
        return {
            "entity_name": code_pdb,
            "type": "PDB",
            "error": "RemoteProtocolError",
        }
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout on {f'{API_PDB}{code_pdb}'}: {e}")
        return {
            "entity_name": code_pdb,
            "type": "PDB",
            "error": "TimeoutException",
        }
    except httpx.RequestError as e:
        logger.warning(f"Request error on {f'{API_PDB}{code_pdb}'}: {e}")
        return {
            "entity_name": code_pdb,
            "type": "PDB",
            "error": "RequestError",
        }


def call_uniprot(code_uniprot: str) -> dict:
    """Query the UniProt API for a given UniProt accession code.

    Parameters
    ----------
    code_uniprot : str
        UniProt accession identifier.

    Returns
    -------
    dict
        Details retrieved from the UniProt database (accession, id, gene_name).
    """
    logger.info(f"Searching for `{code_uniprot}` in Uniprot database...")
    try:
        response = httpx.get(f"{API_UNIPROT}{code_uniprot}.json", timeout=200)
        if response is not None and response.status_code == 200:
            results = response.json()
            logger.success(f"UniProt grounding successful for `{code_uniprot}`.")
            return {
                "entity_name": code_uniprot,
                "type": "UNIPROT",
                "id": results.get("primaryAccession"),
                "name": results.get("genes", [{}])[0].get("geneName", {}).get("value"),
            }
        status = response.status_code if response is not None else "network error"
        logger.warning(
            f"Failed to ground `{code_uniprot}` in UniProt database (HTTP {status})."
        )
        return {
            "entity_name": code_uniprot,
            "type": "UNIPROT",
            "id": status,
            "name": status,
        }
    except httpx.RemoteProtocolError as e:
        logger.warning(f"RemoteProtocolError on {f'{API_UNIPROT}{code_uniprot}'}: {e}")
        return {
            "entity_name": code_uniprot,
            "type": "UNIPROT",
            "error": "RemoteProtocolError",
        }
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout on {f'{API_UNIPROT}{code_uniprot}'}: {e}")
        return {
            "entity_name": code_uniprot,
            "type": "UNIPROT",
            "error": "TimeoutException",
        }
    except httpx.RequestError as e:
        logger.warning(f"Request error on {f'{API_UNIPROT}{code_uniprot}'}: {e}")
        return {
            "entity_name": code_uniprot,
            "type": "UNIPROT",
            "error": "RequestError",
        }


def filter_molecules(molecules: list[str]) -> tuple[list[str], list[dict]]:
    """Split molecules into small molecules and other entity types.

    Small molecules are passed to the ChEBI/PubChem/KEGG pipeline.
    PDB, UNIPROT, DNA, RNA, PROTEIN entities are queried and saved separately.

    Parameters
    ----------
    molecules : list[str]
        The full list of molecule entity names.

    Returns
    -------
    tuple[list[str], list[dict]]
        - list of small molecule names to pass to the chemical pipeline
        - list of dicts for non-small-molecule entities (for TSV output)
    """
    # Only contains mol entities that will go throught a database consensus
    small_molecules = []
    # Contains PDB, Uniprot, and sequences
    other_entities = []
    logger.info(f"Filtering {len(molecules)} molecules ...")
    for mol in molecules:
        entity_type = get_type(mol)
        if entity_type == "SMALL_MOLECULE":
            small_molecules.append(mol)
        elif entity_type == "PDB":
            result = call_pdb(mol)
            other_entities.append(result)
        elif entity_type == "UNIPROT":
            result = call_uniprot(mol)
            other_entities.append(result)
        else:
            other_entities.append(
                {
                    "entity_name": mol,
                    "type": entity_type,
                    "id": None,
                    "name": None,
                }
            )

    logger.info(
        f"Filtered {len(small_molecules)} small molecules and {len(other_entities)}"
        f"other entities."
    )
    return small_molecules, other_entities


def save_pdb_uniprot_seq_entities(
    other_entities: list[dict],
    output_file: Path,
) -> None:
    """Save PDB, UniProt, DNA, RNA, and PROTEIN entities in a TSV file.

    Parameters
    ----------
    other_entities : list[dict]
        List of dicts with keys: entity_name, type, id, name.
    output_file : Path
        The path to the output TSV file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as file:
        file.write("Molecule\tType\tID\tName\n")
        for entity in other_entities:
            file.writelines(
                f"{entity['entity_name']}\t{entity['type']}\t{entity['id']}\t{entity['name']}\n"
            )

    logger.info(
        f"Saved {len(other_entities)} non-small-molecule entities to {output_file}."
    )


def query_kegg_by_name(
    entity_name: str,
) -> tuple[str, str] | tuple[None, None] | tuple[str, None] | tuple[None, str]:
    """Return the KEGG ID linked to the molecule name.

    Parameters
    ----------
    entity_name : str
        The name of the molecule to query.

    Returns
    -------
    tuple[str, str] | tuple[None, None] | tuple[str, None] | tuple[None, str]
        A tuple containing the pubchem ID and chebi ID returned by kegg.
    """
    logger.info(f"KEGG: searching {entity_name} directly")
    try:
        kegg_response = httpx.get(f"{API_KEGG}/find/compound/{entity_name}")
        if kegg_response is None or kegg_response.status_code != 200:
            logger.warning(f"KEGG: Failed to retrieve KEGG ID for {entity_name}")
            return None, None
        kegg_text = kegg_response.text.strip()
        if not kegg_text:
            logger.warning(f"KEGG: No KEGG entry found for {entity_name}")
            return None, None
        kegg_id = kegg_text.split("\t")[0].strip()
        kegg_id = kegg_id.replace("cpd:", "").split(";")[0].strip()
        logger.info(f"KEGG: Extracted KEGG ID {kegg_id}")

        logger.info(f"KEGG: Converting KEGG ID {kegg_id} to a pubchem id")
        try:
            pubchem_response = httpx.get(f"{API_KEGG}/conv/pubchem/cpd:{kegg_id}")
            if pubchem_response is None or pubchem_response.status_code != 200:
                logger.warning(f"KEGG: Failed to retrieve PubChem ID for {entity_name}")
                return None, None
            pubchem_parts = pubchem_response.text.strip().split("\t")
            logger.info(f"KEGG: Extracted Pubchem parts : {pubchem_parts}")
            if len(pubchem_parts) < 2:
                logger.warning(f"KEGG: No PubChem mapping in KEGG for {entity_name}")
                return None, None
            pubchem_id_from_kegg = (
                pubchem_parts[1].split("\n")[0].replace("pubchem:", "").strip()
            )
            logger.info(
                f"KEGG: Converted KEGG ID {kegg_id} to Pubchem {pubchem_id_from_kegg}"
            )
            logger.info(f"KEGG: Converting KEGG ID {kegg_id} to ChEBI ID")
            try:
                chebi_response = httpx.get(f"{API_KEGG}/conv/chebi/cpd:{kegg_id}")
                if chebi_response is None or chebi_response.status_code != 200:
                    logger.warning(
                        f"KEGG: Failed to retrieve CHEBI ID for {entity_name}"
                    )
                    return None, None
                chebi_parts = chebi_response.text.strip().split("\t")
                logger.info(f"KEGG: Extracted ChEBI parts : {chebi_parts}")
                if len(chebi_parts) < 2:
                    logger.warning(f"KEGG: No ChEBI mapping in KEGG for {entity_name}")
                    return pubchem_id_from_kegg, None
                chebi_id_from_kegg = (
                    chebi_parts[1].split("\n")[0].replace("chebi:", "").strip()
                )
                logger.info(
                    f"KEGG: Converted KEGG ID {kegg_id} to {chebi_id_from_kegg}"
                )
                return pubchem_id_from_kegg, chebi_id_from_kegg
            except httpx.RemoteProtocolError as e:
                logger.warning(
                    f"RemoteProtocolError on {API_KEGG} for {entity_name}: {e}"
                )
                return None, None
            except httpx.TimeoutException as e:
                logger.warning(f"Timeout on {API_KEGG} for {entity_name}: {e}")
                return None, None
            except httpx.RequestError as e:
                logger.warning(f"Request error on {API_KEGG} for {entity_name}: {e}")
                return None, None
        except httpx.RemoteProtocolError as e:
            logger.warning(f"RemoteProtocolError on {API_KEGG} for {entity_name}: {e}")
            return None, None
        except httpx.TimeoutException as e:
            logger.warning(f"Timeout on {API_KEGG} for {entity_name}: {e}")
            return None, None
        except httpx.RequestError as e:
            logger.warning(f"Request error on {API_KEGG} for {entity_name}: {e}")
            return None, None
    except httpx.RemoteProtocolError as e:
        logger.warning(f"RemoteProtocolError on {API_KEGG} for {entity_name}: {e}")
        return None, None
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout on {API_KEGG} for {entity_name}: {e}")
        return None, None
    except httpx.RequestError as e:
        logger.warning(f"Request error on {API_KEGG} for {entity_name}: {e}")
        return None, None


def query_chebi_by_name(entity_name: str) -> str | None:
    """Return the CHEBI ID linked to the molecule name.

    Parameters
    ----------
    entity_name : str
        The name of the molecule to query.

    Returns
    -------
    str | None
        The CHEBI ID returned by CHEBI.
    """
    logger.info(f"ChEBI: searching {entity_name}")
    try:
        chebi_response = httpx.get(API_CHEBI, params={"term": entity_name}, timeout=200)
        if chebi_response is not None and chebi_response.status_code == 200:
            results = chebi_response.json().get("results", [])
            if not results:
                logger.warning(f"ChEBI: No ChEBI entry found for {entity_name}")
                return None
            chebi_id = results[0]["_id"]
            logger.info(f"ChEBI: Found CHEBI ID for {chebi_id}")
            return chebi_id
        logger.warning(f"ChEBI: Failed to retrieve CHEBI ID for {entity_name}")
        return None
    except httpx.RemoteProtocolError as e:
        logger.warning(f"RemoteProtocolError on {API_CHEBI} for {entity_name}: {e}")
        return None
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout on {API_CHEBI} for {entity_name}: {e}")
        return None
    except httpx.RequestError as e:
        logger.warning(f"Request error on {API_CHEBI} for {entity_name}: {e}")
        return None


def query_pubchem_by_name(entity_name: str) -> str | None:
    """Return the PubChem compound ID linked to the molecule name.

    Parameters
    ----------
    entity_name : str
        The name of the molecule to query.

    Returns
    -------
    str | None
        The PubChem compound ID returned by PubChem.
    """
    logger.info(f"Pubchem: searching {entity_name}")
    try:
        response = httpx.get(f"{API_PUBCHEM}/compound/name/{entity_name}/JSON")
        if response is not None and response.status_code == 200:
            compounds = response.json().get("PC_Compounds", [])
            if not compounds:
                logger.warning(f"PubChem: No compound found for {entity_name}")
                return None
            pubchem_id = str(compounds[0]["id"]["id"]["cid"])
            logger.info(f"PubChem: Found Pubchem ID {pubchem_id}")
            return pubchem_id
        logger.warning(f"PubChem: Failed to retrieve PubChem ID for {entity_name}")
        return None
    except httpx.RemoteProtocolError as e:
        logger.warning(f"RemoteProtocolError on {API_PUBCHEM} for {entity_name}: {e}")
        return None
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout on {API_PUBCHEM} for {entity_name}: {e}")
        return None
    except httpx.RequestError as e:
        logger.warning(f"Request error on {API_PUBCHEM} for {entity_name}: {e}")
        return None


def query_pubchem_by_substance(substance_id: str) -> str | None:
    """Return the PubChem compound ID linked to the substance ID.

    Parameters
    ----------
    substance_id : str
        The substance ID to query.

    Returns
    -------
    str | None
        The PubChem compound ID returned by PubChem.
    """
    logger.info(f"Pubchem Substance: searching pubchem compound ID from {substance_id}")
    try:
        response = httpx.get(f"{API_PUBCHEM}/substance/sid/{substance_id}/cids/JSON")
        if response is not None and response.status_code == 200:
            information = (
                response.json().get("InformationList", {}).get("Information", [])
            )
            if not information:
                logger.warning(
                    f"Pubchem Substance: No CID found for substance {substance_id}"
                )
                return None
            pubchem_id_from_substance = str(information[0]["CID"])
            logger.info(
                f"Pubchem Substance: Found Pubchem compound ID"
                f"{pubchem_id_from_substance}"
            )
            return pubchem_id_from_substance
        logger.warning(
            f"Pubchem Substance: Failed to retrieve PubChem compound ID for"
            f"{substance_id}"
        )
        return None
    except httpx.RemoteProtocolError as e:
        logger.warning(f"RemoteProtocolError on {API_PUBCHEM} for {substance_id}: {e}")
        return None
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout on {API_PUBCHEM} for {substance_id}: {e}")
        return None
    except httpx.RequestError as e:
        logger.warning(f"Request error on {API_PUBCHEM} for {substance_id}: {e}")
        return None


def query_pubchem_substance_by_name(entity_name: str) -> str | None:
    """Return the PubChem substance ID linked to the molecule name.

    Parameters
    ----------
    entity_name : str
        The name of the molecule to query.

    Returns
    -------
    str | None
        The first PubChem substance ID found, or None.
    """
    logger.info(f"PubChem Substance: Searching {entity_name}")
    try:
        response = httpx.get(f"{API_PUBCHEM}/substance/name/{entity_name}/JSON")
        if response is None or response.status_code != 200:
            logger.warning(
                f"PubChem Substance: Failed to retrieve PubChem substance for"
                f"{entity_name}"
            )
            return None
        substances = response.json().get("PC_Substances", [])
        if not substances:
            logger.warning(
                f"PubChem Substance: No PubChem substance found for {entity_name}"
            )
            return None
        sid = substances[0].get("sid", {}).get("id")
        if sid is None:
            logger.warning(
                f"PubChem Substance: No SID found in PubChem substance response for"
                f" {entity_name}"
            )
            return None
        sid = str(sid)
        logger.info(f"PubChem Substance: Found Pubchem SID {sid} for {entity_name}")
        return sid
    except httpx.RemoteProtocolError as e:
        logger.warning(f"RemoteProtocolError on {API_PUBCHEM} for {entity_name}: {e}")
        return None
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout on {API_PUBCHEM} for {entity_name}: {e}")
        return None
    except httpx.RequestError as e:
        logger.warning(f"Request error on {API_PUBCHEM} for {entity_name}: {e}")
        return None


def get_chebi_id_from_pubchem_synonyms(pubchem_id: str) -> str | None:
    """Return the CHEBI ID linked to the PubChem compound ID via synonyms.

    Parameters
    ----------
    pubchem_id : str
        The PubChem compound ID to query.

    Returns
    -------
    str | None
        The CHEBI ID found in PubChem synonyms.
    """
    logger.info(f"Pubchem: searching ChEBI synonyms from {pubchem_id}")
    try:
        response = httpx.get(
            f"{API_PUBCHEM}/compound/cid/{pubchem_id}/synonyms/JSON", timeout=200
        )
        if response is not None and response.status_code == 200:
            synonyms = response.json()["InformationList"]["Information"][0]["Synonym"]
            for synonym in synonyms:
                if synonym.startswith("CHEBI:"):
                    chebi_id_synonyms = synonym.replace("CHEBI:", "")
                    logger.info(
                        f"Pubchem: Found ChEBI ID {chebi_id_synonyms} from {pubchem_id}"
                    )
                    return chebi_id_synonyms
            logger.warning(
                f"Pubchem: No CHEBI ID found in synonyms for PubChem ID {pubchem_id}"
            )
            return None
        logger.warning(
            f"Pubchem: Failed to retrieve synonyms for PubChem ID {pubchem_id}"
        )
        return None
    except httpx.RemoteProtocolError as e:
        logger.warning(f"RemoteProtocolError on {API_PUBCHEM} for {pubchem_id}: {e}")
        return None
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout on {API_PUBCHEM} for {pubchem_id}: {e}")
        return None
    except httpx.RequestError as e:
        logger.warning(f"Request error on {API_PUBCHEM} for {pubchem_id}: {e}")
        return None


def get_chebi_id_from_pubchem_synonyms_from_sid(sid: str) -> str | None:
    """Return the CHEBI ID found in the synonyms of a PubChem substance SID.

    Parameters
    ----------
    sid : str
        The PubChem substance ID to query.

    Returns
    -------
    str | None
        The CHEBI ID found in the substance synonyms, or None.
    """
    logger.info(f"Pubchem substance: Searching ChEBI ID from {sid}")
    try:
        response = httpx.get(f"{API_PUBCHEM}/substance/sid/{sid}/JSON")
        if response is None or response.status_code != 200:
            logger.warning(
                f"Pubchem substance: Failed to retrieve substance data for SID {sid}"
            )
            return None
        substances = response.json().get("PC_Substances", [])
        for substance in substances:
            if substance.get("sid", {}).get("id") == int(sid):
                synonyms = substance.get("synonyms", [])
                for synonym in synonyms:
                    if synonym.startswith("CHEBI:"):
                        chebi_id = synonym.replace("CHEBI:", "")
                        logger.info(
                            f"Pubchem substance: Found ChEBI ID in pubchem substance"
                            f" synonyms of{chebi_id}"
                        )
                        return chebi_id
                logger.warning(
                    f"Pubchem substance: No CHEBI ID found in synonyms for SID {sid}"
                )
                return None
        logger.warning(f"Pubchem substance: SID {sid} not found in PubChem response")
        return None
    except httpx.RemoteProtocolError as e:
        logger.warning(f"RemoteProtocolError on {API_CHEBI} for {sid}: {e}")
        return None
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout on {API_CHEBI} for {sid}: {e}")
        return None
    except httpx.RequestError as e:
        logger.warning(f"Request error on {API_CHEBI} for {sid}: {e}")
        return None


def get_pubchem_cid_from_substance(sid: str) -> str | None:
    """Return the PubChem compound ID linked to a substance ID via the compound field.

    Parameters
    ----------
    sid : str
        The PubChem substance ID to query.

    Returns
    -------
    str | None
        The PubChem compound ID found in the substance's compound field.
    """
    logger.info(
        f"Pubchem substance: Searching Pubchem compound ID from substance id : {sid} "
    )
    try:
        response = httpx.get(f"{API_PUBCHEM}/substance/sid/{sid}/JSON", timeout=200)
        if response is None or response.status_code != 200:
            logger.warning(
                f"Pubchem substance: Failed to retrieve substance data for SID {sid}"
            )
            return None

        substances = response.json().get("PC_Substances", [])
        for substance in substances:
            if substance.get("sid", {}).get("id") == int(sid):
                compound_list = substance.get("compound", [])
                for entry in compound_list:
                    cid = entry.get("id", {}).get("id", {}).get("cid")
                    if cid is not None:
                        cid = str(cid)
                        logger.info(
                            f"Pubchem substance: Found CID {cid} from substance ID{sid}"
                        )
                        return cid
                logger.warning(
                    f"Pubchem substance: No CID found in compound field for SID {sid}"
                )
                return None

        logger.warning(f"Pubchem substance: SID {sid} not found in PubChem response")
        return None
    except httpx.RemoteProtocolError as e:
        logger.warning(f"RemoteProtocolError on {API_PUBCHEM} for {sid}: {e}")
        return None
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout on {API_PUBCHEM} for {sid}: {e}")
        return None
    except httpx.RequestError as e:
        logger.warning(f"Request error on {API_PUBCHEM} for {sid}: {e}")
        return None


def get_synonym_from_pubchem_substance(sid: str) -> str | None:
    """Return the first synonym of a PubChem substance.

    Parameters
    ----------
    sid : str
        The PubChem substance ID to query.

    Returns
    -------
    str | None
        The first synonym found for the substance.
    """
    logger.info(
        f"Pubchem substance: Searching for the first synonym of pubchem substance {sid}"
    )
    try:
        response = httpx.get(f"{API_PUBCHEM}/substance/sid/{sid}/JSON")
        if response is None or response.status_code != 200:
            logger.warning(
                f"Pubchem substance: Failed to retrieve substance data for SID {sid}"
            )
            return None

        substances = response.json().get("PC_Substances", [])
        for substance in substances:
            if substance.get("sid", {}).get("id") == int(sid):
                synonyms = substance.get("synonyms", [])
                if not synonyms:
                    logger.warning(
                        f"Pubchem substance: No synonyms found for SID {sid}"
                    )
                    return None
                logger.info(
                    f"Pubchem substance: Found {synonyms[0]} as {sid} first synonym "
                )
                return synonyms[0]

        logger.warning(f"Pubchem substance: SID {sid} not found in PubChem response")
        return None
    except httpx.RemoteProtocolError as e:
        logger.warning(f"RemoteProtocolError on {API_PUBCHEM} for {sid}: {e}")
        return None
    except httpx.TimeoutException as e:
        logger.warning(f"Timeout on {API_PUBCHEM} for {sid}: {e}")
        return None
    except httpx.RequestError as e:
        logger.warning(f"Request error on {API_PUBCHEM} for {sid}: {e}")
        return None


def get_compound_id_from_kegg_substance(sid: str) -> str | None:
    """Return the PubChem compound ID from a KEGG substance SID using fallbacks.

    First tries to find a CID directly in the substance's compound field.
    If not found, tries the direct SID to CID mapping via query_pubchem_by_substance.
    If still not found, takes the first synonym and searches PubChem compound by name.

    Parameters
    ----------
    sid : str
        The PubChem substance ID retrieved from KEGG.

    Returns
    -------
    str | None
        The PubChem compound ID found, or None if all methods fail.
    """
    cid = get_pubchem_cid_from_substance(sid)
    logger.info(f"Lookig for pubchem compound ID from pubchem substance {sid}")

    if cid is not None:
        logger.info(f"CID {cid} found in compound field for SID {sid}")
        return cid
    logger.warning("None Pubchem compound ID found")
    logger.info(f"Retry: Lookig for pubchem compound ID from pubchem substance {sid}")
    cid = query_pubchem_by_substance(sid)
    if cid is not None:
        logger.info(f"CID {cid} found via direct SID mapping for SID {sid}")
        return cid
    logger.info(f"Retry: Searching synonyms of substance {sid}")
    synonym = get_synonym_from_pubchem_substance(sid)
    if synonym is None:
        logger.warning(f"No synonym found fallback possible for SID {sid}")
        return None
    logger.success(f"Found first synonym '{synonym}' for SID {sid}")
    logger.info(f"Retry: Searching pubchem ID for : {synonym} ")
    cid = query_pubchem_by_name(synonym)
    if cid is not None:
        logger.success(f"CID {cid} found via synonym '{synonym}' for SID {sid}")
        return cid

    logger.warning(f"All methods failed to find a CID for SID {sid}")
    return None


def _chebi_from_pubchem_cid(pubchem_id: str) -> str | None:
    """Return the ChEBI ID from a PubChem CID via compound synonyms.

    Parameters
    ----------
    pubchem_id : str
        The PubChem compound ID to query.

    Returns
    -------
    str | None
        The ChEBI ID found in synonyms, or None.
    """
    return get_chebi_id_from_pubchem_synonyms(pubchem_id)


def _chebi_via_pubchem_substance(mol: str) -> tuple[str | None, str | None]:
    """Fallback PubChem Substance path when no direct CID is available.

    Attempts in order:
    1. Finds a SID from the molecule name  searches SID synonyms for CHEBI:XXX.
    2. If no ChEBI in SID synonyms, resolves SID →CID via:
       a. compound field of the SID
       b. /substance/sid/{sid}/cids endpoint
       c. first synonym of the SID  compound name search
    Returns (chebi_id_found, pubchem_cid_resolved).

    Parameters
    ----------
    mol : str
        The molecule name to query.

    Returns
    -------
    tuple[str | None, str | None]
        (chebi_id, resolved_pubchem_cid) either may be None.
    """
    logger.info(f"looking for pubchem substance for {mol}")
    sid = query_pubchem_substance_by_name(mol)
    if sid is None:
        logger.warning(f"No SID PubChem found for {mol}")
        return None, None

    logger.success(f"Found pubchem substance ID: {sid} for {mol}")

    logger.info("looking for ChEBI ID in substance synonyms")
    chebi_from_sid = get_chebi_id_from_pubchem_synonyms_from_sid(sid)
    if chebi_from_sid is not None:
        logger.success(
            f"Found ChEBI ID: {chebi_from_sid} in SID synonyms: {chebi_from_sid}"
        )
        return chebi_from_sid, None

    logger.warning("No ChEBI ID found in SID synonymes")
    logger.info("Trying to convert SID to CID ")
    cid = get_compound_id_from_kegg_substance(sid)
    if cid is None:
        logger.warning(f"No Pubchem compound found from {sid}")
        return None, None

    logger.success(f"Found compound ID from substance ID: {cid}")
    logger.info(f"Searching for ChEBI ID from CID {cid}")
    chebi_from_cid = _chebi_from_pubchem_cid(cid)
    return chebi_from_cid, cid


def get_chebi_id_for_molecule(mol: str) -> tuple[str | None, str | None, str | None]:
    """Resolve ChEBI IDs for a molecule from ChEBI, PubChem, and KEGG.

    Covers the following paths:
    - Direct ChEBI search by name.
    - PubChem Compound: name then CID then compound synonyms for CHEBI:XXX.
    - PubChem Substance fallback (when no CID): name then SID then SID synonyms,
      then SID then CID (compound field / cids endpoint / first synonym) then
      compound synonyms.
    - KEGG: name then KEGG ChEBI ID directly.
    - KEGG PubChem CID (if different from direct): CID then compound synonyms.
    - KEGG SID fallback: if KEGG returns a substance ID, resolve to CID then synonyms.

    Parameters
    ----------
    mol : str
        The molecule entity name.

    Returns
    -------
    tuple[str | None, str | None, str | None]
        (chebi_id_direct, chebi_id_from_pubchem, chebi_id_from_kegg)
    """
    logger.info("Looking for direct Chebi and Pubchem ID")
    chebi_id = query_chebi_by_name(mol)
    logger.info(f"direct ChEBI ID: {chebi_id!r}")

    pubchem_cid = query_pubchem_by_name(mol)
    logger.info(f"direct PubChem CID: {pubchem_cid!r}")

    if pubchem_cid is not None:
        chebi_from_pubchem = _chebi_from_pubchem_cid(pubchem_cid)
        logger.success(f"Found ChEBI via CID ({pubchem_cid}): {chebi_from_pubchem!r}")
    else:
        logger.warning("No pubchem CID found retrying via Substance...")
        chebi_from_pubchem, pubchem_cid = _chebi_via_pubchem_substance(mol)
        logger.info(f"ChEBI via Substance: {chebi_from_pubchem}")

    pubchem_id_from_kegg, chebi_from_kegg = query_kegg_by_name(mol)
    logger.info(
        f"  → PubChem via KEGG: {pubchem_id_from_kegg!r}, ChEBI via KEGG: "
        f"{chebi_from_kegg!r}"
    )

    if pubchem_id_from_kegg is not None and pubchem_id_from_kegg != pubchem_cid:
        logger.info(
            "Different CID between KEGG and direct pubchem , loonking in ChEBI with "
            "this CID..."
        )
        chebi_via_kegg_cid = _chebi_from_pubchem_cid(pubchem_id_from_kegg)
        if chebi_via_kegg_cid is not None:
            logger.info(f"ChEBI via CID KEGG: {chebi_via_kegg_cid!r}")
            chebi_from_kegg = chebi_via_kegg_cid
        else:
            logger.info(
                f"No ChEBI via CID KEGG, searching for compound from SID KEGG: "
                f"{pubchem_id_from_kegg}"
            )
            resolved_cid = get_compound_id_from_kegg_substance(pubchem_id_from_kegg)
            if resolved_cid is not None:
                chebi_from_kegg = _chebi_from_pubchem_cid(resolved_cid)
                logger.success(
                    f"Found ChEBI ID via SID KEGG {resolved_cid}: {chebi_from_kegg}"
                )

    return chebi_id, chebi_from_pubchem, chebi_from_kegg


def compare_chebi_ids(
    chebi_id: str | None,
    chebi_id_from_kegg: str | None,
    chebi_id_from_pubchem: str | None,
    mol: str,
) -> bool:
    """Compare CHEBI IDs from ChEBI, KEGG and PubChem.

    Parameters
    ----------
    chebi_id : str | None
        The CHEBI ID retrieved directly from ChEBI.
    chebi_id_from_kegg : str | None
        The CHEBI ID retrieved via KEGG.
    chebi_id_from_pubchem : str | None
        The CHEBI ID retrieved via PubChem synonyms.
    mol : str
        The molecule entity name (used for logging).

    Returns
    -------
    bool
        True if at least 2 CHEBI IDs match, False otherwise.
    """
    if (
        (
            chebi_id is not None
            and chebi_id_from_pubchem is not None
            and chebi_id == chebi_id_from_pubchem
        )
        or (
            chebi_id is not None
            and chebi_id_from_kegg is not None
            and chebi_id == chebi_id_from_kegg
        )
        or (
            chebi_id_from_pubchem is not None
            and chebi_id_from_kegg is not None
            and chebi_id_from_pubchem == chebi_id_from_kegg
        )
    ):
        logger.info(f"CHEBI ID for {mol} is the same in at least 2 databases.")
        return True
    logger.warning(f"CHEBI ID for {mol} is different across databases.")
    return False


def compare_pubchem_ids(
    pubchem_id: str | None,
    pubchem_id_from_kegg: str | None,
    mol: str,
) -> bool:
    """Compare PubChem IDs from direct query and KEGG.

    Parameters
    ----------
    pubchem_id : str | None
        The PubChem ID retrieved directly from PubChem.
    pubchem_id_from_kegg : str | None
        The PubChem ID retrieved via KEGG.
    mol : str
        The molecule entity name (used for logging).

    Returns
    -------
    bool
        True if the PubChem IDs match, False otherwise.
    """
    if pubchem_id is None or pubchem_id_from_kegg is None:
        logger.warning(f"PubChem ID missing for {mol}, cannot compare.")
        return False

    if pubchem_id == pubchem_id_from_kegg:
        logger.info(f"PubChem ID for {mol} matches directly.")
        return True

    pubchem_id_from_kegg_compound = get_compound_id_from_kegg_substance(
        pubchem_id_from_kegg
    )
    if (
        pubchem_id_from_kegg_compound is not None
        and pubchem_id_from_kegg_compound == pubchem_id
    ):
        logger.info(f"PubChem ID for {mol} matches after resolving KEGG substance.")
        return True

    logger.warning(f"PubChem ID for {mol} does not match across sources.")
    return False


def save_chebi_comparaison_in_tsv(
    molecules: list[str],
    output_file: Path,
) -> None:
    """Save the ChEBI ID comparison results in a TSV file.

    Parameters
    ----------
    molecules : list[str]
        The list of molecule entity names.
    output_file : Path
        The path to the output TSV file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(
            "Molecule\tCHEBI_ID\tCHEBI_ID_from_KEGG\tCHEBI_ID_from_PubChem\tMatch\n"
        )

        for index, mol in enumerate(molecules, start=1):
            logger.info("=" * 50)
            logger.info(f"Processing molecule {index}/{len(molecules)}: '{mol}'")

            chebi_id, chebi_from_pubchem, chebi_from_kegg = get_chebi_id_for_molecule(
                mol
            )

            match = compare_chebi_ids(
                chebi_id, chebi_from_kegg, chebi_from_pubchem, mol
            )
            logger.info(f"  → Match (au moins 2 IDs identiques): {match}")

            line = (
                f"{mol}\t{chebi_id}\t{chebi_from_kegg}\t{chebi_from_pubchem}\t{match}\n"
            )
            file.write(line)

    logger.info("=" * 50)
    logger.info(f"Fichier sauvegardé: {output_file}")


def get_no_chebi_match(chebi_comparaison_file: Path) -> list[str]:
    """Extract molecule names with no CHEBI ID match from the comparison file.

    Parameters
    ----------
    chebi_comparaison_file : Path
        The path to the CHEBI comparison TSV file.

    Returns
    -------
    list[str]
        A list of molecule names that have no CHEBI ID match across databases.
    """
    df = pd.read_csv(chebi_comparaison_file, sep="\t")
    no_match_df = df[df["Match"] == False]
    return no_match_df["Molecule"].tolist()


def save_pubchem_comparaison_in_tsv(
    molecules: list[str],
    output_file: Path,
) -> None:
    """Save the PubChem ID comparison results in a TSV file.

    Parameters
    ----------
    molecules : list[str]
        The list of molecule entity names.
    output_file : Path
        The path to the output TSV file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as file:
        file.write("Molecule\tPubChem_ID\tPubChem_ID_from_KEGG\tMatch\n")
        for index, mol in enumerate(molecules, start=1):
            logger.info(f"Processing molecule {index}/{len(molecules)}: {mol}")
            pubchem_id = query_pubchem_by_name(mol)
            pubchem_id_from_kegg, _ = query_kegg_by_name(mol)

            match = compare_pubchem_ids(pubchem_id, pubchem_id_from_kegg, mol)

            file.write(f"{mol}\t{pubchem_id}\t{pubchem_id_from_kegg}\t{match}\n")


def load_molecule_entities(file_path: Path) -> list:
    """Load molecular identifiers from a file into a list.

    Parameters
    ----------
    file_path : Path
        Path to the input file containing molecular identifiers.

    Returns
    -------
    list
        A list of molecular identifiers loaded from the file.
    """
    logger.info(f"Loading MOL entities from {file_path}...")
    entities = pd.read_csv(file_path, sep="\t")
    mol_entities = entities[entities["category"] == "MOL"]
    mol_entities = list(mol_entities["entity"].unique())
    molecule_liste = []
    for molecule in mol_entities:
        if len(molecule) > 3:
            molecule_liste.append(molecule)
    molecule_liste = list(set(molecule_liste))
    logger.info(f"Loaded {len(molecule_liste)} MOL entities successfully.")
    return molecule_liste


if __name__ == "__main__":
    entities_file = Path("data/entities.tsv")
    output_dir = Path("results/ground_molecule/same_grounding_mol")

    all_molecules = load_molecule_entities(entities_file)

    small_molecules, other_entities = filter_molecules(all_molecules)

    # save_pdb_uniprot_seq_entities(
    #     other_entities=other_entities,
    #     output_file=output_dir / "pdb_uniprot_seq_entities.tsv",
    # )

    chebi_output = output_dir / "chebi_comparaison.tsv"
    # save_chebi_comparaison_in_tsv(
    #     molecules=small_molecules,
    #     output_file=chebi_output,
    # )

    # Pipeline PubChem pour les molécules sans match ChEBI
    no_chebi_match_molecules = get_no_chebi_match(
        Path("results/ground_molecule/same_grounding_mol/chebi_comparaison.tsv")
    )
    logger.info(
        f"Number of molecules with no CHEBI ID match: {len(no_chebi_match_molecules)}"
    )
    logger.info("Saving PubChem comparison for molecules with no CHEBI match...")
    save_pubchem_comparaison_in_tsv(
        molecules=no_chebi_match_molecules,
        output_file=output_dir / "pubchem_comparaison_no_chebi_match.tsv",
    )
