"""Pydantic model for the category of a MOL entity, as answered by an LLM."""

from typing import Literal

from pydantic import BaseModel, Field

# Molecular categories of MOL entities. Their order is the sort order of the
# MOL database.
Category = Literal[
    "Protein",
    "Lipid",
    "Small molecule",
    "Ion",
    "Material, polymer",
    "PDB ID",
    "Sugar",
    "Nucleic acid",
    "Sequence",
    "UniProt ID",
    "Undefined",
]

# How the category of a MOL entity was found:
# - "": not classified yet.
# - "regex_match": by its shape (PDB ID, UniProt ID or sequence).
# - "api_match": by ChEBI, UniProt or PubChem.
# - "LLM": by an LLM.
# - "manual": by a human (never changed by the script).
Curation = Literal["", "regex_match", "api_match", "LLM", "manual"]


class MoleculeCategory(BaseModel):
    """Category of a molecular entity, as answered by the LLM."""

    justification: str = Field(
        ..., description="One sentence explaining the choice, using the context."
    )
    candidate_matches: bool = Field(
        ...,
        description=(
            "True if the database candidate is the entity meant in the datasets. "
            "False if it is another thing with the same name, or if there is "
            "no candidate."
        ),
    )
    category: Category = Field(
        ..., description="Category of the entity, 'Undefined' if unsure."
    )
