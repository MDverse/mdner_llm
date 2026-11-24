"""Pydantic model for the category of a MOL entity, as answered by an LLM."""

from typing import Literal

from pydantic import BaseModel, Field

# Molecular categories of MOL entities. Their order is the sort order of the
# MOL database.
Category = Literal[
    "Protein, peptide",
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
