"""Pydantic model for entities extracted by an LLM for NER tasks."""

from typing import Literal

from pydantic import BaseModel, Field

from mdner_llm.models.entities import Entity


class Affiliation(BaseModel):
    """Organization affiliation details."""

    type: str = Field(default="Organization", description="Type of organization.")
    name: str = Field(description="Full name and address of the organization.")


class Person(BaseModel):
    """Author details from CodeMeta schema."""

    id: str = Field(description="ORCID URL or internal unique identifier.")
    type: str = Field(default="Person", description="Entity type, always 'Person'.")
    first_name: str = Field(description="First/given name of the author.")
    last_name: str = Field(description="Last/family name of the author.")
    affiliation: Affiliation | None = Field(
        default=None, description="Organization affiliation."
    )


# =====================================================================
# Normalized Entity Models
# =====================================================================
class NormalizedEntity(Entity):
    """Represents an entity with additional normalization and validation fields."""

    text_normalized: str = Field(
        ..., description="Normalized version of the extracted text."
    )
    is_hallucinated: bool = Field(
        ..., description="True if the entity text is absent from the source text."
    )


# =====================================================================
# Entity subclasses for each specific annotation type
# =====================================================================
class MoleculeNormalized(NormalizedEntity):
    """Entity representing a normalized molecule, protein, lipid, or similar object."""

    category: Literal["MOL"] = Field(
        "MOL", description="Category for molecule entities."
    )


class SimulationTimeNormalized(NormalizedEntity):
    """Entity representing a normalized simulation time duration (e.g., 50 ns, 5 ms)."""

    category: Literal["STIME"] = Field(
        "STIME", description="Category for simulation time entities."
    )
    value: float | None = Field(
        default=None,
        description="Numeric value of the simulation time.",
    )
    unit: str | None = Field(
        default=None,
        description="The time unit in a standardized format (e.g., 'ns', 'μs', 's').",
    )


class ForceFieldModelNormalized(NormalizedEntity):
    """Entity representing a normalized force field used in the MD simulation."""

    category: Literal["FFM"] = Field(
        "FFM", description="Category for force field or model entities."
    )
    tag: Literal["force field", "model", "variation"] | None = Field(
        default=None,
        description="Classification flag: either 'force field' or 'model'.",
    )
    family: str | None = Field(
        default=None,
        description="Family or category of the force field (e.g., 'AMBER', 'CHARMM').",
    )
    aliases: list[str] | None = Field(
        default=None,
        description="List of alternative names or aliases for the force field.",
    )
    resolution: Literal["all-atom", "coarse-grain", "united-atom"] | None = Field(
        default=None,
        description="The structural resolution of the force field model.",
    )
    molecular_type: str | None = Field(
        default=None,
        description="The type of molecules the force field is designed "
        "for (e.g., 'proteins', 'lipids').",
    )
    ontology_link: str | None = Field(
        default=None,
        description="URL mapping the force field entity to the MOLSIM ontology. "
        "Documentation: https://bioportal.bioontology.org/ontologies/MOLSIM",
    )
    publication_link: str | None = Field(
        default=None,
        description="URL linking to the primary publication or reference.",
    )


class SimulationTemperatureNormalized(NormalizedEntity):
    """Entity representing a normalized temperature value used in the simulation."""

    category: Literal["STEMP"] = Field(
        "STEMP", description="Category for simulation temperature entities."
    )
    value: float | None = Field(
        default=None,
        description="Numeric value of the temperature if available, otherwise None.",
    )
    unit: str | None = Field(
        default=None,
        description="The temperature unit after normalization. "
        "All input units (e.g., '°C') are converted to Kelvin ('K').",
    )


class SoftwareNameNormalized(NormalizedEntity):
    """Entity representing the normalized name of software used for simulations."""

    category: Literal["SOFTNAME"] = Field(
        "SOFTNAME", description="Category for software name entities."
    )
    name: str = Field(description="The canonical name of the software (e.g., 'amber').")
    authors: list[Person] | None = Field(
        default=None,
        description="List of authors/contributors associated with the software.",
    )
    description: str | None = Field(
        default=None, description="Brief summary or description of the software suite."
    )
    version: str | None = Field(
        default=None, description="The specific software release version."
    )
    date_last_modification: str | None = Field(
        default=None, description="The last modification or update date (YYYY-MM-DD)."
    )
    code_repository_link: str | None = Field(
        default=None,
        description="URL to the official source code repository (e.g., GitHub).",
    )
    download_url: str | None = Field(
        default=None,
        description="Direct link to download the software source package archive.",
    )
    related_link: str | None = Field(
        default=None, description="URL to the primary project website or home page."
    )
    publication_link: str | None = Field(
        default=None,
        description="DOI or URL linking to the foundational reference paper.",
    )
    license: str | None = Field(
        default=None,
        description="SPDX license identifier or URL governing the software use.",
    )
    keywords: str | list[str] | None = Field(
        default=None, description="Keywords or tags characterizing the software domain."
    )
    programming_language: list[str] | None = Field(
        default=None,
        description="List of primary programming languages used in the repository.",
    )


class SoftwareVersionNormalized(NormalizedEntity):
    """Entity representing the normalized version of a software package."""

    category: Literal["SOFTVERS"] = Field(
        "SOFTVERS", description="Category for software version entities."
    )


# =====================================================================
# Container class for all extracted entities
# =====================================================================
class ListOfEntitiesNormalized(BaseModel):
    """Structured list of all extracted and normalized entities."""

    entities: list[
        MoleculeNormalized
        | SimulationTimeNormalized
        | ForceFieldModelNormalized
        | SimulationTemperatureNormalized
        | SoftwareNameNormalized
        | SoftwareVersionNormalized,
    ] = Field(
        ...,
        description="List of recognized and normalized entities extracted from text.",
    )
