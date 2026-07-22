"""Pydantic model for entities extracted by an LLM for NER tasks."""

from typing import Literal

from pydantic import BaseModel, Field

from mdner_llm.models.entities import Entity


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
    url_from_normalization: str | None = Field(
        default=None,
        description="URL to the database entry.",
    )
    molecular_type: str | None = Field(
        default=None,
        description="Type of molecule (e.g., protein, lipid, small molecule).",
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


class ForceFieldModelNormalized(NormalizedEntity):
    """Entity representing a normalized force field used in the MD simulation."""

    category: Literal["FFM"] = Field(
        "FFM", description="Category for force field or model entities."
    )


class SoftwareNameNormalized(NormalizedEntity):
    """Entity representing the normalized name of software used for simulations."""

    category: Literal["SOFTNAME"] = Field(
        "SOFTNAME", description="Category for software name entities."
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
