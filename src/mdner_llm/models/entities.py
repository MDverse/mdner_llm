"""Pydantic model for entities extracted by an LLM for NER tasks."""

from typing import Literal

from pydantic import BaseModel, Field


# =====================================================================
# Base entity class
# =====================================================================
class Entity(BaseModel):
    """
    Base class for all extracted entities.

    Each entity corresponds to a text span annotated by the LLM and must contain:
    - a `category`: short code representing the entity type (e.g., "MOL", "FFM")
    - a `text`: exact substring extracted from the source text

    Subclasses define fixed labels for each specific entity category.
    """

    # ... is used to indicate that these fields are required
    category: str = Field(..., description="Category identifying the entity type.")
    text: str = Field(..., description="Extracted text content.")
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for the entity extraction, between 0 and 1. "
            "1 indicates high confidence, while 0 indicates low confidence."
        ),
    )


# =====================================================================
# Entity subclasses for each specific annotation type
# =====================================================================
class Molecule(Entity):
    """Entity representing a molecule, protein, lipid, peptide, or similar object."""

    category: Literal["MOL"] = Field(
        "MOL", description="Category for molecule entities."
    )


class SimulationTime(Entity):
    """Entity representing a simulation time duration (e.g., 50 ns, 5 ms)."""

    category: Literal["STIME"] = Field(
        "STIME", description="Category for simulation time entities."
    )


class ForceFieldModel(Entity):
    """Entity representing a force field used in the MD simulation."""

    category: Literal["FFM"] = Field(
        "FFM", description="Category for force field or model entities."
    )


class SimulationTemperature(Entity):
    """Entity representing a temperature value used in the simulation."""

    category: Literal["STEMP"] = Field(
        "STEMP", description="Category for simulation temperature entities."
    )


class SoftwareName(Entity):
    """Entity representing the name of software used for simulations or analysis."""

    category: Literal["SOFTNAME"] = Field(
        "SOFTNAME", description="Category for software name entities."
    )


class SoftwareVersion(Entity):
    """Entity representing the version of a software package."""

    category: Literal["SOFTVERS"] = Field(
        "SOFTVERS", description="Category for software version entities."
    )


# =====================================================================
# Container class for all extracted entities
# =====================================================================
class ListOfEntities(BaseModel):
    """Structured list of all extracted entities."""

    entities: list[
        Molecule
        | SimulationTime
        | ForceFieldModel
        | SimulationTemperature
        | SoftwareName
        | SoftwareVersion
    ] = Field(..., description="List of recognized entities extracted from text.")


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


class ForceFieldModelNormalized(NormalizedEntity):
    """Entity representing a normalized force field used in the MD simulation."""

    category: Literal["FFM"] = Field(
        "FFM", description="Category for force field or model entities."
    )


class SimulationTemperatureNormalized(NormalizedEntity):
    """Entity representing a normalized temperature value used in the simulation."""

    category: Literal["STEMP"] = Field(
        "STEMP", description="Category for simulation temperature entities."
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
