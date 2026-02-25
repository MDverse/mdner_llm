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
    - a `label`: short code representing the entity type (e.g., "MOL", "FFM")
    - a `text`: exact substring extracted from the source text

    Subclasses define fixed labels for each specific entity category.
    """

    # ... is used to indicate that these fields are required
    label: str = Field(..., description="Short label identifying the entity type")
    text: str = Field(..., description="Extracted text content")


# =====================================================================
# Entity subclasses for each specific annotation type
# =====================================================================
class Molecule(Entity):
    """Entity representing a molecule, protein, lipid, peptide, or similar object."""

    label: Literal["MOL"] = Field("MOL", description="Label for molecule entities")


class SimulationTime(Entity):
    """Entity representing a simulation time duration (e.g., 50 ns, 5 ms)."""

    label: Literal["STIME"] = Field(
        "STIME", description="Label for simulation time entities"
    )


class ForceField(Entity):
    """Entity representing a force field used in the MD simulation."""

    label: Literal["FFM"] = Field("FFM", description="Label for force field entities")


class Temperature(Entity):
    """Entity representing a temperature value used in the simulation."""

    label: Literal["TEMP"] = Field("TEMP", description="Label for temperature entities")


class SoftwareName(Entity):
    """Entity representing the name of software used for simulations or analysis."""

    label: Literal["SOFTNAME"] = Field(
        "SOFTNAME", description="Label for software name entities"
    )


class SoftwareVersion(Entity):
    """Entity representing the version of a software package."""

    label: Literal["SOFTVERS"] = Field(
        "SOFTVERS", description="Label for software version entities"
    )


# =====================================================================
# Container class for all extracted entities
# =====================================================================
class ListOfEntities(BaseModel):
    """Structured list of all extracted entities."""

    entities: list[
        Molecule
        | SimulationTime
        | ForceField
        | Temperature
        | SoftwareName
        | SoftwareVersion
    ] = Field(..., description="List of recognized entities extracted from text")
