"""Pydantic model for entities with character positions extracted by an LLM for NER."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mdner_llm.models.entities import Entity


# =====================================================================
# Base entity class with positions
# =====================================================================
class EntityWithPosition(Entity):
    """Entity with text span positions."""

    start: int = Field(..., description="Start index of the entity in the text")
    end: int = Field(..., description="End index of the entity in the text")


# =====================================================================
# Entity subclasses for each specific annotation type with positions
# =====================================================================
class MoleculePosition(EntityWithPosition):
    """Entity representing a molecule, protein, lipid, peptide, or similar object."""

    category: Literal["MOL"] = Field("MOL")


class SimulationTimePosition(EntityWithPosition):
    """Entity representing a simulation time duration (e.g., 50 ns, 5 ms)."""

    category: Literal["STIME"] = Field("STIME")


class ForceFieldPosition(EntityWithPosition):
    """Entity representing a force field used in the MD simulation."""

    category: Literal["FFM"] = Field("FFM")


class TemperaturePosition(EntityWithPosition):
    """Entity representing a temperature value used in the simulation."""

    category: Literal["TEMP"] = Field("TEMP")


class SoftwareNamePosition(EntityWithPosition):
    """Entity representing the name of software used for simulations or analysis."""

    category: Literal["SOFTNAME"] = Field("SOFTNAME")


class SoftwareVersionPosition(EntityWithPosition):
    """Entity representing the version of a software package."""

    category: Literal["SOFTVERS"] = Field("SOFTVERS")


EntityUnion = Annotated[
    MoleculePosition
    | SimulationTimePosition
    | ForceFieldPosition
    | TemperaturePosition
    | SoftwareNamePosition
    | SoftwareVersionPosition,
    Field(discriminator="category"),
]


# =====================================================================
# Container class for all extracted entities with positions
# =====================================================================
class ListOfEntitiesPositions(BaseModel):
    """Structured list of all extracted entities with character positions."""

    entities: list[EntityUnion] = Field(
        ..., description="List of entities with positions"
    )
