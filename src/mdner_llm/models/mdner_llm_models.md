# Output Models for LLM NER Tasks 📑

[Pydantic](https://docs.pydantic.dev/latest/) data models used to structure the output of LLM NER tasks.

This module defines strongly typed entity classes representing different kinds of annotations
extracted from scientific Molecular Dynamics (MD) texts, such as molecules, simulation times,
force fields, and software names.

Each entity subclass uses Python's `Literal` type for its `label` field to enforce strict typing.
This ensures that only the correct entity type can be assigned, preventing misclassification
(e.g., a force field entity cannot be assigned to a Molecule class). All entities share the
base fields `label` and `text`.

```mermaid
classDiagram
    class ListOfEntities {
        entities: list[Molecule | SimulationTime | ForceFieldModelName | ForceFieldModelVersion | Temperature | SoftwareName | SoftwareVersion]
    }

    class SoftwareVersion {
        label: str = 'SOFTVERS'
        text: str
    }

    class Temperature {
        label: str = 'TEMP'
        text: str
    }

    class SimulationTime {
        label: str = 'STIME'
        text: str
    }

    class Molecule {
        label: str = 'MOL'
        text: str
    }

    class SoftwareName {
        label: str = 'SOFTNAME'
        text: str
    }

    class ForceFieldModelName {
        label: str = 'FFMNAME'
        text: str
    }

    class ForceFieldModelVersion {
        label: str = 'FFMVERSION'
        text: str
    }

    class Entity {
        label: str
        text: str
    }

    ListOfEntities ..> Molecule
    ListOfEntities ..> SoftwareVersion
    ListOfEntities ..> SimulationTime
    ListOfEntities ..> Temperature
    ListOfEntities ..> SoftwareName
    ListOfEntities ..> ForceFieldModelName
    ListOfEntities ..> ForceFieldModelVersion
```

## Models

### [`ListOfEntities`](entities.py)

A Pydantic model representing a list of `Entity` instances.Each `Entity` contains:

- `label`: short code identifying the entity type (e.g., `"MOL"`, `"STIME"`).
- `text`: exact substring extracted from the source text.

Useful for storing or validating multiple entities extracted from a document.

### [`ListOfEntitiesPositions`](entities_with_positions.py)

A Pydantic model representing a list of `EntityWithPosition` instances.Each `EntityWithPosition` contains:

- `label`: short code identifying the entity type.
- `text`: exact substring.
- `start`: start index in the source text.
- `end`: end index in the source text.

Useful when you need to retain the exact location of entities for downstream tasks like highlighting.

## Purpose

These models provide a unified schema to validate, store, and manipulate NER results
produced by automated extraction pipelines.
