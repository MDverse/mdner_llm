# Annotation Rules

This document outlines the guidelines for manually annotating entities in scientific texts related to molecular simulations.

## Molecule (MOL) 🧬

### Definition

This entity covers all types of molecular compounds, including simple molecules, ions, nucleic acids, proteins, lipids, sugars, polymers, and complexes.

### Rules

- Exclude whitespaces and punctuation marks around entities
- Annotate both singular and plural forms
- Include chemical formulas and abbreviations
- Include amino acid sequences
- Include any identifiers (PDB ID, UniProt ID...)
- Exclude adjectives or descriptors that modify the molecule (e.g., hydrated, charged, folded, tetrameric).
- Exclude generic terms like `protein`, `lipid`, `phospholipid`, `sugar`, `water`, `ions`...
- Exclude specific résidues or domains like `Lys-353`...

### Examples

- `sodium chloride` ✅
- `DNA` ✅
- `ethanol` ✅
- `ammonia` ✅
- `Q29537` ✅
- `Na⁺` ✅
- `lipids` 🚫 → Too generic.
- `hydrated sodium chloride` 🚫 → Only annotate `sodium chloride` ✅

## Force field and model (FFM) 🛠️

### Definition

This entity refers to any force field or molecular model used to describe the interactions between particles in a simulation. This includes all classical all-atom force fields, coarse-grained models, solvent models, and water models. Both the name and version of the force field/model are considered relevant and should be annotated when available.

### Rules

- Include water models and other specific solvent models (e.g., `TIP3P`, `SPC/E`)
- Exclude generic terms like `force field` or `model`

### Examples

- `CHARMM36` ✅
- `AMBER99SB` ✅
- `GROMOS53a6` ✅
- `GROMOS96 43A1` ✅
- `the force field` 🚫 → Too generic
- `TIP3P water` 🚫 → Annotate only `TIP3P` FFM

See also [`ffm.yaml`](ffm.yaml) for a list of force fields and models.

## Software name (SOFTNAME) ⚙️

### Definition

This entity refers to the name of any software used for molecular simulation, visualization, or analysis. It includes packages for molecular dynamics, modeling, trajectory processing, and any other computational tasks relevant to the simulation workflow.

### Rules

- Exclude generic words such as `software`, `tool`, or `program` unless they are part of the official name.
- Exclude algorithms like `SHAKE`, `RESP`...

### Examples

- `GROMACS` ✅
- `VMD` ✅
- `NAMD` ✅
- `PyMOL` ✅
- `PLUMED` ✅
- `COLVAR` ✅
- `Python` 🚫 → Too generic
- `the simulation software` 🚫 → No specific name
- `GROMACS software` 🚫 → Annotate only `GROMACS` ✅

See also [`softname.yaml`](softname.yaml) for a list of molecular dynamics software.

## Software version (SOFTVERS) 🔢

### Definition

This entity refers to the version identifier of any software used in the simulation process. It includes version numbers, release tags, or labels, regardless of formatting (e.g., numeric, date-based, semantic).

### Rules

- Must follow a corresponding **SOFTNAME** (software/tool name)
- Keep numeric and symbolic parts intact (e.g., `1.2.3-beta`)
- Include any suffixes (e.g., `2020.4`), prefixes (e.g., `v5.1.2`)

### Examples

- `v5.0` ✅
- `2020.4` ✅
- `5.1.4` ✅
- `latest version` 🚫 → No specific version provided
- `software (v. 2016.4)` 🚫 → Annotate only `v. 2016.4` ✅
- `release 2023.1` 🚫 → Annotate only `2023.1` ✅

## Simulation time (STIME) ⏱️

### Definition

This entity refers to the duration for which a production molecular dynamics simulation is run.

### Rules

- Exclude minimization or equilibration time
- If simulation time is presented as a range, repetition, or multiplier (e.g., `5 × 100`, `10–50`), annotate the entire expression if it refers to time
- The unit is not mandatory, but the context must unambiguously indicate that the number refers to a simulation time

### Examples

- `5 × 200` ✅
- `50 picoseconds` ✅
- `100 ns` ✅
- `three runs of 500 each` ✅
- `4-8 μs` ✅
- `10 to 50 ns` ✅
- `for several hours of computation` 🚫 → Computation time, not simulation time
- `10–50 replicas` 🚫 → Number of replicas, not a time duration

## Simulation temperature (TEMP) 🌡️

### Definition

This entity refers to the thermal conditions under which a simulation is conducted. It includes any explicitly stated temperature values, with or without units.

### Rules

- The unit is not mandatory, but the context must unambiguously indicate that the number refers to temperature
- Exclude surrounding words like `temperature of` or `heated to`
- Include `room temperature` as it refers to 300 K

### Examples

- `300K` ✅
- `500 degrees Celsius` ✅
- `298` ✅ (if clearly referring to temperature)
- `heated up` 🚫 → Vague, no value
- `room temperature` ✅
