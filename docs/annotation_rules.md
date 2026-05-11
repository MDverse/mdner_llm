# Annotation guidelines

This document outlines the guidelines for manually annotating entities in scientific texts related to molecular simulations.

## Molecule: MOL

### Definition

The **MOL** entity covers molecular compounds, including simple molecules, ions, nucleic acids, proteins, lipids, sugars, polymers, and complexes.
MOL entities should be uniquely identifiable.

### Rules

- Exclude whitespaces and punctuation marks around entities
- Annotate both singular and plural forms
- Include chemical formulas and abbreviations
- Include amino acid and nuclei acid sequences
- Include any identifiers (PDB ID, UniProt ID...)
- Exclude adjectives or descriptors that modify the molecule (e.g., `hydrated`, `charged`, `folded`, `tetrameric`)
- Exclude generic terms like `protein`, `lipid`, `phospholipid`, `DNA`, `sugar`, `water`, `ions`...
- Exclude specific résidues or domains like `Lys-353`...

### Good examples

- `sodium chloride`
- `ethanol`
- `ammonia`
- `Q29537`
- `2RH1`
- `Na⁺`

### Bad examples

- `lipids`
- `DNA`
- `hydrated sodium chloride` (only annotate `sodium chloride`)
- `binding protein`
- `ions in solvant`

## Force field and model: FFM

### Definition

The **FFM** entity refers to any force field or molecular model used to describe the interactions between particles in a simulation.
This includes all classical all-atom force fields, coarse-grained models, solvent models, and water models.
Both the name and version of the force field/model are relevant and should be annotated when available.

### Rules

- Include water models and other specific solvent models (e.g., `TIP3P`, `SPC/E`)
- Exclude generic terms like `force field` or `model`

### Good examples

- `CHARMM36`
- `AMBER99SB`
- `GROMOS53a6`
- `GROMOS96 43A1`
- `TIP3P`

### Bad examples

- `the force field`
- `TIP3P water` (only annotate `TIP3P`)
- `the CHARMM36 force field` (only annotate `CHARMM36`)

## Software name: SOFTNAME

### Definition

The **SOFTNAME** entity refers to the name of any software used for molecular simulation, visualization, or analysis.
It includes packages for molecular dynamics, modeling, trajectory processing, and any other computational tasks relevant to the simulation workflow.

### Rules

- Exclude generic words such as `software`, `tool`, or `program` unless they are part of the official name.
- Exclude algorithms like `SHAKE`, `RESP`...

### Good examples

- `GROMACS`
- `VMD`
- `NAMD`
- `PyMOL`
- `PLUMED`
- `COLVAR`

### Bad examples

- `Python`
- `the simulation software`
- `GROMACS software` (only annotate `GROMACS`)

## Software version: SOFTVERS

### Definition

The **SOFTVERS** entity refers to the version identifier of any software used in the simulation process.
It includes version numbers, release tags, or labels, regardless of formatting (e.g., numeric, date-based, semantic).

### Rules

- Must follow a corresponding **SOFTNAME** entity (software/tool name)
- Keep numeric and symbolic parts intact (e.g., `1.2.3-beta`)
- Include any suffixes (e.g., `2020.4`), prefixes (e.g., `v5.1.2`)

### Good examples

- `v5.0`
- `2020.4`
- `5.1.4`

### Bad examples

- `latest version`
- `software (v. 2016.4)` (only annotate `v. 2016.4`)
- `release 2023.1` (only annotate `2023.1`)

## Simulation time: STIME

### Definition

The **STIME** entity refers to the duration for which a production molecular dynamics simulation is run.

### Rules

- Exclude minimization or equilibration time
- If simulation time is presented as a range, repetition, or multiplier (e.g., `5 × 100`, `10–50`), annotate the entire expression if it refers to time
- The unit is not mandatory, but the context must unambiguously indicate that the number refers to a simulation time

### Good examples

- `5 × 200`
- `50 picoseconds`
- `100 ns`
- `4-8 μs`
- `10 to 50 ns`

### Bad examples

- `for several hours of computation`
- `10–50 replicas`

## Simulation temperature: STEMP

### Definition

The **STEMP** entity refers to the thermal conditions under which a simulation is conducted.
It includes any explicitly stated temperature values, with or without units.

### Rules

- The unit is not mandatory, but the context must unambiguously indicate that the number refers to temperature
- Exclude surrounding words like `temperature of` or `heated to`
- Include `room temperature` or `body temperature`

### Good examples

- `300K`
- `500 degrees Celsius`
- `298`
- `room temperature`

### Bad examples

- `heated up`
- `at low temperature`
