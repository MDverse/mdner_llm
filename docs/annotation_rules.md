# Annotation guidelines

This document outlines the guidelines for manually annotating entities in scientific texts related to molecular simulations.

## Molecule: MOL

### Definition

The **MOL** entity covers molecular compounds, including simple molecules, ions, nucleic acids, proteins, lipids, sugars, polymers, and complexes.
MOL entities should be uniquely identifiable.

### Rules

1. Exclude whitespaces and punctuation marks around entities
2. It should be at least 2 characters long
3. Annotate always both singular and plural forms (e.g., `alkane` and `alkanes`)
4. Annotate chemical formulas and abbreviations separately from full molecule names.
5. Include amino acid and nucleic acid sequences
6. Include any identifiers (PDB ID, UniProt ID...)
7. Exclude adjectives or descriptors that modify the molecule (e.g., `hydrated`, `charged`, `folded`, `tetrameric`)
8. Exclude generic terms like `protein`, `lipid`, `phospholipid`, `DNA`, `sugar`, `water`, `ions`, `compounds`...
9. Exclude specific résidues or domains like `Lys-353`...

### Good examples

- `sodium chloride` 
- `ethanol`
- `ammonia`
- `Q29537`
- `2RH1`
- `Na⁺`

### Bad examples

- `H` (Rule 2)
- `coronavirus 2019-nCoV protease Mpro` (annotate separately `coronavirus 2019-nCoV protease` and `Mpro`, Rule 4)
- `hydrated sodium chloride` (only annotate `sodium chloride`, Rule 7)
- `lipids` (Rule 8)
- `DNA` (Rule 8)
- `binding protein` (Rule 8)
- `ions in solvant` (Rule 8)

## Force field and model: FFM

### Definition

The **FFM** entity refers to any force field or molecular model used to describe the interactions between particles in a simulation.
This includes all classical all-atom force fields, coarse-grained models, solvent models, and water models.
Both the name and version of the force field/model are relevant and should be annotated when available.

### Rules

1. Include water models and other specific solvent models (e.g., `TIP3P`, `SPC/E`)
2. Exclude generic terms like `force field` or `model`

### Good examples

- `CHARMM36`
- `CHARMM C36m`
- `AMBER99SB`
- `GROMOS53a6`
- `GROMOS96 43A1`
- `Poger GROMOS 53A6_L`
- `TIP3P`
- `charmm36 (v. june 2015)`

### Bad examples

- `TIP3P water` (only annotate `TIP3P`) (Rule 1 and Rule 2)
- `the force field` (Rule 2)
- `the CHARMM36 force field` (only annotate `CHARMM36`) (Rule 2)

## Software name: SOFTNAME

### Definition

The **SOFTNAME** entity refers to the name of any software used for molecular simulation, visualization, or analysis.
It includes packages for molecular dynamics, modeling, trajectory processing, and any other computational tasks relevant to the simulation workflow.

### Rules

1. Exclude generic words such as `software`, `tool`, or `program` unless they are part of the official name.
2. Exclude algorithms like `SHAKE`, `RESP`...
3. Exclude languages like `Python`, `C++`, `Fortran`...

### Good examples

- `GROMACS`
- `VMD`
- `NAMD`
- `PyMOL`
- `PLUMED`
- `COLVAR`
- `CHARMM-Gui`
- `Jupyter`

### Bad examples

- `the simulation software` (Rule 1)
- `GROMACS software` (only annotate `GROMACS`, Rule 1)
- `Python` (Rule 3)

## Software version: SOFTVERS

### Definition

The **SOFTVERS** entity refers to the version identifier of any software used in the simulation process.
It includes version numbers, release tags, or labels, regardless of formatting (e.g., numeric, date-based, semantic).

### Rules

1. Must follow a corresponding **SOFTNAME** entity (software/tool name)
2. Keep numeric and symbolic parts intact (e.g., `1.2.3-beta`)
3. Include any suffixes (e.g., `2020.4`), prefixes (e.g., `v5.1.2`)
4. Exclude surrounding words like `version`, `release`, `software`...

### Good examples

- `v5.0`
- `2020.4`
- `5.1.4`

### Bad examples

- `charmm36 (v. june 2015)` (it is not a software version, Rule 1)
- `software (v. 2016.4)` (only annotate `v. 2016.4`) (Rule 3 and Rule 4)
- `release 2023.1` (only annotate `2023.1`) (Rule 4)
- `latest version` (too generic, Rule 4)

## Simulation time: STIME

### Definition

The **STIME** entity refers to the duration for which a production molecular dynamics simulation is run.

### Rules

1. Exclude minimization or equilibration time
2. If simulation time is presented as a range, repetition, or multiplier (e.g., `5 × 100`, `10–50`), annotate the entire expression if it refers to time
3. The unit is not mandatory, but the context must unambiguously indicate that the number refers to a simulation time

### Good examples

- `5 × 200`
- `50 picoseconds`
- `100 ns`
- `4-8 μs`
- `10 to 50 ns`

### Bad examples

- `10–50 replicas` (not a simulation time, Rule 2)
- `for several hours of computation` (too vague, Rule 3)

## Simulation temperature: STEMP

### Definition

The **STEMP** entity refers to the thermal conditions under which a simulation is conducted.
It includes any explicitly stated temperature values, with or without units.

### Rules

1. The unit is not mandatory, but the context must unambiguously indicate that the number refers to temperature
2. Exclude surrounding words like `temperature of` or `heated to`
3. Include `room temperature` or `body temperature`

### Good examples

- `300K`
- `500 degrees Celsius`
- `298`
- `room temperature`

### Bad examples

- "measured as a function of temperature" `293.15–313.15 K` (not a simulation temperature, Rule 1)
- `heated up`   (too generic, Rule 2)
- `at low temperature` (Rule 2)
  
