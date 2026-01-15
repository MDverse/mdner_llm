# Annotation Rules

This document outlines the guidelines for manually annotating entities in scientific texts related to molecular simulations. Each entity type is defined clearly, with instructions on how to annotate it consistently and examples for clarity.


It is important to note that we will annotate all information related to molecular simulations, regardless of whether it pertains to energy minimization, equilibration, or the production run. All of these stages are considered integral parts of the simulation process and will be treated as simulation data for the purposes of annotation.

The rules should be applied during annotation or used to validate existing annotations.

---

## 1. Molecule (MOL) 🧬

**Description:** This entity covers all types of molecular compounds, including simple molecules, ions, DNA, RNA, proteins, polymers, and complexes.

**Rules:**
- Normalize casing (e.g., `POPC` → `popc`).
- Annotate both singular and plural forms (e.g., `lipid` and `lipids`).
- Remove extra whitespace around names.
- Include chemical formulas and abbreviations.
- Amino acid sequences are acceptable. They may appear as full names, abbreviations, UniProt IDs, or other identifiers.
- Do not annotate adjectives or descriptors that modify the molecule (e.g., hydrated, charged, folded, tetrameric).

**Examples:**
- `sodium chloride` ✅
- `lipids` ✅
- `DNA` ✅
- `ethanol` ✅
- `ammonia` ✅
- `Q29537` ✅
- `Na⁺` ✅
- `hydrated sodium chloride` 🚫 → Only annotate `sodium chloride` ✅


## 2. Force field and/or model (FFM) 🛠️

**Description:** This entity refers to any force field or molecular model used to describe the interactions between particles in a simulation. This includes all classical all-atom force fields, coarse-grained models, solvent models, and water models. Both the name and version of the force field/model are considered relevant and should be annotated when available.

**Rules:**
- Normalize text to lowercase (e.g., `AMBER99SB` → `amber99sb`).
- Keep version identifiers separate if present (e.g., `GROMOS53a6` → ` GROMOS 53a6 `.
- Water models and other specific solvent models (e.g., `TIP3P`, `SPC/E`) are also considered valid FFM entities.
- Generic terms like `"force field"` or `"model"` on their own should not be annotated.
- If a molecule and a model appear together (e.g., `TIP3P water`), only annotate the model name (i.e., `TIP3P`).

**Examples:**
- `CHARMM36` ✅
- `AMBER99SB` ✅
- `GROMOS96 43A1` ✅
- `the force field` 🚫 → Too generic 
- `TIP3P water` 🚫 → Annotate `TIP3P` = FFM & `water` = MOL ✅


## 3. SOFTNAME ⚙️

**Description:** This entity refers to the name of any software used for molecular simulation, visualization, or analysis. It includes packages for molecular dynamics, modeling, trajectory processing, and other computational tasks relevant to the simulation workflow.

**Rules:**
- Avoid trailing or leading spaces.
- Annotate only the actual name of the software, excluding surrounding generic words such as software, tool, or program unless they are part of the official name.
- Only allow software names from the **controlled list** defined in [`docs/md_software.md`](md_software.md).

**Examples:**
- `GROMACS` ✅
- `VMD` ✅
- `NAMD` ✅
- `PyMOL` ✅
- `Python` ✅
- `the simulation software` 🚫 → No specific name
- `GROMACS software` 🚫 → Annotate only `GROMACS` ✅


## 4. SOFTVERS 🔢

**Description:** This entity refers to the version identifier of any software used in the simulation process. It includes version numbers, release tags, or labels, regardless of formatting (e.g., numeric, date-based, semantic).

**Rules:**
- Must follow a corresponding **SOFTNAME** (software/tool name).
- Keep numeric and symbolic parts intact (e.g., `1.2.3-beta`).
- Remove leading/trailing spaces.
- Must contain at least **one digit** to be considered valid.
- This may include numeric versions (e.g., `2020.4`), prefixed versions (e.g., `v5.1.2`), or labeled releases (e.g., `release 2023.1`).

**Examples:**
- `v5.0` ✅
- `2020.4` ✅
- `5.1.4` ✅
- `latest version` 🚫 → No specific version provided
- `software (v. 2016.4)` 🚫 → Annotate only `v. 2016.4` ✅
- `release 2023.1` 🚫 → Annotate only `2023.1` ✅


## 5. STIME ⏱️

**Description:** This entity refers to the duration for which a molecular simulation is run. It includes any explicit mention of time related to the minimization, equilibration, or production stages of the simulation process.

**Rules:**
- If simulation time is presented as a range, repetition, or multiplier (e.g., `5 × 100`, `10–50`), annotate the entire expression if it refers to time.
- Acceptable input units: `s`, `sec`, `second`, `seconds`, `ms`, `millisecond`, `microsec`, `microsecond`, `microseconds`, `ns`, `nanosecond`, `nanoseconds`, `ps`, `picosecond`, `picoseconds`.
- The unit is not mandatory, but the context must unambiguously indicate that the number refers to a simulation time.
- Remove unnecessary spaces between number and unit (e.g., `5000 ps` → `5000ps`).
- Handle ranges consistently: when a duration is expressed as a range, annotate it using the `start–end` format.

**Examples:**
- `5 × 200` ✅
- `50 picoseconds / 100 ns` ✅
- `three runs of 500 each` ✅
- `4-8 μs` ✅
- `10 to 50 ns` 🚫 → not in the right format (`10-50 ns`)
- `for several hours of computation` 🚫 → Computation time, not simulation time
- `10–50 replicas` 🚫 → Number of replicas, not a time duration


## 6. TEMP 🌡️

**Description:** This entity refers to the thermal conditions under which a simulation is conducted. It includes any explicitly stated temperature values, with or without units.

**Rules:**
- Always specify unit (K, °C, °F) immediately after number.
- No space between value and unit: `300 K` → `300K`.
- No point after the unit: `300 K` → `300K`.
- The unit is not mandatory, but the context must unambiguously indicate that the number refers to temperature.
- Convert all temperatures to Kelvin if standardization is required (`25 °C` → `298K`).
- Do not include surrounding words like `“temperature of”` or `“heated to”`.

**Examples:**
- `300K` ✅
- `500 degrees Celsius` ✅
- `298` ✅ (if clearly referring to temperature)
- `340 k.` 🚫 → `340 k` ✅
- `heated up` 🚫 → Vague, no value
- `room temperature` 🚫 →  Not a numerical value

