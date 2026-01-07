# Annotation Rules

## Introduction

This document defines the **annotation rules and business standards** for
the entities extracted from QC annotation files. The goal is to ensure
consistency, accuracy, and clarity across all annotated data.

These rules cover:
- Formatting of entity text
- Standardization of units and numeric values
- Consistent naming and casing
- Handling of rare or ambiguous entities

The rules should be applied during annotation or used to validate
existing annotations.

---

## Classes and Rules

### 1. MOL 🧬
**Description:** Chemical molecules, compounds, ions...etc.

**Rules:**
- Normalize casing (e.g., `POPC` → `popc`).
- Remove extra whitespace around names.

To be continued...

---

### 2. FFM 🛠️
**Description:** Force Field Model.

**Rules:**
- Normalize text to lowercase (e.g., `AMBER99SB` → `amber99sb`)
- Keep version identifiers separate if present (e.g., ` GROMOS 53a6 ` → `gromos53a6`)
  
To be continued...

---

### 3. SOFTNAME ⚙️
**Description:** Software or tool names.

**Rules:**
- Avoid trailing or leading spaces.
- Only allow software names from the **controlled list** defined in [`docs/md_software.md`](md_software.md).

---

### 4. SOFTVERS 🔢
**Description:** Software version identifiers.

**Rules:**
- Must follow a corresponding **SOFTNAME** (software/tool).
- Keep numeric and symbolic parts intact (e.g., `1.2.3-beta`).
- Normalize to lowercase.
- Remove leading/trailing spaces.
- Must contain at least **one digit** to be considered valid.
- Example: `AMBER99SB-ILDN` → `amber99sb-ildn`

To be continued...


### 5. STIME ⏱️
**Description:** The accumulated time simulation.

**Rules:**
- Normalize to a **consistent unit**. Recommended standard: **picoseconds (`ps`)**.
- Acceptable input units: `s`, `sec`, `second`, `seconds`, `ms`, `millisecond`, `microsec`, `microsecond`, `microseconds`, `ns`, `nanosecond`, `nanoseconds`, `ps`, `picosecond`, `picoseconds`.
- Convert all values to **picoseconds** for consistency.
- Remove unnecessary spaces between number and unit (`<value><unit>` (e.g., `5000ps`)).
- Normalize units to lowercase.
- Handle ranges consistently: `start-end` should be represented in the same unit.
  - Example: `1-5 ns` → `1000-5000 ps`


To be continued...

---

### 6. TEMP 🌡️
**Description:** Temperature values.

**Rules:**
- Always specify unit (K, °C, °F) immediately after number.
- No space between value and unit: `300 K` → `300K`
- Convert all temperatures to Kelvin if standardization is required.
- Example: `25 °C` → `298K`

To be continued...

---
