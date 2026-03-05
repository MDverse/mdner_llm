# Normalization rules

This document provides the guidelines for automatically normalizing entities retrieved from scientific texts related to molecular simulations.

See the [annotation rules](annotation_rules.md) document for the definition of entities.

## For all entities

- Normalize text to lowercase (e.g., `POPC` → `popc`).

## Molecule (MOL)

## Force field and model (FFM)

## Software name (SOFTNAME)

## Software version (SOFTVERS)

- Must contain at least one digit to be considered valid.

## Simulation time (STIME)

- Acceptable input units: `s`, `sec`, `second`, `seconds`, `ms`, `millisecond`, `microsec`, `microsecond`, `microseconds`, `ns`, `nanosecond`, `nanoseconds`, `ps`, `picosecond`, `picoseconds`.

### Examples

- `10 to 50 ns` → not `10-50 ns`

## Simulation temperature (TEMP)

- Convert all temperatures to Kelvin if standardization is required (`25 °C` → `298K`)
- Convert `room temperature` to `300 K`
