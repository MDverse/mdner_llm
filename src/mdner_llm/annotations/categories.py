"""Module for defining entity categories."""

CATEGORIES = {
    "MOL": "Molecule or chemical compound involved in the simulation.",
    "FFM": "Force field or model used to describe interatomic interactions.",
    "SOFTNAME": "Names of software packages used in molecular dynamics simulations.",
    "SOFTVERS": "Version numbers of software packages.",
    "TEMP": "Simulation temperature, typically expressed in Kelvin or Celsius",
    "STIME": "Total simulation time or duration.",
}


BLACKLIST = {
    "MOL": {
        "water",
        "waters",
        "lipid",
        "lipids",
        "protein",
        "proteins",
        "salt",
        "membrane",
        "dna",
        "rna",
        "ion",
        "ions",
        "ligand",
        "ligands",
    },
    "SOFTNAME": {"unknown", "software", "tool"},
    "SOFTVERS": {"version"},
    "FFM": {"ffm", "forcefield"},
    "STIME": {"time", "duration"},
    "TEMP": {"temperature", "temp"},
}
