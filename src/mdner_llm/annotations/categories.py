"""Module for defining entity categories."""

CATEGORIES = {
    "MOL": (
        "Molecular compounds, including simple molecules, ions, nucleic acids,"
        " proteins, lipids, sugars, polymers, and complexes. MOL entities must"
        " be uniquely identifiable."
    ),
    "FFM": (
        "Any force field or molecular model used to describe interatomic or"
        " intermolecular interactions, including all-atom force fields,"
        " coarse-grained models, and solvent/water models. Both the name and,"
        " when available, the version are annotated."
    ),
    "SOFTNAME": (
        "Name of any software used for simulation, visualization, or analysis,"
        " including packages for MD, modeling, trajectory processing, and"
        " other computational tasks in the simulation workflow."
    ),
    "SOFTVERS": (
        "Version identifier of any software used in the simulation process,"
        " regardless of formatting (numeric, date-based, or semantic"
        " versioning)."
    ),
    "STEMP": (
        "Thermal conditions under which a simulation is conducted, including"
        " any explicitly stated temperature value, with or without units."
    ),
    "STIME": "Duration for which a production MD simulation is run.",
}

BLACKLIST = {
    "MOL": {
        "dna",
        "ion",
        "ions",
        "ligand",
        "ligands",
        "lipid",
        "lipids",
        "membrane",
        "protein",
        "proteins",
        "rna",
        "salt",
        "water",
        "waters",
    },
    "SOFTNAME": {"software", "tool", "unknown"},
    "SOFTVERS": {"version"},
    "FFM": {"ffm", "forcefield"},
    "STIME": {"duration", "time"},
    "STEMP": {"temp", "temperature"},
}
