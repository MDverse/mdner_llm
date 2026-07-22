"""Script to get regex normalisation results."""

import re
from pathlib import Path

import click
import pandas as pd
from loguru import logger

PATTERN = re.compile(
    r"([0-9]+)(\.?[0-9]+)? *(ps|ns|μs|ms|s)",
    re.IGNORECASE,
)

UNITS_TO_NS = {
    "ps": 1e-3,
    "ns": 1,
    "μs": 1e3,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
}


def get_stime_entities(entities_file: Path) -> pd.DataFrame:
    """Load entities from TSV and filter for STIME category.

    Parameters
    ----------
        entities_file (Path): Path to the TSV file containing entities.

    Returns
    -------
        pd.DataFrame: DataFrame containing only STIME entities.
    """
    entities = pd.read_csv(entities_file, sep="\t")
    stime_entities = entities[entities["category"] == "STIME"].copy()
    return stime_entities


def norm_stime_regex(stime_entity: str) -> dict:
    """Normalize a single simulation time entity using regex pattern matching.

    Parameters
    ----------
        stime_entity (str): The raw simulation time string to normalize.

    Returns
    -------
        dict: A dictionary with keys 'STIME', 'regex_value', and 'regex_unit'.
    """
    reg_value, reg_unit = None, None
    match = PATTERN.search(str(stime_entity))
    if match:
        reg_value = float(match.group(1) + (match.group(2) or ""))
        reg_unit = match.group(3).strip().lower()
        if reg_unit not in UNITS_TO_NS:
            reg_value, reg_unit = None, None
    else:
        logger.warning(f"STIME: No regex match found for '{stime_entity}'.")
    return {
        "STIME": str(stime_entity),
        "regex_value": reg_value if reg_value is not None else "None",
        "regex_unit": reg_unit if reg_unit is not None else "None",
    }


def normalize_stim_regex(stime_entities: pd.DataFrame) -> list[dict]:
    """Normalize simulation time entities using regex pattern matching.

    Parameters
    ----------
        stime_entities (pd.DataFrame): DataFrame containing STIME entities.

    Returns
    -------
        list[dict]: List of dicts with keys STIME, regex_value, and regex_unit.
    """
    results = []
    for _, row in stime_entities.iterrows():
        reg_value, reg_unit = None, None
        match = PATTERN.search(str(row["entity"]))
        if match:
            reg_value = float(match.group(1) + (match.group(2) or ""))
            reg_unit = match.group(3).strip().lower()
            if reg_unit not in UNITS_TO_NS:
                reg_value, reg_unit = None, None
        results.append(
            {
                "STIME": str(row["entity"]),
                "regex_value": reg_value if reg_value is not None else "None",
                "regex_unit": reg_unit if reg_unit is not None else "None",
            }
        )
    return results


def save_results_to_tsv(results: list, output_file: Path):
    """Save the LLM normalization results to a TSV file.

    Parameters
    ----------
        results: A list of dictionaries containing the normalization results.
        output_file: Path to the output TSV file.
    """
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(by="STIME")
    results_df_sorted.to_csv(output_file, sep="\t", index=False)
    print(f"\nFile TSV saved : {output_file} ({len(results_df_sorted)} lines)")


@click.command()
@click.option(
    "--entities-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/entities.tsv"),
    help="Path to the TSV file containing the entities.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("results/norm_simu_times/normalized_stime_results.tsv"),
    help="Path to the output TSV file for normalized results.",
)
def main(entities_file: Path, output_file: Path):
    """Run regex-based normalization of simulation time entities and save results."""
    stime_entities = get_stime_entities(entities_file)
    results = normalize_stim_regex(stime_entities)
    save_results_to_tsv(results, output_file)


if __name__ == "__main__":
    main()
