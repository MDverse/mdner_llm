"""Script to get llm normalisation results."""

import json
from pathlib import Path

import click
import pandas as pd
from mdverse_entity_norm.scripts.evaluate_llm_models import (
    normalize_simulation_time,
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


def normalize_row(raw_time, model_name: str, prompt_path: Path) -> str | None:
    """Normalize a single raw simulation time string using the LLM.

    Parameters
    ----------
        raw_time: The raw simulation time string to normalize.
        model_name: The name of the LLM model to use for normalization.
        prompt_path: Path to the prompt file for the LLM.

    Returns
    -------
        str | None: The normalized simulation time in nanoseconds,
        or None if normalization fails.
    """
    result = normalize_simulation_time(
        raw_simulation_time=str(raw_time),
        model_name=model_name,
        prompt_file_path=prompt_path,
    )
    if result and result[0]:
        return result[0]


def normalize_dataframe_times(
    stime_entities: pd.DataFrame,
    prompt_path: Path,
    model_name: str,
    column: str = "entity",
) -> pd.DataFrame:
    """Apply LLM normalization to the specified column of the DataFrame.

    Returns
    -------
        pd.DataFrame: The input DataFrame with an additional column 'normalized_time'
        containing the normalized values.
    """
    stime_entities["normalized_time"] = stime_entities[column].apply(
        lambda x: normalize_row(
            raw_time=x, model_name=model_name, prompt_path=prompt_path
        )
    )
    return stime_entities


def get_llm_normalization_results(stime: pd.DataFrame) -> list[dict]:
    """Extract LLM normalization results from the DataFrame.

    Parameters
    ----------
        stime: DataFrame containing the original entities and their normalized times.

    Returns
    -------
        list[dict]: A list of dictionaries with keys 'STIME', 'LLM_value',
        and 'LLM_unit'.
    """
    results = []
    for _, row in stime.iterrows():
        llm_outputs = []
        outputs = json.loads(row["normalized_time"])["output"]
        for output in outputs:
            if output.get("value") is not None and output.get("unit") is not None:
                unit = output["unit"].strip().lower()
                if unit in UNITS_TO_NS:
                    llm_outputs.append((float(output["value"]), unit))

        if not llm_outputs:
            llm_outputs = [(None, None)]

        for value, unit in llm_outputs:
            results.append(
                {
                    "STIME": str(row["entity"]),
                    "LLM_value": value if value is not None else "None",
                    "LLM_unit": unit if unit is not None else "None",
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
    "--entities-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the TSV file containing the entities.",
)
@click.option(
    "--normalization-results-path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the output TSV file for normalized results.",
)
@click.option(
    "--prompt-path",
    type=click.Path(file_okay=True, path_type=Path),
    help="Path to the llm prompt file",
)
@click.option(
    "--model-name",
    type=str,
    help="Name of the LLM model to use for normalization.",
)
def main(
    entities_path: Path,
    normalization_results_path: Path,
    prompt_path: Path,
    model_name: str,
):
    """Execute the normalization process."""
    stime_entities = get_stime_entities(entities_path)
    stime_entities = normalize_dataframe_times(stime_entities, prompt_path, model_name)
    results = get_llm_normalization_results(stime_entities)
    save_results_to_tsv(results, normalization_results_path)


if __name__ == "__main__":
    main()
