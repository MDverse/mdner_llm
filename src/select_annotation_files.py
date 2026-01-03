"""Select informative annotation JSON files and export their paths.

This script selects a subset of annotation files based on entity statistics
and file recency, then writes the selected file paths to a text file
(one path per line).

Selection priorities:
1. Files containing all entity types at least once.
2. Files with a moderate number of molecules (2 to 5).
3. Most recent files.


Usage:
======
uv run src/select_annotation_files.py --annotations-dir PATH --nb-files INT
                             [--res-path PATH]


Arguments:
==========
--annotations-dir: Path
    Directory containing annotation JSON files.
    Default: "annotations/v2"

--nb-files: int
    Maximum number of annotation files to select.
    Default: 50

--res-path: Path (Optional)
    Output directory where the result text file will be written.
    If not provided, a timestamped directory is created under "results/".
    Default:
        results/{nb_files}_selected_files_YYYYMMDD_HHMMSS


Example:
========
uv run src/select_annotation_files.py \
        --annotations-dir annotations/v2 \
        --nb-files 50 \
        --res-path results/50_selected_files_20260102

This command selects up to 50 annotation JSON files from `annotations/v2`
according to entity coverage and recency, and writes their paths to:
`results/50_selected_files_20260102.txt`
"""

# METADATAS
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "AGPL-3.0 license"
__date__ = "2025"
__version__ = "1.0.0"


# LIBRARY IMPORTS
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from loguru import logger


# FUNCTIONS
def select_annotation_files(
    annotations_dir: Path,
    nb_files: int,
    tsv_path: Path = Path("results/all_annotations_entities_count.tsv"),
) -> list[Path]:
    """
    Select informative annotation JSON files from a directory.

    Priority:
    1. Files with all entity types present.
    2. Files with 2-5 molecules.
    3. Most recent files.

    Parameters
    ----------
    annotations_dir : Path
        Directory containing annotation JSON files.
    nb_files : int
        Maximum number of files to select.
    tsv_path : Path
        TSV file containing entity counts.

    Returns
    -------
    list[Path]
        Selected annotation file paths.

    Raises
    ------
    ValueError
        If no JSON files are found or the TSV file is invalid.
    """
    logger.info(f"Selecting text to annotate from {annotations_dir}...")
    # Load entity count table (one row per annotation file)
    df = pd.read_csv(tsv_path, sep="\t")

    # Ensure the TSV can be matched to JSON filenames
    if "filename" not in df.columns:
        msg = "TSV file must contain a 'filename' column"
        raise ValueError(msg)

    # List all available annotation JSON files, sorted by recency
    json_files = sorted(
        annotations_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not json_files:
        msg = f"No JSON files found in {annotations_dir}"
        raise ValueError(msg)

    # Map filenames to paths for fast lookup
    file_map = {p.name: p for p in json_files}

    # Accumulator for selected filenames (keeps insertion order)
    selected: list[str] = []

    # Identify entity-count columns (excluding SOFTVERS)
    entity_cols = [
        col for col in df.columns
        if col.endswith("_nb") and col != "SOFTVERS_nb"
    ]

    # Priority 1: files containing at least one instance of each entity type
    if entity_cols:
        df_all = df[(df[entity_cols] > 0).all(axis=1)]
        selected.extend(
            fname for fname in df_all["filename"]
            if fname in file_map
        )

    # Priority 2: files with a moderate number of molecules (2-5)
    # Applied only if the selection is still incomplete
    if "MOLECULE_nb" in df.columns and len(selected) < nb_files:
        df_mol = df[
            (df["MOLECULE_nb"] >= 2) & (df["MOLECULE_nb"] <= 5)
        ]
        selected.extend(
            fname for fname in df_mol["filename"]
            if fname in file_map and fname not in selected
        )

    # Priority 3: fill remaining slots with the most recent files
    if len(selected) < nb_files:
        selected.extend(
            fname for fname in file_map
            if fname not in selected
        )

    selected_files = [file_map[name] for name in selected[:nb_files]]
    logger.success(f"Selected {len(selected_files)} "
                   "interesting annotations successfully!")
    # Return paths, truncated to the requested number of files
    return selected_files


@click.command()
@click.option(
    "--annotations-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
    default=Path("annotations/v2"),
    show_default=True,
    help="Directory containing annotation JSON files.",
)
@click.option(
    "--nb-files",
    type=int,
    default=50,
    show_default=True,
    help="Number of annotation files to select.",
)
@click.option(
    "--res-path",
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    default=None,
    help=(
        "Output text file path. "
        "If not provided, a timestamped file is created under 'results/'."
    ),
)
def main(
    annotations_dir: Path,
    nb_files: int,
    res_path: Path | None,
) -> None:
    """
    Run the annotation file selection and export the results.

    The output is a text file containing one annotation file path per line.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if res_path is None:
        res_path = Path(
            f"results/{nb_files}_selected_files_{timestamp}.txt",
        )
    res_path.parent.mkdir(parents=True, exist_ok=True)

    available_files = list(annotations_dir.glob("*.json"))
    total_files = len(available_files)

    if nb_files > total_files:
        logger.warning(
            f"Requested {nb_files} files, but only {total_files} annotation files "
            f"are available in {annotations_dir}."
        )

    tsv_path = Path("results/all_annotations_entities_count.tsv")
    selected_files = select_annotation_files(
        annotations_dir=annotations_dir,
        nb_files=nb_files,
        tsv_path=tsv_path,
    )

    output_file = res_path / "selected_annotation_files.txt"

    with res_path.open("w", encoding="utf-8") as handle:
        for path in selected_files:
            handle.write(f"{path}\n")

    logger.success(
        f"Wrote selected file paths to {output_file} successfully!"
    )


# MAIN PROGRAM
if __name__ == "__main__":
    main()
