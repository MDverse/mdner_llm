"""Select informative annotation JSON files and export their paths.

This script selects a subset of annotation files based on entity statistics
and file recency, then writes the selected file paths to a text file
(one path per line).

Selection priorities:
1. Files containing all entity types at least once.
2. Files with a moderate number of molecules (2 to 5).
3. Most recent files.
"""

from datetime import UTC, datetime
from pathlib import Path

import click
import loguru

from mdner_llm.core.logger import create_logger
from mdner_llm.utils.count_entities import CLASSES, compute_entity_counts_df


def select_annotation_files(
    annotations_dir: Path,
    nb_files: int,
    logger: "loguru.Logger" = loguru.logger,
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

    Returns
    -------
    list[Path]
        Selected annotation file paths.

    Raises
    ------
    ValueError
        If no JSON files are found or the TSV file is invalid.
    """
    logger.info(f"Selecting text to annotate from {annotations_dir}.")
    # Load all annotation files in the directory
    df = compute_entity_counts_df(annotations_dir, CLASSES)
    # Ensure the DataFrame can be matched to JSON filenames
    if "filename" not in df.columns:
        msg = "DataFrame must contain a 'filename' column to match JSON files."
        raise ValueError(msg)
    # List all available annotation JSON files, sorted by recency
    json_files = sorted(
        annotations_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Map filenames to paths for fast lookup
    file_map = {p.name: p for p in json_files}
    # Accumulator for selected filenames (keeps insertion order)
    selected: list[str] = []
    # Identify entity-count columns (excluding SOFTVERS)
    entity_cols = [
        col for col in df.columns if col.endswith("_nb") and col != "SOFTVERS_nb"
    ]

    # Priority 1: files containing at least one instance of each entity type
    if entity_cols:
        df_all = df[(df[entity_cols] > 0).all(axis=1)]
        selected.extend(fname for fname in df_all["filename"] if fname in file_map)

    # Priority 2: files with a moderate number of molecules (2-10)
    # Applied only if the selection is still incomplete
    if "MOLECULE_nb" in df.columns and len(selected) < nb_files:
        df_mol = df[(df["MOLECULE_nb"] >= 2) & (df["MOLECULE_nb"] <= 10)]
        selected.extend(
            fname
            for fname in df_mol["filename"]
            if fname in file_map and fname not in selected
        )

    # Priority 3: fill remaining slots with the most recent files
    if len(selected) < nb_files:
        selected.extend(fname for fname in file_map if fname not in selected)

    selected_files = [file_map[name] for name in selected[:nb_files]]
    logger.info(f"First annotation file path: {selected_files[0]!s}")
    logger.success(
        f"Selected {len(selected_files)} interesting annotations successfully!"
    )
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
    default=Path("annotations/v3"),
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
        "Example: results/50_selected_files_20260102.txt"
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
    logger = create_logger()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    if res_path is None:
        res_path = Path(
            f"results/{nb_files}_selected_files_{timestamp}.txt",
        )
    res_path.parent.mkdir(parents=True, exist_ok=True)

    selected_files = select_annotation_files(
        annotations_dir=annotations_dir,
        nb_files=nb_files,
        logger=logger,
    )

    with res_path.open("w", encoding="utf-8") as handle:
        for path in selected_files:
            handle.write(f"{path}\n")

    logger.success(f"Wrote selected file paths to {res_path} successfully!")


if __name__ == "__main__":
    main()
