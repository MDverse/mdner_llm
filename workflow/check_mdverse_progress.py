"""CLI tool to monitor MDverse annotation progress across GLiNER and LLM engines."""

from datetime import datetime
from pathlib import Path

import click
import yaml

from mdner_llm.logger import create_logger


def compute_target_sample_count(mdverse_directory: Path) -> int:
    """Count the total number of extracted dataset JSON files to annotate.

    Returns
    -------
    int
        Total number of JSON files in the MDverse directory.
    """
    if not mdverse_directory.exists():
        return 0
    return len(list(mdverse_directory.glob("*.json")))


def count_completed_annotations(results_directory: Path) -> int:
    """Count generated prediction JSON files in the destination directory.

    Returns
    -------
    int
        Number of generated JSON prediction files.
    """
    if not results_directory.exists():
        return 0
    return len(list(results_directory.glob("*.json")))


def compute_progress_status(completed_count: int, target_count: int) -> tuple[int, str]:
    """Compute progress percentage and visual status label.

    Returns
    -------
    tuple[int, str]
        Tuple containing progress percentage and status text.
    """
    if target_count <= 0:
        return 0, "❌ No inputs"
    progress_pct = round(min((completed_count / target_count) * 100, 100))
    if progress_pct >= 100:
        status_label = "✅ Completed"
    elif progress_pct > 0:
        status_label = "⏳ In progress"
    else:
        status_label = "❌ Not started"
    return progress_pct, status_label


def format_model_label(model_identifier: str) -> str:
    """Format local path or HuggingFace repo id into a clean label.

    Returns
    -------
    str
        Formatted model label for display in progress report.
    """
    model_path = Path(model_identifier)
    # Check if checkpoint path like fold_i/best
    if model_path.name == "best" and len(model_path.parts) >= 3:
        return f"{model_path.parts[-3]}/{model_path.parts[-2]}"
    # Return original identifier without truncating slashes
    return str(model_identifier).strip()


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("workflow/configs/mdverse_annotation.yaml"),
    show_default=True,
    help="Path to the MDverse annotation YAML configuration file.",
)
def main(config_path: Path) -> None:
    """Report progress of MDverse entity annotations for GLiNER and LLM engines."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H:%M:%S")
    logger = create_logger(f"logs/check_mdverse_progress_{timestamp}.log")
    with open(config_path, encoding="utf-8") as config_stream:
        config = yaml.safe_load(config_stream) or {}

    mdverse_dir = Path(config.get("mdverse_dir", "data/mdverse"))
    target_count = compute_target_sample_count(mdverse_dir)
    configured_engines = config.get("engines", ["gliner", "llm"])
    models_config = config.get("models", {})

    logger.info("=" * 105)
    logger.info(
        f"MDVERSE ANNOTATION PROGRESS REPORT (Target: {target_count:,} samples)"
    )
    logger.info("=" * 105)
    table_header = (
        f"{'Engine':<10} | {'Model / Setup':<42} | {'Done':<17} "
        f"| {'Progress':<10} | {'Status'}"
    )
    logger.info(table_header)
    logger.info("-" * 105)

    if "gliner" in configured_engines:
        raw_gliner_path = models_config.get("gliner", {}).get(
            "model_path", "gliner-default"
        )
        display_gliner_name = format_model_label(raw_gliner_path)
        gliner_results_dir = Path("results/mdverse/gliner")
        gliner_done = count_completed_annotations(gliner_results_dir)
        pct, status = compute_progress_status(gliner_done, target_count)
        done_str = f"{gliner_done:,} / {target_count:,}"
        logger.info(
            f"{'GLiNER':<10} | "
            f"{display_gliner_name:<42} | "
            f"{done_str:>17} | "
            f"{pct:>7} %  | "
            f"{status}"
        )

    if "llm" in configured_engines:
        llm_config = models_config.get("llm", {})
        llm_model = llm_config.get("model_name", "llm-default")
        framework = llm_config.get("framework", "pydantic")
        display_llm_name = f"{llm_model} ({framework})"
        llm_results_dir = Path("results/mdverse/llm")
        llm_done = count_completed_annotations(llm_results_dir)
        pct, status = compute_progress_status(llm_done, target_count)
        done_str = f"{llm_done:,} / {target_count:,}"
        logger.info(
            f"{'LLM':<10} | "
            f"{display_llm_name:<42} | "
            f"{done_str:>17} | "
            f"{pct:>7} %  | "
            f"{status}"
        )

    logger.info("=" * 105)


if __name__ == "__main__":
    main()
