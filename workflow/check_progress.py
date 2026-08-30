"""CLI tool to report benchmark progress across LLM inferences benchmark."""

from datetime import datetime
from pathlib import Path
from typing import Any

import click
import yaml

from mdner_llm.common import sanitize_filename
from mdner_llm.logger import create_logger


def compute_target_sample_count(config: dict[str, Any]) -> int:
    """Compute the expected number of text samples to process.

    Returns
    -------
    int
        Target count of ground truth text files.
    """
    max_samples = config.get("max_samples")
    if max_samples:
        return int(max_samples)
    texts_dir = Path(config.get("texts_path", "data/groundtruth"))
    if not texts_dir.exists():
        return 0
    return len(list(texts_dir.glob("*.json")))


def count_completed_inferences(inference_dir: Path) -> int:
    """Count generated prediction files in a given directory.

    Returns
    -------
    int
        Number of generated JSON or TXT prediction files.
    """
    if not inference_dir.exists():
        return 0
    # Compte les JSON de résultats
    json_count = len(list(inference_dir.glob("*.json")))
    if json_count > 0:
        return json_count
    return len(list(inference_dir.glob("*.txt")))


def compute_completion_status(completed: int, target: int) -> tuple[float, str]:
    """Compute progress percentage and status label.

    Returns
    -------
    tuple[float, str]
        Tuple containing completion percentage and status text.
    """
    if target <= 0:
        return 0.0, "❌ Empty"
    percentage = min((completed / target) * 100.0, 100.0)
    if percentage >= 100.0:
        status_label = "✅ Completed"
    elif percentage > 0.0:
        status_label = "⏳ In progress"
    else:
        status_label = "❌ Not started"
    return percentage, status_label


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the LLM YAML config file.",
)
def main(config_path: Path) -> None:
    """Display progress report for LLM benchmark strategies and consensus."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H:%M:%S")
    logger = create_logger(f"logs/check_progress_{timestamp}.log")

    # Load configuration settings.
    with open(config_path, encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle) or {}

    # Resolve base paths and parameters.
    base_out = Path(config.get("output_dir_base", "results/llm"))
    target_count = compute_target_sample_count(config)
    strategies = config.get("benchmark_strategies", {})
    strat_models = config.get("benchmark_models", [])
    full_eval_models = config.get("full_eval_models", [])
    consensus_models = config.get("consensus_models", [])
    consensus_temps = [str(t) for t in config.get("consensus_temperatures", [1.0])]

    # Print dashboard header banner.
    logger.info("=" * 105)
    logger.info(f"BENCHMARK PROGRESS REPORT (Target: {target_count} samples)")
    logger.info("=" * 105)
    table_header = f"{'Scenario':<33} | {'Model / Setup':<28} | {'Done':<9} | "
    table_header += f"{'Progress':<10} | {'Status'}"
    logger.info(table_header)
    logger.info("-" * 105)

    # 1. Scenario 1: Benchmark prompting strategies.
    for strat_name in strategies:
        for model_name in strat_models:
            safe_model = sanitize_filename(model_name)
            target_dir = base_out / "inferences" / "raw" / strat_name / safe_model
            done_count = count_completed_inferences(target_dir)
            pct, status = compute_completion_status(done_count, target_count)
            short_model = model_name.split("/")[-1]
            logger.info(
                f"{strat_name:<33} | "
                f"{short_model:<28} | "
                f"{done_count:>4}/{target_count:<4} | "
                f"{pct:>6.1f} %  | "
                f"{status}"
            )

    logger.info("-" * 105)

    # 2. Scenario 2: Full evaluation models (with instructor and guidelines).
    scenario2_combo = "with_instructor_with_guidelines"
    for model_name in full_eval_models:
        safe_model = sanitize_filename(model_name)
        target_dir = base_out / "inferences" / "raw" / scenario2_combo / safe_model
        done_count = count_completed_inferences(target_dir)
        pct, status = compute_completion_status(done_count, target_count)
        short_model = model_name.split("/")[-1]
        logger.info(
            f"{'benchmark_models':<33} | "
            f"{short_model:<28} | "
            f"{done_count:>4}/{target_count:<4} | "
            f"{pct:>6.1f} %  | "
            f"{status}"
        )

    logger.info("-" * 105)

    # 3. Scenario 3: Consensus raw inferences.
    for temp in consensus_temps:
        setup_label = f"consensus_raw/temp_{temp}"
        for model_name in consensus_models:
            safe_model = sanitize_filename(model_name)
            if temp in ["1", "1.0"]:
                target_dir = (
                    base_out
                    / "inferences"
                    / "raw"
                    / "with_instructor_with_guidelines"
                    / safe_model
                )
            else:
                target_dir = (
                    base_out
                    / "inferences"
                    / "consensus_raw"
                    / f"temp_{temp}"
                    / safe_model
                )
            done_count = count_completed_inferences(target_dir)
            pct, status = compute_completion_status(done_count, target_count)
            short_model = model_name.split("/")[-1]
            logger.info(
                f"{setup_label:<33} | "
                f"{short_model:<28} | "
                f"{done_count:>4}/{target_count:<4} | "
                f"{pct:>6.1f} %  | "
                f"{status}"
            )

    logger.info(f"{'=' * 105}\n")


if __name__ == "__main__":
    main()
