"""Report inference progress for LLM, GLiNER, and MDverse workflows."""

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import click
import yaml
from loguru import logger

from mdner_llm.common import sanitize_filename
from mdner_llm.logger import create_logger

# CONSTANTS FOR TABLE FORMATTING
ENGINE_WIDTH = 6
SCENARIO_WIDTH = 32
MODEL_WIDTH = 35
DONE_WIDTH = 15
PROGRESS_WIDTH = 8
STATUS_WIDTH = 16
SEPARATOR = " | "
TABLE_WIDTH = (
    ENGINE_WIDTH
    + SCENARIO_WIDTH
    + MODEL_WIDTH
    + DONE_WIDTH
    + PROGRESS_WIDTH
    + STATUS_WIDTH
    + len(SEPARATOR) * 5
)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML file and validate its top-level mapping.

    Returns
    -------
    dict[str, Any]
        Loaded configuration.
    """
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError) as error:
        logger.error(f"Cannot load {config_path}: {error}")
        return {}
    if not isinstance(config, dict):
        logger.error(f"Expected a YAML mapping in {config_path}.")
        return {}
    return config


def count_json_files(directory: Path) -> int:
    """Count JSON files directly inside a directory.

    Returns
    -------
    int
        Number of JSON files.
    """
    return sum(1 for _ in directory.glob("*.json")) if directory.is_dir() else 0


def get_target_count(config: Mapping[str, Any]) -> int:
    """Determine the expected sample count from a workflow configuration.

    Returns
    -------
    int
        Expected sample count.
    """
    texts_path = config.get("texts_path")
    available = count_json_files(Path(str(texts_path))) if texts_path else 0
    max_samples = config.get("max_samples")
    if max_samples is None or int(max_samples) <= 0:
        return available
    limit = int(max_samples)
    return min(limit, available) if available else limit


def compute_progress(completed: int, target: int) -> tuple[int, str]:
    """Compute completion percentage and status.

    Returns
    -------
    tuple[int, str]
        Completion percentage and status label.
    """
    if target <= 0:
        return 0, "❌ Not started"
    percentage = round(min(completed / target * 100, 100))
    if completed >= target:
        status = "✅ Completed"
    elif completed:
        status = "⏳ In progress"
    else:
        status = "❌ Not started"
    return percentage, status


def get_consensus_models(config: Mapping[str, Any]) -> list[str]:
    """Return unique consensus models in configuration order.

    Returns
    -------
    list[str]
        Ordered model identifiers.
    """
    groups = config.get("consensus_groups", {})
    if groups:
        return list(
            dict.fromkeys(model for models in groups.values() for model in models)
        )
    return list(config.get("consensus_models", []))


def log_table_header(log: Any) -> None:
    """Log the shared progress table header."""
    log.info(
        f"{'Engine':<{ENGINE_WIDTH}}{SEPARATOR}"
        f"{'Scenario':<{SCENARIO_WIDTH}}{SEPARATOR}"
        f"{'Model':<{MODEL_WIDTH}}{SEPARATOR}"
        f"{'Done':>{DONE_WIDTH}}{SEPARATOR}"
        f"{'Progress':>{PROGRESS_WIDTH}}{SEPARATOR}"
        f"{'Status':<{STATUS_WIDTH}}"
    )
    log.info("-" * TABLE_WIDTH)


def log_progress(
    log: Any,
    engine: str,
    scenario: str,
    model: str,
    directory: Path,
    target: int,
) -> None:
    """Log progress for one model configuration."""
    completed = count_json_files(directory)
    percentage, status = compute_progress(completed, target)
    done_label = f"{completed:,} / {target:,}"
    log.info(
        f"{engine:<{ENGINE_WIDTH}}{SEPARATOR}"
        f"{scenario:<{SCENARIO_WIDTH}}{SEPARATOR}"
        f"{model:<{MODEL_WIDTH}}{SEPARATOR}"
        f"{done_label:>{DONE_WIDTH}}{SEPARATOR}"
        f"{percentage:>{PROGRESS_WIDTH - 1}}%{SEPARATOR}"
        f"{status:<{STATUS_WIDTH}}"
    )


def report_llm(log: Any, config: Mapping[str, Any]) -> None:
    """Report LLM strategy, benchmark, and consensus progress."""
    output_dir = Path(config.get("output_dir_base", "results/llm"))
    raw_root = output_dir / "inferences" / "raw"
    target = get_target_count(config)
    benchmark_models = config.get("benchmark_models", [])
    strategies = config.get("benchmark_strategies", {})
    # Check every configured strategy against every benchmark model.
    for strategy in strategies:
        for model in benchmark_models:
            log_progress(
                log,
                "LLM",
                str(strategy),
                str(model).rsplit("/", maxsplit=1)[-1],
                raw_root / str(strategy) / sanitize_filename(str(model)),
                target,
            )
    log.info("-" * TABLE_WIDTH)
    # Use the instructor-guidelines output directory for full evaluation models.
    baseline_scenario = "with_instructor_with_guidelines"
    baseline_root = raw_root / baseline_scenario
    for model in config.get("full_eval_models", []):
        log_progress(
            log,
            "LLM",
            baseline_scenario,
            str(model).rsplit("/", maxsplit=1)[-1],
            baseline_root / sanitize_filename(str(model)),
            target,
        )
    log.info("-" * TABLE_WIDTH)
    # Route temperature-one consensus to baseline outputs,
    # other temperatures have separate folders.
    consensus_root = output_dir / "inferences" / "consensus_raw"
    for temperature in config.get("consensus_temperatures", [1.0]):
        temp_label = str(temperature)
        scenario = f"consensus/temp_{temp_label}"
        for model in get_consensus_models(config):
            if Decimal(str(temperature)) == Decimal(1):
                directory = baseline_root / sanitize_filename(model)
            else:
                directory = (
                    consensus_root / f"temp_{temp_label}" / sanitize_filename(model)
                )
            log_progress(
                log,
                "LLM",
                scenario,
                str(model).rsplit("/", maxsplit=1)[-1],
                directory,
                target,
            )


def report_gliner(log: Any, config: Mapping[str, Any], target: int) -> None:
    """Report zero-shot and fine-tuned GLiNER progress."""
    groups = (
        ("zero-shot", False),
        ("fine-tuning", True),
    )
    # Reuse the same reporting logic for zero-shot and fine-tuning models.
    for scenario, trainable in groups:
        log.info("-" * TABLE_WIDTH)
        for model_key, model_config in config.items():
            if not isinstance(model_config, Mapping):
                # Skip invalid model configurations.
                continue
            if bool(model_config.get("is_trainable", False)) != trainable:
                # Skip models that don't match the current scenario.
                continue
            model_name = str(model_config.get("display_name", model_key))
            directory = Path("results/gliner/inferences") / str(model_key)
            log_progress(log, "GLiNER", scenario, model_name, directory, target)


def report_mdverse(log: Any, config: Mapping[str, Any]) -> None:
    """Report MDverse annotation progress for enabled engines."""
    target = count_json_files(Path(config.get("mdverse_dir")))
    models = config.get("models", {})
    result_directories = {
        "llm": Path("results/mdverse/llm"),
        "gliner": Path("results/mdverse/gliner"),
    }
    for engine in config.get("engines"):
        model_config = models.get(engine, {})
        if engine == "llm":
            model_name = f"{model_config.get('model_name')} "
        else:
            model_name = str(model_config.get("model_path"))
        log.info("-" * TABLE_WIDTH)
        display_engine = "GLiNER" if engine == "gliner" else "LLM"
        log_progress(
            log,
            display_engine,
            "MDVerse Annotation",
            model_name,
            result_directories[engine],
            target,
        )


@click.command()
@click.option(
    "--llm-config",
    type=click.Path(exists=True, path_type=Path),
    required=None,
    help="Path to the LLM benchmark configuration.",
)
@click.option(
    "--gliner-config",
    type=click.Path(exists=True, path_type=Path),
    required=None,
    help="Path to the GLiNER models configuration.",
)
@click.option(
    "--mdverse-config",
    type=click.Path(exists=True, path_type=Path),
    required=None,
    help="Path to the MDverse annotation configuration.",
)
def main(
    llm_config: Path | None, gliner_config: Path | None, mdverse_config: Path | None
) -> None:
    """Report progress for all inference and annotation workflows."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%Hh%Mm:%Ss")
    log = create_logger(f"logs/snakemake/check_progress/{timestamp}.log")
    # Display header and timestamp for the progress report.
    log.info("=" * TABLE_WIDTH)
    log.info("Check Pipeline Progress")
    log.info(timestamp)
    log.info("=" * TABLE_WIDTH)
    log_table_header(log)
    # 1. LLM Benchmark
    if llm_config:
        llm_settings = load_config(llm_config)
        report_llm(log, llm_settings)
    # 2. GLiNER Models
    if gliner_config:
        gliner_settings = load_config(gliner_config)
        # Expected samples inferred from LLM settings if available;
        # not defined in model config (set in training).
        # Fallback to 160.
        target = get_target_count(llm_settings) if llm_settings else 160
        report_gliner(log, gliner_settings, target)
    # 3. MDverse Annotation
    if mdverse_config:
        mdverse_settings = load_config(mdverse_config)
        report_mdverse(log, mdverse_settings)
    log.info("=" * TABLE_WIDTH)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
