"""Run multiple LLM extraction jobs in parallel for different models and frameworks."""

import itertools
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from mdner_llm.core.extract_entities_with_llm_all_texts import (
    extract_entities_all_texts,
)
from mdner_llm.utils.common import sanitize_filename
from mdner_llm.utils.logger import create_logger

MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4.6",
    "z-ai/glm-5.1",
    "openai/gpt-5.4",
    "deepseek/deepseek-v3.2",
    "google/gemma-4-31b-it",
    "meta-llama/llama-4-maverick",
    "nvidia/nemotron-3-super-120b-a12b",
    "mistralai/mistral-large-2512",
    "minimax/minimax-m2.7",
    "openai/gpt-oss-120b",
    "qwen/qwen3.5-122b-a10b",
    "google/gemini-3.1-pro-preview",
]

FRAMEWORKS = [
    "noframework",
    "instructor",
    "pydanticai",
]

PROMPT_FILE = Path("json_few_shot.txt")
TEXT_PATH = Path("data/groundtruth_5_paths.txt")
OUTPUT_DIR = Path("results/llm/annotations_new")
MAX_RETRIES = 3


def run_job(model: str, framework: str) -> str:
    """Execute a single extraction job.

    Returns
    -------
    str:
        A message indicating completion of the job with the model and framework used.
    """
    logger = create_logger(
        f"logs/extract_entities_all_texts_{sanitize_filename(model)}_{framework}.log",
        level="ERROR",
    )
    extract_entities_all_texts(
        prompt_file=PROMPT_FILE,
        model=model,
        texts_path=TEXT_PATH,
        framework=framework,
        output_dir=OUTPUT_DIR,
        max_retries=MAX_RETRIES,
        logger=logger,
    )
    return f"Extraction completed for model '{model}' with '{framework}'."


def generate_jobs() -> Iterable[tuple[str, str]]:
    """Generate all (model, framework) combinations.

    Returns
    -------
    Iterable[tuple[str, str]]:
        An iterable of (model, framework) tuples.
    """
    return itertools.product(MODELS, FRAMEWORKS)


def main(max_workers: int) -> None:
    """Run jobs in parallel."""
    jobs = list(generate_jobs())
    print(
        f"Running {len(jobs)} jobs "
        f"({len(MODELS)} models, {len(FRAMEWORKS)} frameworks) in parallel..."
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_job, model, framework) for model, framework in jobs
        ]

        for future in as_completed(futures):
            print(future.result())
    print("All jobs completed!")


@click.command()
@click.option("--max-workers", default=5, type=int, help="Number of parallel jobs.")
def run_main_from_cli(max_workers: int) -> None:
    """Run the main function from the command line."""
    main(max_workers)


if __name__ == "__main__":
    run_main_from_cli()
