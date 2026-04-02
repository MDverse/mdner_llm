"""Training GLINER2 model to fine-tune on MolecularDynamics-specific NER tasks."""

import io
import json
import os
import random
import sys
from contextlib import contextmanager, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import click
import loguru
import yaml
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pydantic import ValidationError

from mdner_llm.core.logger import create_logger
from mdner_llm.gliner.training_models import GLiNERConfig


def load_config(
    config_path: str | Path, logger: "loguru.Logger" = loguru.logger
) -> GLiNERConfig | None:
    """
    Load and validate YAML configuration for GLiNER training.

    Parameters
    ----------
    config_path : str | Path
        Path to YAML configuration file.

    Returns
    -------
    GLiNERConfig | None
        Validated configuration object or None if loading fails.
    """
    logger.info(f"Loading config from: {config_path}")
    # Ensure config_path is a Path object
    config_path = Path(config_path)
    # Ensure config file exists
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return None
    try:
        # Load raw config from YAML
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)

        if raw_config is None:
            logger.warning("Config file is empty.")
            return None

        # Validate config through Pydantic model
        validated_config = GLiNERConfig.model_validate(raw_config)

    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML config: {exc}")
        return None

    except ValidationError as exc:
        logger.error(f"Config validation error: {exc}")
        return None

    else:
        for field_name, field_value in validated_config.model_dump().items():
            logger.debug(f"Config - {field_name}: {field_value}")

        logger.success("Training config loaded and validated successfully!")
        return validated_config


def log_dataset_stats(dataset, logger) -> None:
    """Log statistics about the training dataset."""
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        dataset.print_stats()
    stats = buffer.getvalue().strip()
    for line in stats.splitlines():
        logger.info(line)


def filter_entity_descriptions(
    entities: dict[str, list[str]],
    descriptions: dict[str, str] | None,
) -> dict[str, str]:
    """Filter entity descriptions to include only those relevant to the entities.

    Parameters
    ----------
    entities : dict[str, list[str]]
        Dictionary of entity labels and their corresponding values in the dataset.
    descriptions : dict[str, str] | None
        Optional dictionary mapping entity labels to their descriptions.
        If None, no filtering is applied and an empty dictionary is returned.

    Returns
    -------
    dict[str, str]
        Filtered dictionary of entity descriptions.
    """
    if not descriptions:
        return {}
    return {key: value for key, value in descriptions.items() if key in entities}


def build_example(
    annotation_path: Path,
    entity_descriptions: dict[str, str] | None,
) -> tuple[InputExample, str]:
    """Build a single InputExample from a JSON annotation file.

    Parameters
    ----------
    annotation_path : Path
        Path to the JSON annotation file containing raw text and entity annotations.
    entity_descriptions : dict[str, str] | None
        Optional dictionary mapping entity labels to their descriptions.

    Returns
    -------
    tuple[InputExample, str]
        A tuple containing the constructed InputExample and an optional URL
        if present in the annotation.
    """
    # Read the annotation JSON file
    with annotation_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)
    # Extract raw text and entities from the JSON data
    raw_text = json_data["raw_text"]
    raw_entities = json_data.get("entities", [])
    url = json_data.get("url")
    # Format entities into the structure expected by InputExample
    formatted_entities = {}
    for ent in raw_entities:
        label = ent["label"]
        value = ent["text"]
        formatted_entities.setdefault(label, [])
        if value not in formatted_entities[label]:
            formatted_entities[label].append(value)
    # Create an InputExample
    example = InputExample(
        # with the raw text
        text=raw_text,
        # the formatted entities (label -> list of values)
        entities=formatted_entities,
        # and the filtered entity descriptions (only for labels present in the dataset)
        entity_descriptions=filter_entity_descriptions(
            formatted_entities, entity_descriptions
        ),
    )

    return example, url


def build_train_dataset(
    annotation_paths_file: Path,
    entity_descriptions: dict[str, str] | None = None,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[TrainingDataset, list[Path], list[str]]:
    """
    Build a TrainingDataset from annotation JSON files specified in a text file.

    Parameters
    ----------
    annotation_paths_file : Path
        Path to a text file containing paths to annotation JSON files (one per line).
    entity_descriptions : dict[str, str] | None
        Optional mapping from label to description.

    Returns
    -------
    TrainingDataset
        Training dataset containing the formatted InputExample objects.
    list[Path]
        List of annotation paths that were successfully processed.
    urls : list[str]
        List of URLs extracted from the annotation files (if present).
    """
    logger.info(f"Creating dataset from annotation paths file: {annotation_paths_file}")
    train_examples = []
    processed_annotation_paths = []
    urls = []
    first_logged = False
    # Read annotation file paths from the provided text file
    with annotation_paths_file.open("r", encoding="utf-8") as file:
        selected_annotation_paths = [
            Path(line.strip()) for line in file if line.strip()
        ]
    # Process each annotation file to build InputExample objects
    for annotation_path in selected_annotation_paths:
        # Check if the annotation file exists before attempting to read it
        if not annotation_path.exists():
            logger.warning(f"Annotation file not found: {annotation_path}")
            continue
        # Build an InputExample from the annotation file
        example, url = build_example(annotation_path, entity_descriptions)
        # Add it to the list of training examples
        train_examples.append(example)
        processed_annotation_paths.append(annotation_path)
        urls.append(url or "")
        # Log the first example for debugging purposes
        if not first_logged:
            first_logged = True
            logger.info("First training example:")
            if url:
                logger.info(f"URL: {url}")
            logger.info(f"Text: {example.text.replace('\n', ' ')[:70]}...")
            logger.info("Entities:")
            for label, values in example.entities.items():
                logger.info(f"  {label}: {values}")
            if entity_descriptions:
                logger.info("Entity Descriptions:")
                for label, desc in entity_descriptions.items():
                    logger.info(f"  {label}: {desc}")
    # Instantiate TrainingDataset with the list of InputExample objects
    dataset = TrainingDataset(train_examples)
    log_dataset_stats(dataset, logger)
    logger.success(f"Created dataset with {len(train_examples)} examples successfully!")
    return dataset, processed_annotation_paths, urls


def validate_dataset(
    dataset: TrainingDataset, logger: "loguru.Logger" = loguru.logger
) -> TrainingDataset:
    """
    Validate the TrainingDataset for consistency and correctness.

    Parameters
    ----------
    dataset : TrainingDataset
        The dataset to validate.
    logger : loguru.Logger
        Logger for logging validation results.

    Returns
    -------
    TrainingDataset
        The original dataset if validation passes.
    """
    logger.info("Validating training dataset...")
    dataset.validate(raise_on_error=True)
    logger.success("Training dataset validation passed successfully!")
    return dataset


@contextmanager
def suppress_stdout():
    """Temporarily suppress stdout (used for noisy library prints)."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def split_randomly(
    elements: list,
    config_data: GLiNERConfig,
) -> tuple[list, list, list]:
    """
    Split a list into train/val/test subsets using deterministic indexing.

    Parameters
    ----------
    elements : list
        List of elements to split.
    config_data : GLiNERConfig
        Configuration object containing split ratios and seed.

    Returns
    -------
    tuple
        The list split into (train, val, test)
        according to the specified ratios and shuffled using the
        provided seed for reproducibility.
    """
    indices = list(range(len(elements)))

    if config_data.shuffle:
        random.seed(config_data.seed)
        random.shuffle(indices)

    n = len(indices)
    train_end = int(n * config_data.train_ratio)
    val_end = train_end + int(n * config_data.val_ratio)

    train_elements = [elements[i] for i in indices[:train_end]]
    val_elements = [elements[i] for i in indices[train_end:val_end]]
    test_elements = [elements[i] for i in indices[val_end:]]

    return train_elements, val_elements, test_elements


def save_paths_txt(
    paths: list[Path],
    urls: list[str],
    target_path: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """
    Save a list of paths with urls to a .txt file, one path per line.

    The output file is derived from the dataset path by replacing its suffix.
    """
    txt_path = target_path.with_name(f"{target_path.stem}_metadata.txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(f"{p}\t{url}\n" for p, url in zip(paths, urls, strict=False))
    logger.debug(
        f"Saved {len(paths)} {target_path.stem} examples metadata to {txt_path}"
    )


def check_alignment(
    train_data: TrainingDataset,
    train_paths: list[Path],
    logger: "loguru.Logger" = loguru.logger,
) -> list[dict[str, int | Path]] | None:
    """Check alignment between train_data inputs and raw_text in JSON files.

    Parameters
    ----------
    train_data : TrainingDataset
        Training data containing an "input" key with text entries.
    train_paths : list[Path]
        List of paths to JSON files containing a "raw_text" field.
    logger : loguru.Logger, optional
        Logger instance.

    Returns
    -------
    list[dict[str, Any]] | None
        A list of mismatches with index and path if any inconsistency is found,
        otherwise None.
    """
    mismatches = []
    inputs = [example.text for example in train_data]

    for idx, (expected_text, path) in enumerate(zip(inputs, train_paths, strict=False)):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            raw_text = data["raw_text"]
        except OSError as exc:
            logger.warning(f"I/O error at index {idx} for {path}: {exc}")
            mismatches.append({"index": idx, "path": path})
            continue
        except json.JSONDecodeError as exc:
            logger.warning(f"Invalid JSON at index {idx} for {path}: {exc}")
            mismatches.append({"index": idx, "path": path})
            continue
        except KeyError:
            logger.warning(f"Missing 'raw_text' in {path} (index {idx})")
            mismatches.append({"index": idx, "path": path})
            continue

        if raw_text != expected_text:
            logger.warning(f"Mismatch at index {idx} for file {path}")
            mismatches.append({"index": idx, "path": path})

    return mismatches or None


def split_and_save_dataset(
    dataset: TrainingDataset,
    selected_annotation_paths: list[Path],
    urls: list[str],
    config_data: GLiNERConfig,
    logger: "loguru.Logger" = loguru.logger,
):
    """
    Split dataset into train/validation/test sets and save them to disk.

    Parameters
    ----------
    dataset : TrainingDataset
        Validated GLiNER dataset to split.
    selected_annotation_paths : list[Path]
        List of annotation paths that were successfully processed.
    urls : list[str]
        List of URLs extracted from the annotation files (if present).
    config_data : DataConfig
        Configuration object containing split ratios, paths and seed.
    logger : loguru.Logger
        Logger instance.

    Returns
    -------
    tuple
        (train_data, val_data, test_data) - the split datasets.
    """
    logger.info(
        f"Splitting dataset of {len(dataset)} examples into train/val/test sets"
    )
    logger.info(
        f"Ratios - Train: {config_data.train_ratio}, "
        f"Val: {config_data.val_ratio}, "
        f"Test: {config_data.test_ratio}"
    )
    # Split the dataset into train/validation/test sets
    # using the specified ratios and seed
    train_data, val_data, test_data = dataset.split(
        train_ratio=config_data.train_ratio,
        val_ratio=config_data.val_ratio,
        test_ratio=config_data.test_ratio,
        shuffle=config_data.shuffle,
        seed=config_data.seed,
    )
    # Split the list of annotation paths in the same way
    # to keep track of which examples belong to which set
    train_paths, val_paths, test_paths = split_randomly(
        selected_annotation_paths, config_data
    )
    # Split the list of URLs in the same way (if present)
    train_urls, val_urls, test_urls = split_randomly(urls, config_data)

    with suppress_stdout():
        train_data.save(config_data.train_data_path)
        issues = check_alignment(train_data, train_paths, logger)
        if not issues:
            save_paths_txt(train_paths, train_urls, config_data.train_data_path, logger)
        logger.success(
            f"Saved {len(train_data)} training examples to "
            f"{config_data.train_data_path}"
        )
        val_data.save(config_data.val_data_path)
        issues = check_alignment(val_data, val_paths, logger)
        if not issues:
            save_paths_txt(val_paths, val_urls, config_data.val_data_path, logger)
        logger.success(
            f"Saved {len(val_data)} validation examples to {config_data.val_data_path}"
        )
        test_data.save(config_data.test_data_path)
        issues = check_alignment(test_data, test_paths, logger)
        if not issues:
            save_paths_txt(test_paths, test_urls, config_data.test_data_path, logger)
        logger.success(
            f"Saved {len(test_data)} test examples to {config_data.test_data_path}"
        )
    return train_data, val_data, test_data


def load_model_and_config(
    model_name: str,
    config: GLiNERConfig,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[GLiNER2, TrainingConfig]:
    """
    Load a GLiNER2 model and the trainfing configuation for training.

    Parameters
    ----------
    model_name : str
        Name of the pretrained model.
    config : GLiNERConfig
        Training configuration.
    logger : loguru.Logger
        Logger instance.

    Returns
    -------
    tuple
        (model, training_config)
            The loaded GLiNER2 model and the training configuration.
    """
    # Capture stdout during model loading
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model = GLiNER2.from_pretrained(model_name)
    # Display captured output in logger
    output = buffer.getvalue().strip()
    if output:
        for line in output.splitlines():
            logger.info(line.strip())

    training_config = TrainingConfig(
        # Model & output
        output_dir=f"{config.model.output_dir!s}/{config.model.experiment_name}",
        experiment_name=config.model.experiment_name,
        # Training schedule
        num_epochs=config.training.num_epochs,
        max_steps=config.training.max_steps,
        batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        # Learning rates
        encoder_lr=config.training.encoder_lr,
        task_lr=config.training.task_lr,
        warmup_ratio=config.training.warmup_ratio,
        scheduler_type=config.training.scheduler_type,
        # Optimization
        weight_decay=config.training.weight_decay,
        # Precision / hardware
        fp16=config.training.fp16,
        # Checkpointing
        eval_strategy=config.training.eval_strategy,
        metric_for_best=config.training.metric_for_best,
        save_best=config.training.save_best,
        # Logging
        logging_steps=config.training.logging_steps,
    )

    return model, training_config


def train_gliner_model(
    model: GLiNER2,
    train_dataset: TrainingDataset,
    eval_dataset: TrainingDataset,
    training_config: TrainingConfig,
    logger: "loguru.Logger" = loguru.logger,
) -> dict:
    """
    Train the GLiNER2 model using the provided training and evaluation datasets.

    Parameters
    ----------
    model : GLiNER2
        The GLiNER2 model to train.
    train_dataset : TrainingDataset
        The dataset to use for training.
    eval_dataset : TrainingDataset
        The dataset to use for evaluation during training.
    training_config : TrainingConfig
        Configuration object containing training parameters.
    logger : loguru.Logger
        Logger instance for logging training progress and results.

    Returns
    -------
    dict
        Dictionary containing training results and metrics.
    """
    logger.info("Starting GLiNER2 training...")
    trainer = GLiNER2Trainer(model, training_config)
    results = trainer.train(train_data=train_dataset, eval_data=eval_dataset)
    logger.info("Training results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")
    logger.success("GLiNER2 training completed successfully!")
    return results


def save_plot_training_curves(
    results: dict,
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Plot training and evaluation loss curves with improved styling."""
    # Extract training history, evaluation history, total epochs and total time
    train_history = results.get("train_metrics_history", [])
    eval_history = results.get("eval_metrics_history", [])
    total_epochs = results.get("total_epochs", "?")
    total_time = results.get("total_time_seconds", 0)
    train_epochs = [int(entry["epoch"]) for entry in train_history]
    train_losses = [entry["loss"] for entry in train_history]
    eval_epochs = [int(entry["epoch"]) for entry in eval_history]
    eval_losses = [entry["eval_loss"] for entry in eval_history]
    # Identify minimum eval loss and corresponding epoch
    if eval_losses:
        min_loss = min(eval_losses)
        min_idx = eval_losses.index(min_loss)
        min_epoch = eval_epochs[min_idx]
    else:
        min_loss = None
        min_epoch = None

    # Identify minimum train loss and corresponding epoch
    if train_losses:
        min_train_loss = min(train_losses)
        min_train_idx = train_losses.index(min_train_loss)
        min_train_epoch = train_epochs[min_train_idx]
    else:
        min_train_loss = None
        min_train_epoch = None
    # Plotting
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Train
    if train_losses:
        ax.plot(
            train_epochs,
            train_losses,
            marker="o",
            linewidth=2,
            markersize=4,
            color="blue",
            label="Train",
        )
        ax.axvline(min_train_epoch, linestyle=":", linewidth=1.5, color="blue")
        ax.annotate(
            f"min={min_train_loss:.3f}\nepoch={min_train_epoch}",
            xy=(min_train_epoch, min_train_loss),
            xytext=(min_train_epoch + 0.2, min_train_loss * 5),
            textcoords="data",
            arrowprops={"arrowstyle": "->", "linewidth": 1, "color": "blue"},
            fontsize=9,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "alpha": 0.4,
                "facecolor": "blue",
                "edgecolor": "blue",
            },
            color="blue",
        )

    # Eval
    if eval_losses:
        ax.plot(
            eval_epochs,
            eval_losses,
            marker="o",
            linewidth=2,
            markersize=4,
            color="orange",
            label="Validation",
        )
        ax.axvline(min_epoch, linestyle=":", linewidth=1.5, color="orange")
        ax.annotate(
            f"min={min_loss:.3f}\nepoch={min_epoch}",
            xy=(min_epoch, min_loss),
            xytext=(min_epoch + 0.4, min_loss - 50),
            textcoords="data",
            arrowprops={"arrowstyle": "->", "linewidth": 1, "color": "orange"},
            fontsize=9,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "alpha": 0.4,
                "facecolor": "orange",
                "edgecolor": "orange",
            },
            color="orange",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()

    # Overall title and layout
    fig.suptitle(
        f"Training and Validation Loss (epochs={total_epochs}, "
        f"duration={total_time:.1f}s)",
        fontsize=12,
    )
    plt.tight_layout()
    # Save the plot to the output directory
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.success(f"Saved training curves plot to {output_path}")


def main(config_path: str | Path):
    """Train GLINER2 model using the specified training configuration."""
    # Initialize logger
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    logger = create_logger(f"logs/train_gliner_{timestamp}.log")
    # Load config
    cfg = load_config(config_path, logger=logger)
    if not cfg:
        logger.error("Failed to load training configuration.")
        logger.error("Exiting training process.")
        return
    # Setup output directory
    output_dir = Path(cfg.model.output_dir) / f"{cfg.model.experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create dataset
    # Build training examples from annotation files
    dataset, selected_annotation_paths, urls = build_train_dataset(
        cfg.data.annotation_paths, entity_descriptions=cfg.entities, logger=logger
    )
    # Validate dataset
    validated_dataset = validate_dataset(dataset, logger)
    # Split dataset into train/val/test
    train_data, val_data, _test_data = split_and_save_dataset(
        validated_dataset, selected_annotation_paths, urls, cfg.data, logger
    )
    # Build model and configure training parameters
    model, training_config = load_model_and_config(cfg.model.name, cfg, logger)
    # Train model
    results = train_gliner_model(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        training_config=training_config,
        logger=logger,
    )
    # Plot training curves
    save_plot_training_curves(results, output_dir, logger)
    logger.success(f"✓ Training complete! Model saved to {output_dir}")


@click.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default="src/mdner_llm/gliner/training_config.yaml",
    help="Path to the training config YAML file.",
)
def run_main_from_cli(config_path: str | Path):
    """Run the main function with config path from CLI."""
    main(config_path)


if __name__ == "__main__":
    run_main_from_cli()
