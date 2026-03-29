"""Training GLINER2 model to fine-tune on MolecularDynamics-specific NER tasks."""

import io
import json
import os
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
) -> TrainingDataset:
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
    """
    logger.info(f"Creating dataset from annotation paths file: {annotation_paths_file}")
    train_examples = []
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
    return dataset


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


def split_and_save_dataset(
    dataset: TrainingDataset,
    config_data: GLiNERConfig,
    logger: "loguru.Logger" = loguru.logger,
):
    """
    Split dataset into train/validation/test sets and save them to disk.

    Parameters
    ----------
    dataset : TrainingDataset
        Validated GLiNER dataset to split.
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

    train_data, val_data, test_data = dataset.split(
        train_ratio=config_data.train_ratio,
        val_ratio=config_data.val_ratio,
        test_ratio=config_data.test_ratio,
        shuffle=config_data.shuffle,
        seed=config_data.seed,
    )
    with suppress_stdout():
        train_data.save(config_data.train_data_path)
    logger.success(
        f"Saved {len(train_data)} training examples to {config_data.train_data_path}"
    )
    with suppress_stdout():
        val_data.save(config_data.val_data_path)
    logger.success(
        f"Saved {len(val_data)} validation examples to {config_data.val_data_path}"
    )
    with suppress_stdout():
        test_data.save(config_data.test_data_path)
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
    results: dict, output_dir: Path, logger: "loguru.Logger"
) -> None:
    """Plot evaluation loss as a function of epochs."""
    # Extract training history from results
    train_history = results.get("train_metrics_history", [])
    eval_history = results.get("eval_metrics_history", [])
    # Extract total epochs and time for title
    total_epochs = results.get("total_epochs", "?")
    total_time = results.get("total_time_seconds", 0)

    # Prepare data for plotting
    train_epochs = [int(entry["epoch"]) for entry in train_history]
    train_losses = [entry["loss"] for entry in train_history]

    eval_epochs = [int(entry["epoch"]) for entry in eval_history]
    eval_losses = [entry["eval_loss"] for entry in eval_history]

    # Find minimum eval loss and corresponding epoch for annotation
    if eval_losses:
        min_loss = min(eval_losses)
        min_idx = eval_losses.index(min_loss)
        min_epoch = eval_epochs[min_idx]
    else:
        min_loss = None
        min_epoch = None

    # Create subplots for train and eval loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Train subplot
    if train_losses:
        axes[0].plot(train_epochs, train_losses, marker="o")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    # Eval subplot
    if eval_losses:
        axes[1].plot(eval_epochs, eval_losses, marker="o")

        # Highlight minimum
        axes[1].axhline(min_loss, linestyle="--")
        axes[1].annotate(
            f"min={min_loss:.2f}",
            xy=(min_epoch, min_loss),
            xytext=(min_epoch, min_loss * 1.02),
        )

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    # Global title
    fig.suptitle(f"Training curves | epochs={total_epochs} | time={total_time:.1f}s")
    plt.tight_layout()

    # Save the plot to the output directory
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=300)

    # Display the plot
    plt.show()
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
    output_dir = Path(cfg.model.output_dir) / f"{cfg.model.experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create dataset
    # Build training examples from annotation files
    dataset = build_train_dataset(
        cfg.data.annotation_paths, entity_descriptions=cfg.entities, logger=logger
    )
    # Validate dataset
    validated_dataset = validate_dataset(dataset, logger)
    # Split dataset into train/val/test
    train_data, val_data, _test_data = split_and_save_dataset(
        validated_dataset, cfg.data, logger
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
    default="src/mdner_llm/gliner/training_config.yml",
    help="Path to the training config YAML file.",
)
def run_main_from_cli(config_path: str | Path):
    """Run the main function with config path from CLI."""
    main(config_path)


if __name__ == "__main__":
    run_main_from_cli()
