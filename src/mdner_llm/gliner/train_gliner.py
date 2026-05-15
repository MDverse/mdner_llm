"""Training GLINER2 model to fine-tune on Molecular Dynamics-specific NER tasks."""

import json
import operator
import random
from collections import defaultdict
from pathlib import Path

import click
import loguru
import numpy as np
import yaml
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from pydantic import ValidationError

from mdner_llm.gliner.training_models import GLiNERConfig
from mdner_llm.logger import create_logger


def load_config(
    config_path: str | Path, logger: "loguru.Logger" = loguru.logger
) -> GLiNERConfig | None:
    """
    Load and validate YAML configuration for GLiNER training.

    Returns
    -------
    GLiNERConfig | None
        Validated configuration object or None if loading fails.
    """
    logger.info(f"Loading config from: {config_path}.")
    # Ensure config_path is a Path object
    config_path = Path(config_path)
    # Ensure config file exists
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}.")
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
        logger.success("Training config loaded and validated successfully!")
        return validated_config


def build_example(
    annotation_path: Path,
    entity_descriptions: dict[str, str] | None,
) -> tuple[InputExample, str]:
    """Build a single InputExample from a JSON annotation file.

    Returns
    -------
    tuple[InputExample, str]
        A tuple containing the constructed InputExample and an optional URL
        if present in the annotation.
    """
    # Read the annotation JSON file
    with annotation_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)
    # Extract raw text, entities and url from the JSON data
    raw_text = json_data["raw_text"]
    raw_entities = json_data.get("entities", [])
    # Format entities into the structure expected by InputExample
    formatted_entities = defaultdict(list)
    for entity in raw_entities:
        if entity["text"] not in formatted_entities[entity["category"]]:
            formatted_entities[entity["category"]].append(entity["text"])
    # Filter entity descriptions to include only those relevant to the dataset
    filtered_entity_descriptions = (
        {
            key: description
            for key, description in entity_descriptions.items()
            if key in formatted_entities
        }
        if entity_descriptions
        else {}
    )
    # Create an InputExample
    example = InputExample(
        # with the raw text
        text=raw_text,
        # the formatted entities (category -> list of values)
        entities=formatted_entities,
        # and the filtered entity descriptions
        # (only for categories present in the dataset)
        entity_descriptions=filtered_entity_descriptions,
    )
    return example, json_data.get("url", "")


def build_train_dataset(
    annotations_path: Path,
    entity_descriptions: dict[str, str] | None = None,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[TrainingDataset, list[Path], list[str]]:
    """
    Build a TrainingDataset from annotation JSON files specified in a text file.

    Returns
    -------
    TrainingDataset
        Training dataset containing the formatted InputExample objects.
    list[Path]
        List of annotation paths that were successfully processed.
    urls : list[str]
        List of URLs extracted from the annotation files (if present).
    """
    logger.info(f"Creating dataset from annotation paths file: {annotations_path}.")
    train_examples = []
    processed_annotation_paths = []
    urls = []
    first_logged = False
    # Process each annotation file to build InputExample objects
    for annotation_path in list(annotations_path.glob("*.json")):
        # Check if the annotation file exists before attempting to read it
        if not annotation_path.exists():
            logger.warning(f"Annotation file not found: {annotation_path}.")
            continue
        # Build an InputExample from the annotation file
        example, url = build_example(annotation_path, entity_descriptions)
        # Add it to the list of training examples
        train_examples.append(example)
        processed_annotation_paths.append(annotation_path)
        urls.append(url)
        # Log the first example for debugging purposes
        if not first_logged:
            first_logged = True
            logger.info("First training example:")
            if url:
                logger.info(f"URL: {url}")
            logger.info(f"Text: {example.text.replace('\n', ' ')[:70]}...")
            logger.info("Entities:")
            for category, values in example.entities.items():
                logger.info(f"  {category}: {values}")
            if entity_descriptions:
                logger.info("Entity Descriptions:")
                for category, desc in entity_descriptions.items():
                    logger.info(f"  {category}: {desc}")
    # Instantiate TrainingDataset with the list of InputExample objects
    dataset = TrainingDataset(train_examples)
    dataset.print_stats()
    logger.success(f"Created dataset with {len(train_examples)} examples successfully!")
    return dataset, processed_annotation_paths, urls


def check_alignment(
    train_data: TrainingDataset,
    train_paths: list[Path],
    logger: "loguru.Logger" = loguru.logger,
) -> list[dict[str, int | Path]] | None:
    """Check alignment between train_data inputs and raw_text in JSON files.

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


def save_dataset_to_jsonl(
    dataset: TrainingDataset, path: Path, logger: "loguru.Logger" = loguru.logger
) -> None:
    """Serialize a TrainingDataset to a JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            for example in dataset:
                record = {
                    "input": example.text,
                    "output": {
                        "entities": example.entities,
                        "entity_descriptions": example.entity_descriptions,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error(f"Failed to save dataset to {path}: {exc}.")


def save_metadata_to_txt(
    paths: list[Path],
    urls: list[str],
    target: Path,
) -> None:
    """Save paths and urls to a .txt file aligned with the dataset JSONL file."""
    out = target.with_name(f"{target.stem}_metadata.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(f"{path}\t{url}\n" for path, url in zip(paths, urls, strict=False))


def k_fold_split(
    dataset: TrainingDataset,
    paths: list[Path],
    urls: list[str],
    cfg: GLiNERConfig,
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> list[tuple[TrainingDataset, TrainingDataset]]:
    """
    Split the full dataset into K folds for nested cross-validation.

    For each fold:
      - one bucket is used as the test set
      - all remaining buckets are merged into a train/validation pool
      - train, validation and test splits are saved as JSONL with metadata

    Returns
    -------
    list of (fold_train, fold_val)
    """
    k = cfg.training.cv_folds
    seed = cfg.data.seed
    logger.info(f"Splitting dataset into {k} folds for nested CV.")
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    # Split shuffled indices into K buckets.
    fold_size = len(indices) // k
    buckets = [indices[i * fold_size : (i + 1) * fold_size] for i in range(k)]
    # If the dataset size is not divisible by K, remaining examples are added
    # to the last bucket.
    buckets[-1].extend(indices[k * fold_size :])

    folds = []
    for fold_id, test_idx in enumerate(buckets, start=1):
        # Create the output folder for the current fold.
        fold_dir = output_dir / f"fold_{fold_id}" / "data"
        fold_dir.mkdir(parents=True, exist_ok=True)
        # Use the current bucket as test set.
        # All other buckets are merged into the train/validation pool.
        trainval_idx = [
            j for bucket in buckets if bucket is not test_idx for j in bucket
        ]
        # Shuffle the train/validation pool before splitting it.
        random.shuffle(trainval_idx)
        # Compute where the train split ends.
        train_end = int(len(trainval_idx) * cfg.data.train_ratio)
        # Store split indices
        split_indices = {
            "train": trainval_idx[:train_end],
            "val": trainval_idx[train_end:],
            "test": test_idx,
        }
        # Build actual TrainingDataset objects from the selected indices.
        splits = {
            name: TrainingDataset([dataset[j] for j in idxs])
            for name, idxs in split_indices.items()
        }
        logger.info(
            f"Fold {fold_id}/{k} — "
            f"train={len(splits['train'])}, "
            f"val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )
        # Save each split and its metadata.
        for name, split_ds in splits.items():
            save_path = fold_dir / f"{name}.jsonl"
            split_idx = split_indices[name]
            # Save the dataset split as JSONL.
            save_dataset_to_jsonl(split_ds, save_path, logger)
            # Select the source paths and URLs matching the examples in this split.
            split_paths = [paths[j] for j in split_idx]
            split_urls = [urls[j] for j in split_idx]
            # Check that dataset examples and source paths are still aligned
            issues = check_alignment(split_ds, split_paths, logger)
            if issues:
                logger.warning(
                    f"Alignment check failed for {name} split of fold {fold_id}. "
                    f"Metadata file will not be saved for this split."
                )
                for issue in issues:
                    logger.debug(
                        f"  Mismatch at index {issue['index']} for file {issue['path']}"
                    )
                continue
            # Save metadata only when the alignment check succeeds.
            save_metadata_to_txt(split_paths, split_urls, save_path)
        folds.append((splits["train"], splits["val"]))
    logger.success(
        f"Completed {k}-fold splitting and saving into {fold_dir} successfully!"
    )
    return folds


def build_training_config(
    config: GLiNERConfig,
    fold_output_dir: Path,
) -> TrainingConfig:
    """Build a TrainingConfig for GLiNER2Trainer.

    Returns
    -------
    TrainingConfig
        Configured TrainingConfig object with parameters from the GLiNERConfig.
    """
    return TrainingConfig(
        # Model & output
        output_dir=str(fold_output_dir),
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
        bf16=config.training.bf16,
        # LoRa
        use_lora=config.training.use_lora,
        lora_r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        lora_target_modules=config.training.lora_target_modules,
        save_adapter_only=config.training.save_adapter_only,
        # Checkpointing
        eval_strategy=config.training.eval_strategy,
        metric_for_best=config.training.metric_for_best,
        save_best=config.training.save_best,
        # Logging
        logging_steps=config.training.logging_steps,
    )


def train_gliner_model(
    model: GLiNER2,
    train_dataset: TrainingDataset,
    eval_dataset: TrainingDataset,
    training_config: TrainingConfig,
    logger: "loguru.Logger" = loguru.logger,
) -> dict:
    """
    Train the GLiNER2 model using the provided training and evaluation datasets.

    Returns
    -------
    dict
        Dictionary containing training results and metrics.
    """
    trainer = GLiNER2Trainer(model, training_config)
    results = trainer.train(train_data=train_dataset, eval_data=eval_dataset)
    logger.success("✓ Training complete successfully!")
    # Find the epoch with the lowest eval_loss from the results
    best_loss_epoch = min(
        results.get("eval_metrics_history", []),
        key=operator.itemgetter("eval_loss"),
    )["epoch"]
    logger.info(f"Duration: {results.get('total_time_seconds')} seconds")
    logger.info(f"Total steps: {results.get('total_steps')}")
    logger.info(f"Total epoch: {results.get('total_epochs')}")
    logger.info(f"Lowest loss: {results.get('best_metric')} at epoch {best_loss_epoch}")
    logger.success(
        f"Best model saved to {training_config.output_dir}/best successfully!"
    )
    return results


def save_loss_points(
    results_list: list[dict],
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Save training and validation loss points across K-folds to a JSON file."""
    loss_points = []
    for fold_id, results in enumerate(results_list, start=1):
        train_history = results.get("train_metrics_history", [])
        eval_history = results.get("eval_metrics_history", [])
        for train_entry, eval_entry in zip(train_history, eval_history, strict=False):
            loss_points.append(
                {
                    "fold": fold_id,
                    "epoch": int(train_entry["epoch"]),
                    "train_loss": train_entry["loss"],
                    "eval_loss": eval_entry["eval_loss"],
                }
            )
    output_path = output_dir / "loss_points.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(loss_points, f, indent=2)
    logger.success(
        f"Saved training and validation loss points to {output_path} successfully!"
    )


def _generate_gradient_colors(
    base_color: str,
    n_colors: int,
    min_alpha: float = 0.35,
    max_alpha: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """Generate RGBA colors with opacity gradient.

    Returns
    -------
    list of RGBA tuples
    """
    r, g, b, _ = to_rgba(base_color)
    alphas = np.linspace(min_alpha, max_alpha, max(n_colors, 1))
    return [(r, g, b, a) for a in alphas]


def save_plot_training_curves(
    results_list: list[dict],
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Plot train/eval curves with fold opacity gradients."""
    _fig, ax = plt.subplots(figsize=(13, 7))
    # Styling
    ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Generate color gradients for train and eval curves based on the number of folds
    n_folds = len(results_list)
    train_colors = _generate_gradient_colors("#0055FF", n_folds)
    eval_colors = _generate_gradient_colors("#FFAA00", n_folds)
    # Track global minimum validation loss
    global_min = {"loss": float("inf"), "epoch": None, "fold": None, "color": None}
    for idx, results in enumerate(results_list):
        # Train curves
        train_history = results.get("train_metrics_history", [])
        eval_history = results.get("eval_metrics_history", [])
        ax.plot(
            [int(x["epoch"]) for x in train_history],
            [x["loss"] for x in train_history],
            color=train_colors[idx],
            linewidth=2,
            marker="o",
            markersize=4,
        )
        # Eval curves
        eval_epochs = [int(x["epoch"]) for x in eval_history]
        eval_losses = [x["eval_loss"] for x in eval_history]
        ax.plot(
            eval_epochs,
            eval_losses,
            color=eval_colors[idx],
            linewidth=2,
            marker="o",
            markersize=4,
        )
        # Update global minimum validation loss
        if eval_losses:
            min_idx = int(np.argmin(eval_losses))
            if eval_losses[min_idx] < global_min["loss"]:
                global_min = {
                    "loss": eval_losses[min_idx],
                    "epoch": eval_epochs[min_idx],
                    "fold": idx + 1,
                    "color": eval_colors[idx],
                }
    # Annotate only the global minimum validation loss
    if global_min["epoch"] is not None:
        ax.scatter(
            global_min["epoch"],
            global_min["loss"],
            s=120,
            color=global_min["color"],
            edgecolors="dimgrey",
            linewidths=1.5,
            zorder=10,
        )
        ax.annotate(
            f"Best validation loss\nFold: {global_min['fold']}\n"
            f"Epoch: {global_min['epoch']}\nLoss: {global_min['loss']:.2f}",
            xy=(global_min["epoch"], global_min["loss"]),
            xytext=(20, -20),
            textcoords="offset points",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": global_min["color"],
                "alpha": 0.95,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": global_min["color"],
                "linewidth": 1.5,
            },
        )

    # Custom legends for train and validation folds
    for handles, title, anchor, hex_color in [
        (
            [
                Line2D([0], [0], color=train_colors[i], lw=3, label=f"Fold {i + 1}")
                for i in range(n_folds)
            ],
            "Train",
            (1.00, 0.95),
            "#0055FF",
        ),
        (
            [
                Line2D([0], [0], color=eval_colors[i], lw=3, label=f"Fold {i + 1}")
                for i in range(n_folds)
            ],
            "Validation",
            (0.92, 0.95),
            "#FFAA00",
        ),
    ]:
        leg = ax.legend(
            handles=handles,
            title=title,
            loc="upper right",
            bbox_to_anchor=anchor,
            frameon=True,
            title_fontsize=11,
            fontsize=9,
        )
        leg.get_title().set_color(hex_color)
        ax.add_artist(leg)
    # Set title and axis labels
    last_epoch = max(
        int(x["epoch"])
        for results in results_list
        for x in results.get("train_metrics_history", [])
    )
    # compute duration by epoch
    sum_duration = np.sum([r.get("total_time_seconds", 0) for r in results_list])
    duration_per_epoch = sum_duration / last_epoch if last_epoch > 0 else 0
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(
        f"K-Fold Training Curves (folds={n_folds}, epochs={last_epoch}, "
        f"duration={duration_per_epoch:,.0f}s per epoch)",
        fontsize=13,
    )
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Saved CV training curves plot to {output_path} successfully!")


def main(config_path: str | Path) -> None:
    """Train GLINER2 model using the specified training configuration."""
    # Initialize logger
    logger = create_logger(level="DEBUG")
    logger.info("Starting GLiNER2 finetuning process.")
    # Load config
    cfg = load_config(config_path, logger=logger)
    if not cfg:
        logger.error("Failed to load training configuration.")
        logger.error("Exiting training process.")
        return
    # Setup output directory
    output_dir = Path(cfg.model.output_dir) / f"{cfg.model.experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create dataset from annotation files
    dataset, selected_annotation_paths, urls = build_train_dataset(
        cfg.data.annotations_path, entity_descriptions=cfg.entities, logger=logger
    )
    # Validate dataset
    dataset.validate(raise_on_error=True)
    # K-fold nested CV directly on the full dataset
    folds = k_fold_split(
        dataset,
        selected_annotation_paths,
        urls,
        cfg,
        output_dir,
        logger,
    )
    # Train a separate model for each fold
    all_results = []
    for fold_id, (train_data, val_data) in enumerate(folds, start=1):
        logger.info(f"Starting training of fold {fold_id}/{cfg.training.cv_folds}.")
        model = GLiNER2.from_pretrained(cfg.model.name)
        training_config = build_training_config(cfg, output_dir / f"fold_{fold_id}")
        training_config.output_dir = f"{output_dir}/fold_{fold_id}"
        logger_fold = create_logger(f"{training_config.output_dir}/logs/training.log")
        results = train_gliner_model(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            training_config=training_config,
            logger=logger_fold,
        )
        all_results.append(results)
    # Save loss points
    save_loss_points(all_results, output_dir, logger)
    # Plot training curves
    save_plot_training_curves(all_results, output_dir, logger)


@click.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default="src/mdner_llm/gliner/training_config.yaml",
    help="Path to the training config YAML file.",
)
def run_main_from_cli(config_path: str | Path) -> None:
    """Run the main function with config path from CLI."""
    main(config_path)


if __name__ == "__main__":
    run_main_from_cli()
