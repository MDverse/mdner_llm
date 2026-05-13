"""Training GLINER2 model to fine-tune on MolecularDynamics-specific NER tasks."""

import json
import random
from collections import defaultdict
from pathlib import Path

import click
import loguru
import yaml
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from matplotlib import pyplot as plt
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
        for field_name, field_value in validated_config.model_dump().items():
            logger.debug(f"Config - {field_name}: {field_value}")

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
    url = json_data.get("url", "")
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

    return example, url


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


def split_randomly(
    elements: list,
    config_data: GLiNERConfig,
) -> tuple[list, list]:
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
        The list split into (train, test)
        according to the specified ratios and shuffled using the
        provided seed for reproducibility.
    """
    indices = list(range(len(elements)))

    if config_data.shuffle:
        random.seed(config_data.seed)
        random.shuffle(indices)

    n = len(indices)
    train_end = int(n * config_data.train_ratio)

    train_elements = [elements[i] for i in indices[:train_end]]
    test_elements = [elements[i] for i in indices[train_end:]]

    return train_elements, test_elements


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

    Returns
    -------
    tuple
        (train_data, val_data, test_data) - the split datasets.
    """
    logger.info(
        f"Splitting dataset of {len(dataset)} examples into train/val/test sets"
    )
    logger.info(
        f"Ratios - Train: {config_data.train_ratio}, Test: {config_data.test_ratio}"
    )
    # Split the dataset into train/validation/test sets
    # using the specified ratios and seed
    train_data, _val_data, test_data = dataset.split(
        train_ratio=config_data.train_ratio,
        val_ratio=0.0,  # No separate validation set, only train and test
        test_ratio=config_data.test_ratio,
        shuffle=config_data.shuffle,
        seed=config_data.seed,
    )
    # Split the list of annotation paths in the same way
    # to keep track of which examples belong to which set
    train_paths, test_paths = split_randomly(selected_annotation_paths, config_data)
    # Split the list of URLs in the same way (if present)
    train_urls, test_urls = split_randomly(urls, config_data)

    with suppress_stdout():
        train_data.save(config_data.train_data_path)
        issues = check_alignment(train_data, train_paths, logger)
        if not issues:
            save_paths_txt(train_paths, train_urls, config_data.train_data_path, logger)
        logger.success(
            f"Saved {len(train_data)} training examples to "
            f"{config_data.train_data_path}"
        )
        test_data.save(config_data.test_data_path)
        issues = check_alignment(test_data, test_paths, logger)
        if not issues:
            save_paths_txt(test_paths, test_urls, config_data.test_data_path, logger)
        logger.success(
            f"Saved {len(test_data)} test examples to {config_data.test_data_path}"
        )
    return train_data, test_data


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
    else:
        logger.debug(f"Dataset saved to {path} successfully!")


def save_metadata(
    paths: list[Path],
    urls: list[str],
    target: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Save paths and urls to a .txt file aligned with the dataset JSONL file."""
    out = target.with_name(f"{target.stem}_metadata.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(f"{path}\t{url}\n" for path, url in zip(paths, urls, strict=False))
    logger.success(f"Metadata saved → {out} ({len(paths)} entries) successfully!")


def k_fold_split(
    dataset: TrainingDataset,
    paths: list[Path],
    urls: list[str],
    cfg: GLiNERConfig,
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> list[tuple[TrainingDataset, TrainingDataset, TrainingDataset]]:
    """
    Nested cross-validation split on the full dataset.

    For each fold i (outer loop):
      - fold i          → test
      - remaining folds → split into train/val using cfg.train_ratio / cfg.val_ratio

    Each fold's splits are saved as JSONL to output_dir/fold_{i}/data/.

    Returns
    -------
    list of (fold_train, fold_val, fold_test)
    """
    k = cfg.training.cv_folds
    seed = cfg.data.seed
    n = len(dataset)

    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)

    fold_size = n // k
    # Build the K index buckets
    buckets = [idx[i * fold_size : (i + 1) * fold_size] for i in range(k)]
    # Remaining examples (if n % k != 0) go into the last bucket
    if n % k:
        buckets[-1].extend(idx[k * fold_size :])

    folds = []
    for i in range(k):
        fold_dir = output_dir / f"fold_{i + 1}" / "data"
        fold_dir.mkdir(parents=True, exist_ok=True)
        # Outer split: bucket i → test, rest → train+val pool
        test_idx = buckets[i]
        trainval_idx = [j for b, bucket in enumerate(buckets) if b != i for j in bucket]
        # Inner split: train+val pool → train / val
        random.seed(seed)
        random.shuffle(trainval_idx)
        tv_n = len(trainval_idx)
        # Normalise ratios so they sum to 1 over the train+val pool
        ratio_sum = cfg.data.train_ratio + cfg.data.val_ratio
        train_end = int(tv_n * cfg.data.train_ratio / ratio_sum)

        train_idx = trainval_idx[:train_end]
        val_idx = trainval_idx[train_end:]

        fold_train = TrainingDataset([dataset[j] for j in train_idx])
        fold_val = TrainingDataset([dataset[j] for j in val_idx])
        fold_test = TrainingDataset([dataset[j] for j in test_idx])

        logger.info(
            f"Fold {i + 1}/{k} — "
            f"train={len(fold_train)}, val={len(fold_val)}, test={len(fold_test)}"
        )

        # Save each split as JSONL + metadata
        for split_name, split_ds, split_idx in [
            ("train", fold_train, train_idx),
            ("val", fold_val, val_idx),
            ("test", fold_test, test_idx),
        ]:
            save_path = fold_dir / f"{split_name}.jsonl"
            save_dataset_to_jsonl(split_ds, save_path, logger)

            split_paths = [paths[j] for j in split_idx]
            split_urls = [urls[j] for j in split_idx]
            if check_alignment(split_ds, split_paths, logger):
                save_metadata(split_paths, split_urls, save_path, logger)

            logger.success(f"  {split_name}: {len(split_ds)} examples → {save_path}")

        folds.append((fold_train, fold_val, fold_test))

    return folds


def load_model(model_name: str, logger: "loguru.Logger" = loguru.logger) -> GLiNER2:
    """Load a fresh GLiNER2 model (must be called once per fold to reset LoRA state)."""
    with capture() as buf:
        model = GLiNER2.from_pretrained(model_name)
    for line in buf.getvalue().strip().splitlines():
        logger.info(line.strip())
    return model


def build_training_config(
    cfg: GLiNERConfig,
    fold_output_dir: Path,
) -> TrainingConfig:
    """Build a TrainingConfig for GLiNER2Trainer.

    Returns
    -------
    TrainingConfig
        Configured TrainingConfig object with parameters
        from GLiNERConfig and fold output directory.
    """
    return TrainingConfig(
        output_dir=str(fold_output_dir),
        experiment_name=cfg.model.experiment_name,
        num_epochs=cfg.training.num_epochs,
        max_steps=cfg.training.max_steps,
        batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        encoder_lr=cfg.training.encoder_lr,
        task_lr=cfg.training.task_lr,
        warmup_ratio=cfg.training.warmup_ratio,
        scheduler_type=cfg.training.scheduler_type,
        weight_decay=cfg.training.weight_decay,
        fp16=cfg.training.fp16,
        eval_strategy=cfg.training.eval_strategy,
        metric_for_best=cfg.training.metric_for_best,
        save_best=cfg.training.save_best,
        logging_steps=cfg.training.logging_steps,
    )


def load_model_and_config(
    model_name: str,
    config: GLiNERConfig,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[GLiNER2, TrainingConfig]:
    """
    Load a GLiNER2 model and the trainfing configuation for training.

    Returns
    -------
    tuple
        (model, training_config)
            The loaded GLiNER2 model and the training configuration.
    """
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
        # LoRa
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


def plot_mean_with_annotation(data_by_epoch, color, facecolor, y_offset, ax=None):
    """Plot mean loss curve with annotation for minimum point."""
    epochs = sorted(data_by_epoch)
    losses = [sum(data_by_epoch[ep]) / len(data_by_epoch[ep]) for ep in epochs]
    label = "Train (mean)" if color == "#0000FF" else "Validation (mean)"
    ax.plot(
        epochs,
        losses,
        color=color,
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=label,
    )
    min_loss, min_ep = min(losses), epochs[losses.index(min(losses))]
    ax.axvline(min_ep, color=color, linestyle=":", linewidth=1.5)
    ax.annotate(
        f"min={min_loss:.3f}\nepoch={min_ep}",
        xy=(min_ep, min_loss),
        xytext=(min_ep + 0.3, min_loss + y_offset),
        fontsize=9,
        fontweight="bold",
        color=color,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": facecolor, "alpha": 0.8},
        arrowprops={"arrowstyle": "->", "color": color},
    )


def save_plot_training_curves(
    results_list: list[dict],
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Save a plot of training and validation loss curves across K-folds."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_folds = len(results_list)
    all_train, all_eval = {}, {}

    for results in results_list:
        train_history = results.get("train_metrics_history", [])
        eval_history = results.get("eval_metrics_history", [])
        train_epochs = [int(x["epoch"]) for x in train_history]
        train_losses = [x["loss"] for x in train_history]
        eval_epochs = [int(x["epoch"]) for x in eval_history]
        eval_losses = [x["eval_loss"] for x in eval_history]
        ax.plot(
            train_epochs,
            train_losses,
            color="#0000FF",
            alpha=0.3,
            linewidth=1.5,
            marker="o",
            markersize=3,
        )
        ax.plot(
            eval_epochs,
            eval_losses,
            color="#FFA500",
            alpha=0.3,
            linewidth=1.5,
            marker="o",
            markersize=3,
        )
        for ep, loss in zip(train_epochs, train_losses, strict=False):
            all_train.setdefault(ep, []).append(loss)
        for ep, loss in zip(eval_epochs, eval_losses, strict=False):
            all_eval.setdefault(ep, []).append(loss)

    plot_mean_with_annotation(all_train, "#0000FF", "#DDDDFF", 10, ax=ax)
    plot_mean_with_annotation(all_eval, "#FFA500", "#FFE8CC", 20, ax=ax)

    last = results_list[-1]
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_xlim(0, 15)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.legend(loc="upper right")
    fig.suptitle(
        f"Training and Validation Loss (kfolds={n_folds}, "
        f"epochs={last.get('total_epochs', '?')}, "
        f"duration={last.get('total_time_seconds', 0):.0f}s)",
        fontsize=12,
    )
    plt.tight_layout()
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.success(f"Saved CV training curves plot to {output_path} successfully!")


def main(config_path: str | Path):
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
    # Create dataset
    # Build training examples from annotation files
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
    # Train model using K-fold cross-validation
    all_results = []
    for fold_id, (train_data, val_data) in enumerate(folds, start=1):
        logger.info(f"Starting fold {fold_id}/{cfg.training.cv_folds}")
        model = load_model(cfg.model.name, logger)
        training_config = build_training_config(cfg, output_dir / f"fold_{fold_id}")
        training_config.output_dir = f"{output_dir}/fold_{fold_id}"
        logger_fold = create_logger(training_config.output_dir / "training.log")
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
    logger.success(f"✓ Training complete! Models saved to {output_dir} successfully!")


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
