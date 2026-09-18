"""Training GLINER2 model to fine-tune on Molecular Dynamics-specific NER tasks."""

import json
import operator
import os
import random
from collections import defaultdict
from pathlib import Path

import click
import loguru
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import yaml
from gliner2 import AutoExtractor, GLiNER2
from gliner2.processor import WhitespaceTokenSplitter
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from pydantic import ValidationError

from mdner_llm.core.evaluate_entities_extraction import compute_scores
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
    # Ensure config_path is a Path object.
    config_path = Path(config_path)
    # Ensure config file exists.
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}.")
        return None
    try:
        # Load raw config from YAML.
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)
        if raw_config is None:
            logger.warning("Config file is empty.")
            return None
        # Validate config through Pydantic model.
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


def split_text_by_tokens(
    text: str,
    url: str | None,
    max_tokens: int,
    overlap_tokens: int,
    logger: "loguru.Logger" = loguru.logger,
) -> list[str]:
    """Split long text into overlapping chunks to avoid CUDA OOM errors.

    Returns
    -------
    list[str]
        List of text chunks.
    """
    # Tokenize the text with same tokenizer used in GLiNER models.
    tokenizer = WhitespaceTokenSplitter()
    # List of tuples: (token_str, start_char, end_char).
    token_spans = list(tokenizer(text))
    # If the text is already within the max token limit,
    if len(token_spans) <= max_tokens:
        # return as a single chunk.
        return [text]
    chunks = []
    # Calculate step size between consecutive sliding windows.
    # Text: "Simulation of lipid bilayer using CHARMM36 force field."
    # Chunking step: max_tokens=5, overlap_tokens=2 -> step = 5 - 2 = 3.
    step = max_tokens - overlap_tokens
    for start_index in range(0, len(token_spans), step):
        # Window 1 (start_index=0, takes tokens 0 to 5):
        # Window 2 (start_index=3, takes tokens 3 to 8):
        window_spans = token_spans[start_index : start_index + max_tokens]
        # First token gives start boundary and last token gives end boundary.
        _first_token, start_char, _end_char = window_spans[0]
        _last_token, _start_char, end_char = window_spans[-1]
        # Extract substring text[start_char:end_char] with overlap.
        # Window 1: text[0:33]   -> "Simulation of lipid bilayer using".
        # Window 2: text[20:55]  -> "bilayer using CHARMM36 force field.".
        chunks.append(text[start_char:end_char].strip())
    logger.warning(f"Text (URL: {url}) is too long ({len(token_spans)} tokens).")
    logger.info(f"Split into {len(chunks)} chunks.")
    return chunks


def build_example(
    annotation_path: Path,
    entity_descriptions: dict[str, str] | None,
    max_tokens: int,
    overlap_tokens: int,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[list[InputExample], str]:
    """Build a single InputExample from a JSON annotation file.

    Returns
    -------
    tuple[list[InputExample], str]
        A tuple containing the list of constructed InputExamples (chunks) and
        an optional URL if present in the annotation.
    """
    # Read the annotation JSON file.
    try:
        json_data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning(f"Failed to read {annotation_path.name}: {error}.")
        return [], ""
    # Break raw text down into overlapping chunks if it exceeds max_tokens,
    # to avoid context truncation during training.
    chunks = split_text_by_tokens(
        text=json_data["raw_text"],
        url=json_data.get("url"),
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        logger=logger,
    )
    examples = []
    for chunk in chunks:
        # Extract raw entities.
        raw_entities = json_data.get("entities", [])
        # Collect matched entities only for categories defined in schema.
        chunk_entities = defaultdict(list)
        # Populate matched entities into their respective category lists.
        # Exemple: {"MOL": ["protein kinase A", "PKA"], "SOFTNAME": ["GROMACS"]}.
        for entity in raw_entities:
            category = entity["category"]
            entity_text = entity["text"]
            # Only add entity if it appears in the current chunk,
            # and is not already added.
            if (entity_text in chunk) and (entity_text not in chunk_entities[category]):
                chunk_entities[category].append(entity_text)
        # Filter descriptions to keep only classes present in this specific chunk.
        chunk_descriptions = {
            category: entity_descriptions[category]
            for category in chunk_entities
            if chunk_entities[category]
        }
        # Create an InputExample,
        examples.append(
            InputExample(
                # with the raw text,
                text=chunk,
                # the formatted entities (category -> list of values),
                entities=chunk_entities,
                # and the entity descriptions.
                entity_descriptions=chunk_descriptions,
            )
        )
    return examples, json_data.get("url", "")


def build_train_dataset(
    annotations_dir_path: Path,
    entity_descriptions: dict[str, str] | None,
    max_tokens: int,
    overlap_tokens: int,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[TrainingDataset, list[Path], list[str]]:
    """
    Build a TrainingDataset from annotation JSON files in a specified directory.

    Returns
    -------
    TrainingDataset
        Training dataset containing the formatted InputExample objects.
    list[Path]
        List of annotation paths that were successfully processed.
    urls : list[str]
        List of URLs extracted from the annotation files (if present).
    """
    logger.info(f"Creating dataset from JSON annotations in: {annotations_dir_path}.")
    train_examples = []
    processed_annotation_paths = []
    urls = []
    first_logged = False
    # Iterate over annotation files to build model training examples.
    for annotation_path in annotations_dir_path.glob("*.json"):
        # Create InputExample objects for each annotation file,
        # splitting long texts into overlapping chunks (~2048 tokens/words) to prevent,
        # context truncation and information loss during model training.
        # Example: InputExample(
        #   text="The protein kinase A (PKA) was simulated with GROMACS.",
        #   entities={"MOL": ["protein kinase A", "PKA"], "SOFTNAME": ["GROMACS"]},
        #   entity_descriptions={"MOL": "Molecular compounds, including ..."}
        # )
        examples, url = build_example(
            annotation_path,
            entity_descriptions,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        for example in examples:
            # Add each chunks to the list of training examples
            train_examples.append(example)
            processed_annotation_paths.append(annotation_path)
            urls.append(url)
            # Log the first example for debugging purposes
            if not first_logged:
                first_logged = True
                logger.info("First training example:")
                logger.info(f"URL: {url}")
                logger.info(f"Text: {example.text.replace('\n', ' ')[:70]}...")
                logger.info("Entities:")
                for category, definition in example.entities.items():
                    logger.info(f"  {category}: {definition}")
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
    list[dict[str, int | Path]] | None
        A list of mismatches with index and path if any inconsistency is found,
        otherwise None.
    """
    # Initialize mismatch collector and extract input text sequences.
    mismatches = []
    inputs = [example.text for example in train_data]
    # Verify correspondence between each input string and its source file.
    for index, (expected_text, path) in enumerate(
        zip(inputs, train_paths, strict=False)
    ):
        try:
            # Read source JSON annotation file.
            with open(path, encoding="utf-8") as file:
                payload = json.load(file)
            raw_text = payload.get("raw_text", "")
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(f"Error reading {path} at index {index}: {error}.")
            mismatches.append({"index": index, "path": path})
            continue
        # Verify that the example text exists as a substring of the raw text.
        if expected_text not in raw_text:
            logger.warning(f"Actual text mismatch at index {index} for file {path}.")
            mismatches.append({"index": index, "path": path})
    # Return collected mismatch entries or None when fully aligned.
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


def k_fold_split(
    dataset: TrainingDataset,
    paths: list[Path],
    urls: list[str],
    cfg: GLiNERConfig,
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> list[tuple[TrainingDataset, TrainingDataset]]:
    """
    Split dataset into folds for cross-validation or a single train/val/test split.

    Returns
    -------
    list of (fold_train, fold_val)
    """
    total_folds = cfg.data.cv_folds
    seed = cfg.data.seed
    num_samples = len(dataset)
    # Shuffle dataset indices deterministically.
    # Example: with 5 samples -> sample_indices = [3, 0, 4, 1, 2].
    sample_indices = list(range(num_samples))
    random.seed(seed)
    random.shuffle(sample_indices)

    folds = []
    # Handle single train/val/test split (no cross-validation).
    if total_folds == 1:
        logger.info("Single split mode (cv_folds == 1).")
        # Step 1: Reserve the test set at the tail of the shuffled indices.
        # Example: 5 samples, val_ratio=0.2
        #           -> test_size = 1
        #           -> test_indices = [2]
        test_size = int(num_samples * cfg.data.val_ratio)
        trainval_indices = (
            sample_indices[:-test_size] if test_size > 0 else sample_indices
        )
        test_indices = sample_indices[-test_size:] if test_size > 0 else []
        # Step 2: Split remaining data into train and validation by train_ratio.
        # Example: 4 remaining samples, train_ratio=0.8
        #          -> train_cutoff = 3
        #          -> train: 3, val: 1
        #          -> train_indices: [3, 0, 4], val_indices: [1].
        train_end_idx = int(len(trainval_indices) * cfg.data.train_ratio)
        split_indices_list = [
            (
                1,
                {
                    "train": trainval_indices[:train_end_idx],
                    "val": trainval_indices[train_end_idx:],
                    "test": test_indices,
                },
            )
        ]
    else:
        # Standard K-fold nested cross-validation.
        logger.info(f"Splitting dataset into {total_folds} folds for nested CV.")
        # Step 1: Divide indices into K buckets and append remainder to the last bucket.
        # Example: 5 samples across 2 folds
        #          -> Fold size = 5 // 2 = 2
        #          -> Bucket 1 = [3, 0]
        #          -> Bucket 2 = [4, 1, 2] (last bucket gets the remainder).
        fold_size = num_samples // total_folds
        fold_buckets = []
        for fold_index in range(total_folds):
            start_index = fold_index * fold_size
            end_index = (fold_index + 1) * fold_size
            # Iteration 0 (fold_index=0): indices[0:2] -> Bucket 1: [3, 0].
            # Iteration 1 (fold_index=1): indices[2:4] -> Bucket 2: [4, 1].
            fold_buckets.append(sample_indices[start_index:end_index])
        # Add remaining examples to the last bucket.
        # Example: 5 samples, 2 folds
        #          ->  Remainder = 5 % 2 = 1 sample
        #          ->  Sample_indices[2 * 2 :] = sample_indices[4:] -> [2]
        #          ->  Result: Bucket 2 becomes [4, 1, 2]
        fold_buckets[-1].extend(sample_indices[total_folds * fold_size :])
        # Step 2: Use 1 bucket as test and combine remaining K-1 buckets for train/val.
        # Example Fold 1: test = [3, 0], trainval pool = [4, 1, 2].
        # Example Fold 2: test = [4, 1, 2], trainval pool = [3, 0].
        split_indices_list = []
        for fold_id, test_indices in enumerate(fold_buckets, start=1):
            trainval_indices = []
            for bucket in fold_buckets:
                if bucket is not test_indices:
                    trainval_indices.extend(bucket)
            # Shuffle the combined pool to eliminate original bucket ordering,
            # which could bias the train/val split.
            random.shuffle(trainval_indices)
            # Partition trainval pool according to train_ratio.
            # Example (3 pool samples, train_ratio=0.66)
            #          -> train_cutoff = 2
            #          -> train: 2, val: 1
            #          -> train_indices: [4, 1], val_indices: [2]
            train_end_idx = int(len(trainval_indices) * cfg.data.train_ratio)
            split_indices_list.append(
                (
                    fold_id,
                    {
                        "train": trainval_indices[:train_end_idx],
                        "val": trainval_indices[train_end_idx:],
                        "test": test_indices,
                    },
                )
            )
    # Process and save splits for each fold.
    for fold_id, split_indices in split_indices_list:
        fold_dir = output_dir / f"fold_{fold_id}" / "data"
        fold_dir.mkdir(parents=True, exist_ok=True)
        # Create TrainingDataset objects for each split.
        splits = {
            split_name: TrainingDataset([dataset[sample_idx] for sample_idx in idxs])
            for split_name, idxs in split_indices.items()
        }
        logger.info(
            f"Fold {fold_id}/{max(total_folds, 1)} — "
            f"train={len(splits['train'])}, "
            f"val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )
        # Save each split to JSONL and check alignment with annotation files.
        for split_name, split_dataset in splits.items():
            save_path = fold_dir / f"{split_name}.jsonl"
            current_split_indices = split_indices[split_name]
            # Save dataset split as JSONL.
            save_dataset_to_jsonl(split_dataset, save_path, logger)
            # Retrieve corresponding paths and URLs.
            split_paths = [paths[sample_idx] for sample_idx in current_split_indices]
            split_urls = [urls[sample_idx] for sample_idx in current_split_indices]
            # Check alignment between dataset and annotations.
            alignment_issues = check_alignment(split_dataset, split_paths, logger)
            if alignment_issues:
                logger.warning(
                    f"Alignment check failed for {split_name} split of fold {fold_id}. "
                    f"Metadata file will not be saved for this split."
                )
                continue
            # Save metadata when alignment succeeds.
            out = save_path.with_name(f"{save_path.stem}_metadata.txt")
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.writelines(
                    f"{path}\t{url}\n"
                    for path, url in zip(split_paths, split_urls, strict=False)
                )

        folds.append((splits["train"], splits["val"]))

    logger.success(
        f"Completed splitting and saving datasets to {output_dir} successfully!"
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
        # DataLoader
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        prefetch_factor=config.training.prefetch_factor,
        # Training schedule & batch sizes
        num_epochs=config.training.num_epochs,
        max_steps=config.training.max_steps,
        batch_size=config.training.batch_size,
        eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        # Learning rates & scheduler
        encoder_lr=config.training.encoder_lr,
        task_lr=config.training.task_lr,
        warmup_ratio=config.training.warmup_ratio,
        warmup_steps=config.training.warmup_steps,
        scheduler_type=config.training.scheduler_type,
        # Optimization
        weight_decay=config.training.weight_decay,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        # Precision / hardware
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        # LoRA
        use_lora=config.training.use_lora,
        lora_r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        lora_target_modules=config.training.lora_target_modules,
        save_adapter_only=config.training.save_adapter_only,
        # Checkpointing & evaluation
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_total_limit=config.training.save_total_limit,
        save_best=config.training.save_best,
        metric_for_best=config.training.metric_for_best,
        greater_is_better=config.training.greater_is_better,
        # Logging & tracking
        logging_steps=config.training.logging_steps,
        logging_first_step=config.training.logging_first_step,
        # Early stopping
        early_stopping=config.training.early_stopping,
        early_stopping_patience=config.training.early_stopping_patience,
        # Execution & validation
        seed=config.training.seed,
        deterministic=config.training.deterministic,
        gradient_checkpointing=config.training.gradient_checkpointing,
        max_train_samples=config.training.max_train_samples,
        max_eval_samples=config.training.max_eval_samples,
        validate_data=config.training.validate_data,
        # Use allow_invalid_samples to skip samples,
        # without all categories present in the schema.
        # Necessary for gliner2.5 training.
        allow_invalid_samples=True,
    )


def compute_evaluation_metrics(
    model: GLiNER2,
    eval_dataset: TrainingDataset,
) -> dict[str, float]:
    """Evaluate GLiNER model predictions and compute global confusion-based scores.

    Returns
    -------
    dict[str, float]
        Mapping of evaluation metric names to their float values.
    """
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_clean_false_positives = 0
    for source_text, metadata in eval_dataset:
        raw_entities_dict = metadata.get("entities", {})
        entity_descriptions = metadata.get("entity_descriptions", {})
        # Collect unique ground truth mentions across categories.
        # Format input: {"FFM": ["CHARMM36", "AMBER"], "SOFTNAME": ["GROMACS"]}
        # Format output: {"CHARMM36", "AMBER", "GROMACS"}
        groundtruth_entities = set()
        for entities_by_category in raw_entities_dict.values():
            groundtruth_entities.update(entities_by_category)
        # Predict candidate entities for the input text.
        predictions = model.extract_entities(source_text, entity_descriptions)
        # Extract text mentions from predictions dictionary.
        # Format input:{"entities": {"FFM": [{"text": "CHARMM36", "confidence": 0.99}]}}
        # Format output: {"CHARMM36"}
        predicted_entities = set()
        entities_by_category_with_scores = predictions.get("entities", {})
        for entities_by_category in entities_by_category_with_scores.values():
            predicted_entities.update(entities_by_category)
        # Compute intersection and discrepancies between ground truth and predictions.
        true_positives = groundtruth_entities & predicted_entities
        false_positives = predicted_entities - groundtruth_entities
        false_negatives = groundtruth_entities - predicted_entities
        # Filter false positives to exclude text hallucinations not grounded in source.
        clean_false_positives = {
            entity for entity in false_positives if entity in source_text
        }
        # Accumulate sample-level confusion counts.
        total_true_positives += len(true_positives)
        total_false_positives += len(false_positives)
        total_false_negatives += len(false_negatives)
        total_clean_false_positives += len(clean_false_positives)

    # Compute global micro-averaged metrics from accumulated counts.
    computed_scores = compute_scores(
        tp=pd.Series([total_true_positives]),
        fp=pd.Series([total_false_positives]),
        fn=pd.Series([total_false_negatives]),
        fp_clean=pd.Series([total_clean_false_positives]),
    )
    # Extract scalar float metric values.
    return {
        f"eval_{metric_name}": float(metric_series.iloc[0])
        for metric_name, metric_series in computed_scores.items()
    }


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
    trainer = ExtractorTrainer(
        model, training_config, compute_metrics=compute_evaluation_metrics
    )
    results = trainer.train(train_data=train_dataset, eval_data=eval_dataset)
    logger.success("✓ Training complete successfully!")
    # Identify best checkpoints for loss and F1 score across evaluation epochs.
    history = results.get("eval_metrics_history", [])
    best_loss = min(history, key=operator.itemgetter("eval_loss"), default={})
    best_f1 = max(history, key=operator.itemgetter("eval_f1"), default={})
    results.update(
        best_eval_loss=best_loss.get("eval_loss"),
        best_eval_loss_epoch=best_loss.get("epoch"),
        best_eval_f1=best_f1.get("eval_f1"),
        best_eval_f1_epoch=best_f1.get("epoch"),
    )
    logger.info(f"Duration: {results.get('total_time_seconds'):.2f} seconds")
    logger.info(f"Total steps: {results.get('total_steps')}")
    logger.info(f"Total epoch: {results.get('total_epochs')}")
    logger.info(
        f"Lowest loss: {int(results.get('best_eval_loss'))} "
        f"at epoch {results.get('best_eval_loss_epoch')}"
    )
    logger.info(
        f"Lowest eval F1: {results.get('best_eval_f1'):.2f} "
        f"at epoch {results.get('best_eval_f1_epoch')}"
    )
    logger.success(
        f"Best model saved to {training_config.output_dir}/best successfully!"
    )
    # Explicitly clear GPU memory and delete trainer to avoid memory leaks.
    # Without this, VRAM was not fully released between folds,
    # causing out-of-memory errors on subsequent iterations.
    if hasattr(trainer, "optimizer"):
        trainer.optimizer.state.clear()
        del trainer.optimizer
    del trainer
    return results


def train_single_fold_process(
    fold_id: int,
    cfg: GLiNERConfig,
    train_data: TrainingDataset,
    val_data: TrainingDataset,
    output_dir: Path,
    result_queue: mp.Queue,
) -> None:
    """Execute training of an individual fold inside an isolated child process.

    Spawning an independent OS process for each fold guarantees that all PyTorch
    C++/CUDA runtime state, autograd computational graphs, and allocator caches
    are fully discarded upon process termination, completely preventing GPU VRAM
    accumulation across cross-validation folds.
    """
    # Must be set before importing torch or initializing CUDA driver bindings.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Deferred imports ensure child process starts with an unpolluted CUDA context.
    # Load the model.
    model = AutoExtractor.from_pretrained(cfg.model.name)
    # Build a fold-specific training configuration and logger.
    training_config = build_training_config(cfg, output_dir / f"fold_{fold_id}")
    training_config.output_dir = f"{output_dir}/fold_{fold_id}"
    logger_fold = create_logger(f"{training_config.output_dir}/logs/training.log")
    # Train the model for this fold.
    results = train_gliner_model(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        training_config=training_config,
        logger=logger_fold,
    )
    result_queue.put(results)


def save_training_history(
    results_list: list[dict],
    output_dir: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Save raw training and evaluation results across all folds to JSON."""
    history_records = [
        {"fold": fold_id, **results}
        for fold_id, results in enumerate(results_list, start=1)
    ]
    output_path = output_dir / "training_history.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history_records, file, indent=2)
    logger.success(f"Saved full training history to {output_path} successfully!")


def generate_gradient_colors(
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
    red, green, blue, _ = to_rgba(base_color)
    alphas = np.linspace(min_alpha, max_alpha, max(n_colors, 1))
    return [(red, green, blue, a) for a in alphas]


def plot_loss_evolution(
    axis: plt.Axes, results_list: list[dict], cfg: "TrainingConfig"
) -> None:
    """Plot training and validation loss curves across folds."""
    # Generate gradient colors for training and evaluation curves.
    number_of_folds = len(results_list)
    train_colors = generate_gradient_colors("#0055FF", number_of_folds)
    eval_colors = generate_gradient_colors("#FFAA00", number_of_folds)
    # Configure axes layout and appearance.
    axis.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.4, color="#94A3B8")
    for spine_name in ("top", "right"):
        axis.spines[spine_name].set_visible(False)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_ylim(bottom=0, top=1000)
    axis.set_xlim(left=0, right=cfg.training.num_epochs)
    global_min_loss = {"loss": float("inf"), "epoch": None, "fold": None, "color": None}
    # Plot loss trajectory per fold and track best evaluation point.
    for fold_index, results in enumerate(results_list):
        fold_train_color = train_colors[fold_index]
        fold_eval_color = eval_colors[fold_index]
        for key, color, is_eval in (
            ("train_metrics_history", fold_train_color, False),
            ("eval_metrics_history", fold_eval_color, True),
        ):
            metric_key = "eval_loss" if is_eval else "loss"
            loss_by_epoch = {}
            for entry in results.get(key, []):
                epoch = int(entry["epoch"])
                loss_value = float(entry[metric_key])
                loss_by_epoch[epoch] = min(
                    loss_by_epoch.get(epoch, float("inf")), loss_value
                )
            epochs = sorted(loss_by_epoch)
            losses = [loss_by_epoch[epoch] for epoch in epochs]
            axis.plot(
                epochs, losses, color=color, linewidth=1.6, marker="o", markersize=3
            )
            if is_eval:
                for epoch, loss_val in zip(epochs, losses, strict=False):
                    if loss_val < global_min_loss["loss"]:
                        global_min_loss = {
                            "loss": loss_val,
                            "epoch": epoch,
                            "fold": fold_index + 1,
                            "color": color,
                        }
    # Annotate minimum evaluation loss point.
    if global_min_loss["epoch"] is not None:
        axis.scatter(
            global_min_loss["epoch"],
            global_min_loss["loss"],
            s=80,
            color=global_min_loss["color"],
            edgecolors="#1E293B",
            linewidths=1.2,
            zorder=10,
        )
        axis.annotate(
            f"Best eval loss\nFold {global_min_loss['fold']} "
            f"(Ep. {global_min_loss['epoch']})\nLoss: {global_min_loss['loss']:.0f}",
            xy=(global_min_loss["epoch"], global_min_loss["loss"]),
            xytext=(15, -18),
            textcoords="offset points",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": global_min_loss["color"],
                "alpha": 0.95,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": global_min_loss["color"],
                "linewidth": 1.2,
            },
        )
    # Render training and validation fold legends.
    for title, anchor_x, color_theme, colors in (
        ("Train", 1.00, "#0055FF", train_colors),
        ("Validation", 0.87, "#FFAA00", eval_colors),
    ):
        handles = [
            Line2D([0], [0], color=colors[fold_idx], lw=2, label=f"Fold {fold_idx + 1}")
            for fold_idx in range(number_of_folds)
        ]
        legend = axis.legend(
            handles=handles,
            title=title,
            loc="upper right",
            bbox_to_anchor=(anchor_x, 1.00),
            frameon=True,
            facecolor="white",
            edgecolor="#E2E8F0",
            fontsize=7.5,
            title_fontsize=8,
        )
        legend.get_title().set_color(color_theme)
        legend.get_title().set_weight("bold")
        axis.add_artist(legend)
    # Set axis titles and labels.
    axis.set_xlabel("Epoch", fontsize=10.5)
    axis.set_ylabel("Loss", fontsize=10.5)
    axis.set_title("Loss Evolution", fontsize=11.5, pad=8, fontweight="medium")


def collect_validation_metric_records(
    results_list: list[dict],
    target_metric_keys: tuple[str, ...],
) -> tuple[dict[str, dict[int, list[float]]], dict[str, float | int | None]]:
    """Extract validation scores per epoch and identify global peak F1 performance.

    Returns
    -------
    tuple[dict[str, dict[int, list[float]]], dict[str, float | int | None]]
        Tuple containing scores grouped by metric/epoch and global best F1 record.

    Examples
    --------
    >>> # Input fold results from cross-validation:
    >>> # results = [
    >>> #     {"eval_metrics_history":
    >>> #      [{"epoch": 1, "eval_f1": 0.82, "eval_precision": 0.85}]},
    >>> #     {"eval_metrics_history":
    >>> #      [{"epoch": 1, "eval_f1": 0.88, "eval_precision": 0.90}]},
    >>> # ]
    >>> # Output scores_by_metric:
    >>> # {"eval_f1": {1: [0.82, 0.88]}, "eval_precision": {1: [0.85, 0.90]}}
    >>> # Output peak_f1_record:
    >>> # {"score": 0.88, "epoch": 1, "fold": 2}
    """
    scores_by_metric = {
        metric_key: defaultdict(list) for metric_key in target_metric_keys
    }
    peak_f1_record = {"score": float("-inf"), "epoch": None, "fold": None}
    # Traverse each fold evaluation history.
    # Example fold entry: {"epoch": 2, "eval_f1": 0.85, "eval_precision": 0.80}.
    for fold_index, results in enumerate(results_list):
        current_fold_number = fold_index + 1
        for entry in results.get("eval_metrics_history", []):
            epoch_number = int(entry["epoch"])
            for metric_key in target_metric_keys:
                raw_metric_value = entry.get(metric_key)
                if raw_metric_value is None:
                    continue

                metric_score = float(raw_metric_value)
                scores_by_metric[metric_key][epoch_number].append(metric_score)

                # Keep track of global maximum validation F1 score across all folds.
                if metric_key == "eval_f1" and metric_score > peak_f1_record["score"]:
                    peak_f1_record["score"] = metric_score
                    peak_f1_record["epoch"] = epoch_number
                    peak_f1_record["fold"] = current_fold_number

    return scores_by_metric, peak_f1_record


def plot_validation_metrics_evolution(
    axis: plt.Axes,
    results_list: list[dict],
    cfg: "TrainingConfig",
) -> None:
    """Plot aggregated mean validation metrics across folds."""
    metric_configs = {
        "eval_precision": {
            "label": "Precision",
            "color": "#2563EB",
            "style": "--",
            "marker": "^",
            "width": 1.8,
        },
        "eval_f1": {
            "label": "F1",
            "color": "#7C3AED",
            "style": "-",
            "marker": "o",
            "width": 2.2,
        },
        "eval_recall": {
            "label": "Recall",
            "color": "#DC2626",
            "style": ":",
            "marker": "s",
            "width": 1.8,
        },
    }
    axis.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.4, color="#94A3B8")
    for spine_name in ("top", "right"):
        axis.spines[spine_name].set_visible(False)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_ylim(bottom=0, top=1.0)
    axis.set_xlim(left=0, right=cfg.training.num_epochs)
    scores_by_metric, peak_f1 = collect_validation_metric_records(
        results_list, tuple(metric_configs.keys())
    )
    f1_color = metric_configs["eval_f1"]["color"]
    for epoch_number, f1_scores in scores_by_metric["eval_f1"].items():
        for single_f1_score in f1_scores:
            axis.plot(
                epoch_number,
                single_f1_score,
                marker="o",
                markersize=2.5,
                color=f1_color,
                alpha=0.25,
            )
    legend_handles = []
    for metric_key, config in metric_configs.items():
        epoch_records = scores_by_metric[metric_key]
        if not epoch_records:
            continue
        sorted_epochs = sorted(epoch_records)
        epoch_values = [epoch_records[epoch] for epoch in sorted_epochs]
        metric_means = np.array([np.mean(values) for values in epoch_values])
        axis.plot(
            sorted_epochs,
            metric_means,
            color=config["color"],
            linestyle=config["style"],
            marker=config["marker"],
            linewidth=config["width"],
            markersize=4.5,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=config["color"],
                lw=config["width"],
                linestyle=config["style"],
                marker=config["marker"],
                markersize=4,
                label=config["label"],
            )
        )
    if peak_f1["epoch"] is not None:
        axis.scatter(
            peak_f1["epoch"],
            peak_f1["score"],
            s=85,
            color=f1_color,
            edgecolors="#1E293B",
            linewidths=1.2,
            zorder=10,
        )
        axis.annotate(
            f"Best eval F1\nFold {peak_f1['fold']} "
            f"(Ep. {peak_f1['epoch']})\nF1: {peak_f1['score']:.2f}",
            xy=(peak_f1["epoch"], peak_f1["score"]),
            xytext=(15, -18),
            textcoords="offset points",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": f1_color,
                "alpha": 0.95,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": f1_color,
                "linewidth": 1.2,
            },
        )
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#E2E8F0",
        fontsize=8.5,
    )
    axis.set_xlabel("Epoch", fontsize=10.5)
    axis.set_ylabel("Validation metric", fontsize=10.5)
    axis.set_title(
        "Cross-Validation Metrics (Mean)",
        fontsize=11.5,
        pad=8,
        fontweight="medium",
    )


def save_plot_training_curves(
    results_list: list[dict],
    model_name: str,
    output_dir: Path,
    cfg: GLiNERConfig,
    file_name: str = "training_and_metrics_curves.png",
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Plot and save consolidated training loss and validation metrics curves."""
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=300)
    # Plot evloution of the loss over epochs.
    plot_loss_evolution(axes[0], results_list, cfg=cfg)
    # Plot evolution of validation metrics (F1, Precision, Recall) over epochs.
    plot_validation_metrics_evolution(axes[1], results_list, cfg=cfg)
    # Styling and layout adjustments
    # Compute total training duration across all folds.
    total_seconds = sum(
        results.get("total_time_seconds", 0) for results in results_list
    )
    minutes, seconds = divmod(int(total_seconds), 60)
    duration_string = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    # Add the title summarizing the model and training duration.
    figure.suptitle(
        f"Model Training Overview: '{model_name}' | Runtime: {duration_string}",
        fontsize=12.5,
        fontweight="semibold",
        color="#0F172A",
        y=1.02,
    )
    plt.tight_layout()
    # Save the figure to the specified output directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / file_name
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(
        f"Saved consolidated training & metrics plot to {plot_path} successfully!"
    )


def train_all_folds(
    cfg: GLiNERConfig, logger: "loguru.Logger" = loguru.logger
) -> list[dict]:
    """Execute complete nested K-fold training pipeline from a GLiNERConfig object.

    Returns
    -------
    list[dict]
        List of training results for each fold, containing metrics and loss history.
    """
    # Setup output directory.
    output_dir = Path(cfg.model.output_dir) / f"{cfg.model.experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create dataset from annotation files
    dataset, selected_annotation_paths, urls = build_train_dataset(
        annotations_dir_path=cfg.data.annotations_path,
        entity_descriptions=cfg.categories,
        max_tokens=cfg.data.max_tokens,
        overlap_tokens=cfg.data.overlap_tokens,
        logger=logger,
    )
    # Validate dataset.
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
    # Train a separate model for each fold and collect results.
    # Using "spawn" context ensures that each fold runs in a fresh process,
    # preventing GPU memory accumulation and OOM errors across folds.
    # Doc: https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
    ctx = mp.get_context("spawn")
    all_results = []
    for fold_id, (train_data, val_data) in enumerate(folds, start=1):
        logger.info(f"Starting training of fold {fold_id}/{cfg.data.cv_folds}.")
        # Spawn a new process for the current fold.
        result_queue = ctx.Queue()
        # Execute training.
        process = ctx.Process(
            target=train_single_fold_process,
            args=(fold_id, cfg, train_data, val_data, output_dir, result_queue),
        )
        process.start()
        # Wait for the process to finish and retrieve results.
        results = result_queue.get()
        process.join()
        # Check if the process exited successfully.
        if process.exitcode != 0:
            logger.error(f"Fold {fold_id} failed with exitcode {process.exitcode}.")
        # Append results to the overall list for plotting.
        all_results.append(results)

    # Save training history in JSON format.
    save_training_history(all_results, output_dir, logger=logger)
    # Save training (Loss, F1, and Precision) curves across all folds.
    save_plot_training_curves(
        all_results, cfg.model.name, output_dir, logger=logger, cfg=cfg
    )
    return all_results


def main(config_path: str | Path) -> None:
    """Train GLINER2 model using the specified training configuration file."""
    # Initialize logger.
    logger = create_logger(level="DEBUG")
    logger.info("Starting GLiNER2 finetuning process.")
    # Load config.
    cfg = load_config(config_path, logger=logger)
    if not cfg:
        logger.error("Failed to load training configuration.")
        logger.error("Exiting training process.")
        return
    # Execute the training pipeline.
    train_all_folds(cfg, logger=logger)


@click.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the training config YAML file.",
)
def run_main_from_cli(config_path: str | Path) -> None:
    """Run the main function with config path from CLI."""
    main(config_path)


if __name__ == "__main__":
    run_main_from_cli()
