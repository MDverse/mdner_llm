"""Training GLINER2 model to fine-tune on MolecularDynamics-specific NER tasks."""

from datetime import datetime
from pathlib import Path

import click

from mdner_llm.core.logger import create_logger
from gliner2 import GLiNER2
from gliner2.training.data import TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

from mdner_llm.core.logger import create_logger
from mdner_llm.utils.evaluate_gliner2 import build_train_examples
from mdner_llm.utils.select_annotation_files import select_annotation_files

def main(config_path: str | Path):
    """Main training function."""
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = create_logger(f"logs/train_gliner_{timestamp}.log")
    # Load config
    cfg = load_config_as_namespace(config_path)

    # Convert to dicts for model building
    model_cfg = namespace_to_dict(cfg.model)
    train_cfg = namespace_to_dict(cfg.training)

    # Setup output directory
    output_dir = Path(cfg.data.root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    logger.info(f"Loading training data from: {cfg.data.train_data}")
    train_dataset = load_json_data(cfg.data.train_data)
    logger.info(f"Training samples: {len(train_dataset)}")

    eval_dataset = None
    if hasattr(cfg.data, "val_data_dir") and cfg.data.val_data_dir.lower() not in (
        "none",
        "null",
        "",
    ):
        logger.info(f"Loading validation data from: {cfg.data.val_data_dir}")
        eval_dataset = load_json_data(cfg.data.val_data_dir)
        logger.info(f"Validation samples: {len(eval_dataset)}")

    # Build model
    model = build_model(model_cfg, train_cfg)
    logger.info(f"Model type: {model.__class__.__name__}")

    # Get freeze components
    freeze_components = train_cfg.get("freeze_components", None)
    if freeze_components:
        logger.info(f"Freezing components: {freeze_components}")

    # Train
    logger.info("\nStarting training...")
    trainer = model.train_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        # Schedule
        max_steps=cfg.training.num_steps,
        lr_scheduler_type=cfg.training.scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        # Batch & optimization
        per_device_train_batch_size=cfg.training.train_batch_size,
        per_device_eval_batch_size=cfg.training.train_batch_size,
        learning_rate=float(cfg.training.lr_encoder),
        others_lr=float(cfg.training.lr_others),
        weight_decay=float(cfg.training.weight_decay_encoder),
        others_weight_decay=float(cfg.training.weight_decay_other),
        max_grad_norm=float(cfg.training.max_grad_norm),
        # Loss
        focal_loss_alpha=float(cfg.training.loss_alpha),
        focal_loss_gamma=float(cfg.training.loss_gamma),
        focal_loss_prob_margin=float(getattr(cfg.training, "loss_prob_margin", 0.0)),
        loss_reduction=cfg.training.loss_reduction,
        negatives=float(cfg.training.negatives),
        masking=cfg.training.masking,
        # Logging & saving
        save_steps=cfg.training.eval_every,
        logging_steps=cfg.training.eval_every,
        save_total_limit=cfg.training.save_total_limit,
        # Freezing
        freeze_components=freeze_components,
    )

    trainer.save_model()
    logger.success(f"\n✓ Training complete! Model saved to {output_dir}")


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
