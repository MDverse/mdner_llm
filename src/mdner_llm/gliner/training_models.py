"""
Configuration schema for GLiNER training pipeline.

This module defines structured Pydantic models used to validate and parse
YAML configuration files for model training, dataset preparation, and
optimization settings.
"""

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    model_validator,
)


class ModelConfig(BaseModel):
    """Configuration for the GLiNER model architecture and experiment setup."""

    name: str = Field(
        ...,
        description="Name or identifier of the pretrained model checkpoint to load.",
    )
    experiment_name: str = Field(
        ...,
        description="Name of the experiment used for logging and tracking runs.",
    )
    output_dir: Path = Field(
        ...,
        description="Directory where model checkpoints, logs, and outputs are saved.",
    )


class DataConfig(BaseModel):
    """Configuration describing dataset construction and splitting strategy."""

    annotations_path: DirectoryPath = Field(
        ...,
        description="Path to a directory containing annotation JSON files.",
    )
    cv_folds: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of cross-validation folds to use for training. If set to 1, "
            "no cross-validation (CV) is performed and the model is trained on the "
            "entire training set."
        ),
    )
    train_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Fraction of dataset used for training split in each CV fold.",
    )
    val_ratio: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of dataset used for validation split in each CV fold.",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle dataset before splitting.",
    )
    seed: int = Field(
        default=42,
        description=(
            "Random seed ensuring reproducibility of dataset splits and training."
        ),
    )
    max_tokens: int = Field(
        default=2048,  # using WhitespaceTokenSplitter, so equivalent to ~2048 words.
        ge=1,
        description="Maximum number of tokens per sequence when splitting "
        "long documents into chunks.",
    )
    overlap_tokens: int = Field(
        default=20,
        ge=0,
        description="Number of overlapping tokens between consecutive sequences.",
    )

    @model_validator(mode="after")
    def check_split_ratios(self) -> "DataConfig":
        """Validate that train/val split ratios sum to 1.0.

        Returns
        -------
        DataConfig
            The validated DataConfig instance.

        Raises
        ------
        ValueError
            If the sum of train_ratio and val_ratio does not equal 1
            within a small numerical tolerance.
        """
        total = self.train_ratio + self.val_ratio

        if abs(total - 1.0) > 1e-6:
            msg = f"Dataset split ratios must sum to 1.0, got {total}"
            raise ValueError(msg)

        return self


class TrainConfig(BaseModel):
    """Hyperparameters and optimization configuration for training GLiNER."""

    # DataLoader settings
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of subprocess workers for DataLoader.",
    )
    pin_memory: bool = Field(
        default=True,
        description="Whether to pin memory for faster host-to-GPU transfers.",
    )
    prefetch_factor: int | None = Field(
        default=2,
        ge=1,
        description="Number of batches loaded in advance by each worker.",
    )
    # Training steps & epochs
    max_steps: int = Field(
        default=-1,
        description=(
            "Maximum number of training steps. If set to -1, training will run for "
            "the number of epochs specified by num_epochs."
        ),
    )
    num_epochs: int = Field(
        default=50,
        ge=1,
        description=(
            "Number of complete passes through the training dataset. Ignored if "
            "max_steps is set to a positive integer."
        ),
    )
    # Batch size
    batch_size: int = Field(
        default=2,
        ge=1,
        description="Number of samples per training batch per GPU.",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of forward/backward passes before performing an optimizer step. "
            "Used to simulate larger batch sizes under memory constraints."
        ),
    )
    eval_batch_size: int = Field(
        default=4,
        ge=1,
        description="Number of samples per evaluation batch per GPU.",
    )
    # Learning rates & scheduler
    encoder_lr: float = Field(
        default=5e-6,
        gt=0,
        description="Learning rate applied to the encoder backbone.",
    )
    task_lr: float = Field(
        default=1e-4,
        gt=0,
        description="Learning rate applied to task-specific heads.",
    )
    warmup_ratio: float = Field(
        default=0.05,
        ge=0,
        le=0.2,
        description=(
            "Portion of total training steps where LR ramps up linearly from 0 to peak."
        ),
    )
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description=(
            "Exact step count to ramp up LR linearly. If > 0, overrides warmup_ratio."
        ),
    )
    scheduler_type: Literal["linear", "cosine", "cosine_restarts", "constant"] = Field(
        default="cosine",
        description="Learning rate scheduler strategy.",
    )
    # Optimization
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description=(
            "L2 penalty on weights to curb overfitting. "
            "Higher: forces smaller weights, fights overfitting, but can underfit. "
            "Lower: gives model full capacity, but risks overfitting on small datasets."
        ),
    )
    adam_beta1: float = Field(
        default=0.9,
        ge=0.0,
        lt=1.0,
        description=(
            "Exponential decay for 1st moment (gradient momentum/direction). "
            "Higher: smoother steps, ignores batch noise, but reacts slowly to changes."
            "Lower: adapts quickly to new gradients, but steps become noisy."
        ),
    )
    adam_beta2: float = Field(
        default=0.999,
        ge=0.0,
        lt=1.0,
        description=(
            "Exponential decay for 2nd moment (uncentered gradient variance/scale). "
            "Higher: stable scaling across epochs, standard for dense tasks."
            "Lower: tracks sudden gradient spikes faster, common in large-batch setups."
        ),
    )
    # Precision & hardware
    fp16: bool = Field(
        default=True,
        description="Enable mixed precision training using 16-bit floating point.",
    )
    bf16: bool = Field(
        default=False,
        description="Enable bfloat16 precision training.",
    )
    # LoRA settings
    use_lora: bool = Field(
        default=True,
        description="Enable LoRA (Low-Rank Adaptation) for efficient fine-tuning.",
    )
    lora_r: int = Field(
        default=4,
        ge=1,
        description="Rank (4, 8, 16, 32) for LoRA layers.",
    )
    lora_alpha: float = Field(
        default=8.0,
        gt=0,
        description="Scaling factor (usually 2*r) for LoRA layers.",
    )
    lora_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout rate for LoRA layers.",
    )
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["encoder"],
        description=(
            "List of model submodules to apply LoRA to "
            "(e.g., ['encoder'] applies to all encoder layers)."
        ),
    )
    save_adapter_only: bool = Field(
        default=True,
        description=(
            "Whether to save only the LoRA adapter weights instead "
            "of the full model during checkpointing."
        ),
    )
    # Checkpointing & evaluation strategy
    eval_strategy: Literal["epoch", "steps", "no"] = Field(
        default="epoch",
        description="Frequency of evaluation during training.",
    )
    eval_steps: int = Field(
        default=10,
        ge=1,
        description="Evaluate and save every N steps (when eval_strategy='steps').",
    )
    save_total_limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of checkpoints to retain on disk.",
    )
    save_best: bool = Field(
        default=True,
        description="Whether to save the best performing model during training.",
    )
    metric_for_best: str = Field(
        default="eval_loss",
        description="Metric used to select the best model checkpoint.",
    )
    greater_is_better: bool = Field(
        default=False,
        description="Whether a higher metric value corresponds to a better model.",
    )
    # Logging & tracking
    logging_steps: int = Field(
        default=5,
        ge=1,
        description="Number of steps between logging events.",
    )
    logging_first_step: bool = Field(
        default=True,
        description="Whether to log training metrics at the first step.",
    )
    # Early stopping
    early_stopping: bool = Field(
        default=False,
        description="Enable early stopping based on evaluation metrics.",
    )
    early_stopping_patience: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of evaluations without improvement before stopping training."
        ),
    )
    # Execution controls & reproducibility
    seed: int = Field(
        default=42,
        description="Random seed ensuring reproducibility during training.",
    )
    deterministic: bool = Field(
        default=True,
        description="Enforce deterministic PyTorch algorithms where available.",
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to reduce VRAM memory footprint.",
    )
    max_train_samples: int = Field(
        default=-1,
        description="Maximum number of training samples to load (-1 for all).",
    )
    max_eval_samples: int = Field(
        default=-1,
        description="Maximum number of evaluation samples to load (-1 for all).",
    )
    validate_data: bool = Field(
        default=True,
        description="Validate training dataset consistency prior to training.",
    )


class GLiNERConfig(BaseModel):
    """Root configuration combining model, data, and training settings."""

    model: ModelConfig = Field(
        ...,
        description="Model architecture and experiment configuration.",
    )
    data: DataConfig = Field(
        ...,
        description="Dataset loading, splitting, and serialization configuration.",
    )
    training: TrainConfig = Field(
        ...,
        description="Optimization and training hyperparameters.",
    )
    categories: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of entity types to their natural language descriptions.",
    )
