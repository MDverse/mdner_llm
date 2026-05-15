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
    train_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Fraction of dataset used for training split.",
    )
    val_ratio: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of dataset used for validation split.",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle dataset before splitting.",
    )
    seed: int = Field(
        default=42,
        description="Random seed ensuring reproducibility of dataset splits.",
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
    batch_size: int = Field(
        default=2,
        ge=1,
        description="Number of samples per training batch.",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of forward/backward passes before performing an optimizer step. "
            "Used to simulate larger batch sizes under memory constraints."
        ),
    )
    cv_folds: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of cross-validation folds to use for training. If set to 1, "
            "no cross-validation is performed and the model is trained on the entire "
            "training set."
        ),
    )
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
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of total training steps used for linear learning rate warmup."
        ),
    )
    scheduler_type: Literal["cosine", "linear", "constant"] = Field(
        default="cosine",
        description="Learning rate scheduler strategy.",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description=(
            "Weight decay coefficient for regularization during AdamW optimization."
        ),
    )
    fp16: bool = Field(
        default=True,
        description="Enable mixed precision training using 16-bit floating point.",
    )
    bf16: bool = Field(
        default=False,
        description=("Enable bfloat16 precision training."),
    )
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
        default=["encoder"],
        description=(
            "List of model submodules to apply LoRA "
            "to (e.g., ['encoder'] applies to all encoder layers)."
        ),
    )
    save_adapter_only: bool = Field(
        default=True,
        description=(
            "Whether to save only the LoRA adapter weights instead "
            "of the full model during checkpointing."
        ),
    )
    eval_strategy: Literal["epoch", "steps", "no"] = Field(
        default="epoch",
        description="Frequency of evaluation during training.",
    )
    metric_for_best: str = Field(
        default="eval_loss",
        description="Metric used to select the best model checkpoint.",
    )
    save_best: bool = Field(
        default=True,
        description="Whether to save the best performing model during training.",
    )
    logging_steps: int = Field(
        default=50,
        ge=1,
        description="Number of steps between logging events.",
    )
    logging_first_step: bool = Field(
        default=True,
        description="Whether to log training metrics at the first step.",
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
    entities: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of entity types to their natural language descriptions.",
    )
