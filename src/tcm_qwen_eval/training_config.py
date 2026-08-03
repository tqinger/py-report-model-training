"""Typed configuration loading for the tongue QLoRA training run."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuantizationConfig:
    """4-bit model-loading options."""

    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: str
    torch_dtype: str

    def __post_init__(self) -> None:
        if not self.load_in_4bit:
            raise ValueError("quantization.load_in_4bit must be true for QLoRA training")
        if self.bnb_4bit_quant_type != "nf4":
            raise ValueError("quantization.bnb_4bit_quant_type must be 'nf4'")
        supported_dtypes = {"bfloat16", "float16"}
        if self.bnb_4bit_compute_dtype not in supported_dtypes:
            raise ValueError(
                "quantization.bnb_4bit_compute_dtype must be one of "
                f"{sorted(supported_dtypes)}"
            )
        if self.torch_dtype not in supported_dtypes:
            raise ValueError(f"quantization.torch_dtype must be one of {sorted(supported_dtypes)}")


@dataclass(frozen=True)
class LoraHyperparameters:
    """Adapter capacity and target-layer options."""

    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if self.r <= 0:
            raise ValueError("lora.r must be positive")
        if self.alpha <= 0:
            raise ValueError("lora.alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("lora.dropout must be in [0, 1)")
        if not self.target_modules or any(not module for module in self.target_modules):
            raise ValueError("lora.target_modules must contain at least one module name")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("lora.target_modules must not contain duplicates")


@dataclass(frozen=True)
class TrainingHyperparameters:
    """Trainer, sequence, and reproducibility options."""

    seed: int
    max_length: int
    num_train_epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    max_grad_norm: float
    optim: str
    eval_strategy: str
    save_strategy: str
    logging_strategy: str
    logging_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    bf16: bool
    tf32: bool
    gradient_checkpointing: bool
    gradient_checkpointing_use_reentrant: bool
    report_to: str
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("training.max_length must be positive")
        if self.num_train_epochs <= 0:
            raise ValueError("training.num_train_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("training.per_device_train_batch_size must be positive")
        if self.per_device_eval_batch_size <= 0:
            raise ValueError("training.per_device_eval_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("training.warmup_ratio must be in [0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("training.max_grad_norm must be positive")
        if self.logging_strategy == "steps" and self.logging_steps <= 0:
            raise ValueError("training.logging_steps must be positive when logging_strategy is 'steps'")
        if self.save_total_limit <= 0:
            raise ValueError("training.save_total_limit must be positive")
        if not self.lr_scheduler_type:
            raise ValueError("training.lr_scheduler_type must not be empty")
        if self.lr_scheduler_kwargs is not None and not isinstance(self.lr_scheduler_kwargs, dict):
            raise ValueError("training.lr_scheduler_kwargs must be a TOML inline table or omitted")
        if self.lr_scheduler_type == "cosine_with_min_lr":
            scheduler_kwargs = self.lr_scheduler_kwargs or {}
            if "min_lr" in scheduler_kwargs or "min_lr_rate" not in scheduler_kwargs:
                raise ValueError(
                    "training.cosine_with_min_lr requires only lr_scheduler_kwargs.min_lr_rate"
                )
            min_lr_rate = scheduler_kwargs["min_lr_rate"]
            if (
                not isinstance(min_lr_rate, (int, float))
                or isinstance(min_lr_rate, bool)
                or not 0 <= min_lr_rate <= 1
            ):
                raise ValueError("training.lr_scheduler_kwargs.min_lr_rate must be in [0, 1]")
        if self.load_best_model_at_end and self.eval_strategy != self.save_strategy:
            raise ValueError(
                "training.eval_strategy and training.save_strategy must match when "
                "load_best_model_at_end is true"
            )


@dataclass(frozen=True)
class TongueQLoRAConfig:
    """Complete hyperparameter set for a tongue QLoRA run."""

    quantization: QuantizationConfig
    lora: LoraHyperparameters
    training: TrainingHyperparameters


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    section = payload.get(name)
    if not isinstance(section, dict):
        raise TypeError(f"Configuration requires a [{name}] table")
    return section


def _construct(config_class: type[Any], section_name: str, values: dict[str, Any]) -> Any:
    try:
        return config_class(**values)
    except TypeError as error:
        raise ValueError(f"Invalid [{section_name}] configuration: {error}") from error


def load_tongue_qlora_config(path: Path) -> TongueQLoRAConfig:
    """Load and validate a TOML file containing all training hyperparameters."""
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Training configuration does not exist: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML in training configuration {path}: {error}") from error

    return TongueQLoRAConfig(
        quantization=_construct(QuantizationConfig, "quantization", _section(payload, "quantization")),
        lora=_construct(LoraHyperparameters, "lora", _section(payload, "lora")),
        training=_construct(TrainingHyperparameters, "training", _section(payload, "training")),
    )
