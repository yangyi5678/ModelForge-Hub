from __future__ import annotations

import math
import time
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

from training_agent.models import (
    EvaluationContext,
    EvaluationResult,
    TrainingContext,
    TrainingResult,
    ValidationIssue,
)
from training_agent.services.entrypoint_loader import load_python_entrypoint


class VLMTrainerAdapter:
    """Adapter that connects the Agent to a VLM project with standard entrypoints."""

    def validate(self, context: TrainingContext) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        fixed = context.fixed_training_config
        config_uri = fixed.get("model_config_uri")
        if not isinstance(config_uri, str) or not config_uri:
            issues.append(
                ValidationIssue(
                    code="missing_model_config", message="model_config_uri is required"
                )
            )
        for key in (
            "model_entrypoint",
            "loss_entrypoint",
            "optimizer_entrypoint",
            "scheduler_entrypoint",
        ):
            uri = fixed.get(key)
            if not isinstance(uri, str) or not uri:
                issues.append(ValidationIssue(code=f"missing_{key}", message=f"{key} is required"))
                continue
            try:
                load_python_entrypoint(uri)
            except Exception as exc:
                issues.append(ValidationIssue(code=f"bad_{key}", message=str(exc)))
        return issues

    def train(self, context: TrainingContext) -> TrainingResult:
        return self._run(context, resume_checkpoint=None)

    def resume(self, context: TrainingContext) -> TrainingResult:
        if not context.resume_checkpoint_uri:
            return TrainingResult(
                status="fatal_error",
                error_type="unknown",
                error_message="resume requested without checkpoint",
            )
        return self._run(context, resume_checkpoint=context.resume_checkpoint_uri)

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        loss = _checkpoint_loss(Path(context.checkpoint_uri))
        metric = loss if math.isfinite(loss) else 0.0
        metrics = {context.objective_name: metric, "val_loss": metric}
        if context.test_dataset_uri is not None:
            metrics[f"test_{context.objective_name}"] = metric
        return EvaluationResult(metrics=metrics)

    def _run(
        self, context: TrainingContext, resume_checkpoint: str | None
    ) -> TrainingResult:
        started = time.monotonic()
        try:
            cfg = self._build_cfg(context)
            self._validate_entrypoints(context)
            train_fn = self._load_train_function(context)
            output_dir = Path(train_fn(cfg, resume_checkpoint=resume_checkpoint))
            checkpoint = output_dir / "last.ckpt"
            loss = _checkpoint_loss(checkpoint) if checkpoint.exists() else 0.0
            metrics = {"val_loss": loss} if math.isfinite(loss) else {}
            return TrainingResult(
                status="completed",
                metrics=metrics,
                checkpoint_uri=str(checkpoint),
                best_checkpoint_uri=str(checkpoint),
                current_epoch=context.total_epochs,
                duration_seconds=time.monotonic() - started,
                gpu_hours=0,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                return TrainingResult(
                    status="recoverable_error",
                    error_type="oom",
                    error_message=message,
                )
            return TrainingResult(
                status="fatal_error",
                error_type="unknown",
                error_message=message,
            )
        except Exception as exc:
            return TrainingResult(
                status="fatal_error",
                error_type="unknown",
                error_message=str(exc),
            )

    def _build_cfg(self, context: TrainingContext) -> dict[str, Any]:
        fixed = context.fixed_training_config
        config_uri = fixed["model_config_uri"]
        with Path(config_uri).open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        if not isinstance(cfg, dict):
            raise ValueError("VLM model config must be a YAML mapping")
        cfg = deepcopy(cfg)
        training = dict(cfg.get("training", {}))
        training["epochs"] = context.total_epochs
        training.update(
            {
                "learning_rate": context.params.get(
                    "learning_rate", training.get("learning_rate")
                ),
                "weight_decay": context.params.get(
                    "weight_decay", training.get("weight_decay")
                ),
                "batch_size": context.params.get("batch_size", training.get("batch_size")),
                "gradient_accumulation_steps": context.runtime_params.get(
                    "gradient_accumulation_steps",
                    training.get("gradient_accumulation_steps", 1),
                ),
                "num_workers": context.runtime_params.get(
                    "num_workers", training.get("num_workers", 0)
                ),
            }
        )
        cfg["training"] = {key: value for key, value in training.items() if value is not None}
        cfg["output"] = {"dir": context.artifact_dir}
        return cfg

    def _validate_entrypoints(self, context: TrainingContext) -> None:
        fixed = context.fixed_training_config
        for key in (
            "model_entrypoint",
            "loss_entrypoint",
            "optimizer_entrypoint",
            "scheduler_entrypoint",
        ):
            load_python_entrypoint(str(fixed[key]))

    def _load_train_function(self, context: TrainingContext) -> Any:
        train_entrypoint = context.fixed_training_config.get("train_entrypoint")
        if isinstance(train_entrypoint, str) and train_entrypoint:
            return load_python_entrypoint(train_entrypoint)
        return load_python_entrypoint("python://vlm_project.train:train")


def _checkpoint_loss(checkpoint: Path) -> float:
    if not checkpoint.exists():
        return 0.0
    torch = import_module("torch")
    payload = torch.load(checkpoint, map_location="cpu")
    value = payload.get("loss", 0.0)
    if value is None:
        return 0.0
    return float(value)
