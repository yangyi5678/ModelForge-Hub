from __future__ import annotations

import json
import math
import os
from pathlib import Path

from training_agent.errors import FatalTrainingError, TrainingInterruptedError, TrainingOOMError
from training_agent.models import (
    EvaluationContext,
    EvaluationResult,
    TrainingContext,
    TrainingResult,
    ValidationIssue,
)


class DeterministicMockTrainerAdapter:
    def validate(self, context: TrainingContext) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        lr = float(context.params.get("learning_rate", 0))
        if lr <= 0:
            issues.append(ValidationIssue(code="bad_lr", message="learning_rate must be positive"))
        if int(context.runtime_params.get("micro_batch_size", 1)) < 1:
            issues.append(
                ValidationIssue(code="bad_micro_batch", message="micro batch must be positive")
            )
        if context.resume_checkpoint_uri:
            self._read_checkpoint(context.resume_checkpoint_uri, context)
        return issues

    def train(self, context: TrainingContext) -> TrainingResult:
        return self._run(context, resume=False)

    def resume(self, context: TrainingContext) -> TrainingResult:
        if not context.resume_checkpoint_uri:
            raise TrainingInterruptedError("resume requested without checkpoint")
        self._read_checkpoint(context.resume_checkpoint_uri, context)
        return self._run(context, resume=True)

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        checkpoint = json.loads(Path(context.checkpoint_uri).read_text(encoding="utf-8"))
        metric = float(checkpoint["metric"])
        metrics = {
            context.objective_name: metric,
            "val_loss": round(max(0.01, 1.0 - metric), 6),
        }
        if context.test_dataset_uri is not None:
            metrics[f"test_{context.objective_name}"] = round(max(0.0, metric - 0.015), 6)
        return EvaluationResult(metrics=metrics)

    def _run(self, context: TrainingContext, resume: bool) -> TrainingResult:
        try:
            if context.behavior == "oom" and context.attempt == 1:
                raise TrainingOOMError("mock out of memory")
            if context.behavior == "fatal":
                raise FatalTrainingError("mock fatal failure")
            if context.behavior == "nan":
                return TrainingResult(
                    status="fatal_error",
                    error_type="nan_loss",
                    error_message="mock NaN loss",
                    current_epoch=1,
                )
            start_epoch = 1
            if resume and context.resume_checkpoint_uri:
                ckpt = self._read_checkpoint(context.resume_checkpoint_uri, context)
                start_epoch = int(str(ckpt["epoch"])) + 1
            if context.behavior == "interrupted" and context.attempt == 1:
                epoch = max(1, context.total_epochs // 2)
                last = self._write_checkpoint(context, epoch)
                return TrainingResult(
                    status="interrupted",
                    metrics={"val_f1": self._metric(context, epoch)},
                    checkpoint_uri=last,
                    best_checkpoint_uri=last,
                    current_epoch=epoch,
                    duration_seconds=0.01 * epoch,
                    error_type="job_interrupted",
                    error_message="mock interruption",
                )
            best_uri = None
            latest_metric = 0.0
            for epoch in range(start_epoch, context.total_epochs + 1):
                latest_metric = self._metric(context, epoch)
                best_uri = self._write_checkpoint(context, epoch, latest_metric)
            assert best_uri is not None
            return TrainingResult(
                status="completed",
                metrics={"val_f1": latest_metric, "train_loss": round(1.0 - latest_metric, 6)},
                checkpoint_uri=best_uri,
                best_checkpoint_uri=best_uri,
                current_epoch=context.total_epochs,
                duration_seconds=0.01 * context.total_epochs,
                gpu_hours=0,
            )
        except TrainingOOMError as exc:
            return TrainingResult(
                status="recoverable_error",
                error_type="oom",
                error_message=str(exc),
                current_epoch=0,
            )
        except FatalTrainingError as exc:
            return TrainingResult(
                status="fatal_error", error_type="unknown", error_message=str(exc)
            )

    def _metric(self, context: TrainingContext, epoch: int) -> float:
        lr = float(context.params.get("learning_rate", 1e-4))
        wd = float(context.params.get("weight_decay", 0.0))
        batch = float(context.params.get("batch_size", 16))
        lr_score = max(0.0, 1.0 - abs(math.log10(lr) - math.log10(3e-4)) / 2.0)
        wd_score = max(0.0, 1.0 - abs(wd - 0.02) / 0.2)
        batch_score = 1.0 if batch == 16 else 0.95 if batch == 32 else 0.9
        progress = epoch / max(1, context.total_epochs)
        value = 0.55 + 0.32 * lr_score + 0.08 * wd_score + 0.04 * batch_score
        return round(min(0.995, value * (0.75 + 0.25 * progress)), 6)

    def _write_checkpoint(
        self, context: TrainingContext, epoch: int, metric: float | None = None
    ) -> str:
        attempt_dir = Path(context.artifact_dir)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        path = attempt_dir / "last.ckpt"
        metric = self._metric(context, epoch) if metric is None else metric
        payload = {
            "kind": "mock_checkpoint",
            "task_id": context.task_id,
            "trial_number": context.trial_number,
            "attempt": context.attempt,
            "epoch": epoch,
            "params_hash": context.params_hash,
            "metric": metric,
            "optimizer_state": {"mock": True},
            "scheduler_state": {"mock": True},
            "rng_state": {"seed": 0},
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        best = attempt_dir.parent.parent / "best.ckpt"
        tmp_best = best.with_suffix(".tmp")
        tmp_best.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_best, best)
        return str(best)

    def _read_checkpoint(self, uri: str, context: TrainingContext) -> dict[str, object]:
        path = Path(uri)
        if not path.exists():
            raise TrainingInterruptedError("checkpoint does not exist")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("task_id") != context.task_id
            or payload.get("trial_number") != context.trial_number
        ):
            raise TrainingInterruptedError("checkpoint does not belong to current trial")
        if payload.get("params_hash") != context.params_hash:
            raise TrainingInterruptedError("checkpoint params hash mismatch")
        return payload
