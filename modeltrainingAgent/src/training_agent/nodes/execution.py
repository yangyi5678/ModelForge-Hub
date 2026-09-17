from __future__ import annotations

from typing import Any

from training_agent.executors.local import LocalExecutor
from training_agent.models import ExecutionUpdate, RemoteTrainingRequest, Services, TrainingContext
from training_agent.state import TrainState, event, now_iso
from training_agent.trainers.base import stable_params_hash


def build_training_context(
    state: TrainState, services: Services, resume: bool = False
) -> TrainingContext:
    assert state["trial_number"] is not None
    attempt_dir = services.artifacts.attempt_dir(
        state["task_id"], state["trial_number"], state["trial_attempt"]
    )
    return TrainingContext(
        task_id=state["task_id"],
        trial_number=state["trial_number"],
        attempt=state["trial_attempt"],
        params=state["trial_params"],
        runtime_params=state["runtime_params"],
        params_hash=stable_params_hash(state["trial_params"]),
        base_checkpoint_uri=state["base_checkpoint_uri"],
        resume_checkpoint_uri=state["trial_checkpoint_uri"] if resume else None,
        train_dataset_uri=state["train_dataset_uri"] or state["dataset_uri"],
        validation_dataset_uri=state["validation_dataset_uri"] or state["dataset_uri"],
        total_epochs=state["total_epochs"],
        artifact_dir=str(attempt_dir),
        fixed_training_config=state["fixed_training_config"],
        behavior=getattr(services.config.execution, "mock_behavior", "completed"),
    )


def run_training(state: TrainState, services: Services) -> dict[str, Any]:
    context = build_training_context(
        state, services, resume=state["trial_checkpoint_uri"] is not None
    )
    result = LocalExecutor(services.trainer).run(context)
    assert state["trial_number"] is not None
    services.repository.update_execution(
        ExecutionUpdate(
            task_id=state["task_id"],
            trial_number=state["trial_number"],
            attempt=state["trial_attempt"],
            status=result.status,
            checkpoint_uri=result.checkpoint_uri,
            best_checkpoint_uri=result.best_checkpoint_uri,
            metrics=result.metrics,
            current_epoch=result.current_epoch,
            duration_seconds=result.duration_seconds,
            gpu_hours=result.gpu_hours,
            error_type=result.error_type,
            error_message=result.error_message,
        )
    )
    return {
        "updated_at": now_iso(),
        "execution_status": result.status,
        "latest_metrics": result.metrics,
        "trial_checkpoint_uri": result.checkpoint_uri,
        "trial_best_checkpoint_uri": result.best_checkpoint_uri,
        "current_epoch": result.current_epoch,
        "training_duration_seconds": result.duration_seconds,
        "gpu_hours": result.gpu_hours,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "error_node": "run_training" if result.error_type else None,
        "retryable": result.status in {"interrupted", "recoverable_error"},
        "events": [
            event("run_training", result.status, result.error_message or "training finished")
        ],
    }


def submit_job(state: TrainState, services: Services) -> dict[str, Any]:
    if state["external_job_id"]:
        return {"execution_status": "submitted"}
    if services.remote is None:
        raise ValueError("remote execution requested without remote executor")
    request_id = f"{state['task_id']}:{state['trial_number']}:{state['trial_attempt']}"
    context = build_training_context(
        state, services, resume=state["trial_checkpoint_uri"] is not None
    )
    job_id = services.remote.submit(RemoteTrainingRequest(request_id=request_id, context=context))
    return {
        "updated_at": now_iso(),
        "external_job_id": job_id,
        "execution_status": "submitted",
        "events": [event("submit_job", "submitted", job_id)],
    }


def monitor_job(state: TrainState, services: Services) -> dict[str, Any]:
    if services.remote is None or state["external_job_id"] is None:
        raise ValueError("monitor_job requires remote executor and job id")
    job = services.remote.status(state["external_job_id"])
    result = job.result
    if result is None:
        return {
            "execution_status": job.status,
            "events": [event("monitor_job", job.status, "pending")],
        }
    assert state["trial_number"] is not None
    services.repository.update_execution(
        ExecutionUpdate(
            task_id=state["task_id"],
            trial_number=state["trial_number"],
            attempt=state["trial_attempt"],
            status=result.status,
            checkpoint_uri=result.checkpoint_uri,
            best_checkpoint_uri=result.best_checkpoint_uri,
            metrics=result.metrics,
            current_epoch=result.current_epoch,
            duration_seconds=result.duration_seconds,
            gpu_hours=result.gpu_hours,
            error_type=result.error_type,
            error_message=result.error_message,
        )
    )
    return {
        "updated_at": now_iso(),
        "execution_status": result.status,
        "latest_metrics": result.metrics,
        "trial_checkpoint_uri": result.checkpoint_uri,
        "trial_best_checkpoint_uri": result.best_checkpoint_uri,
        "current_epoch": result.current_epoch,
        "training_duration_seconds": result.duration_seconds,
        "gpu_hours": result.gpu_hours,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "error_node": "monitor_job" if result.error_type else None,
        "retryable": result.status in {"interrupted", "recoverable_error"},
        "events": [event("monitor_job", result.status, result.error_message or "remote finished")],
    }
