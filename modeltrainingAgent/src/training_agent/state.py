from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

from training_agent.config import AppConfig

TaskType = Literal["classification", "detection", "segmentation", "llm_finetuning"]
ExecutionMode = Literal["local", "remote"]
ObjectiveDirection = Literal["minimize", "maximize"]
TrialStatus = Literal["none", "suggested", "validated", "running", "completed", "pruned", "failed"]
ExecutionStatus = Literal[
    "idle",
    "submitted",
    "running",
    "completed",
    "interrupted",
    "recoverable_error",
    "fatal_error",
]
WorkflowStatus = Literal["initializing", "optimizing", "completed", "failed"]
ErrorType = Literal[
    "oom",
    "nan_loss",
    "timeout",
    "job_interrupted",
    "invalid_data",
    "invalid_config",
    "dependency_error",
    "disk_full",
    "unknown",
]
StopReason = Literal[
    "target_reached",
    "max_trials_reached",
    "timeout_reached",
    "no_improvement",
    "user_cancelled",
    "fatal_error",
]


class StateEvent(TypedDict):
    timestamp: str
    node: str
    event: str
    message: str


class Diagnosis(TypedDict, total=False):
    problem: str
    confidence: float
    explanation: str
    suggested_actions: list[str]
    suggested_search_space_changes: dict[str, Any]


class TrainState(TypedDict):
    schema_version: int
    task_id: str
    workflow_status: WorkflowStatus
    created_at: str
    updated_at: str
    task_type: TaskType | None
    base_checkpoint_uri: str
    fixed_training_config: dict[str, Any]
    random_seed: int
    dataset_uri: str
    dataset_version: str | None
    train_dataset_uri: str | None
    validation_dataset_uri: str | None
    test_dataset_uri: str | None
    runtime_dataset_uri: str | None
    runtime_dataset: dict[str, Any]
    dataset_summary: dict[str, Any]
    input_validation_passed: bool
    study_name: str
    optuna_storage_key: str
    objective_name: str
    objective_direction: ObjectiveDirection
    search_space: dict[str, Any]
    search_space_version: int
    trial_request_id: str | None
    trial_number: int | None
    trial_status: TrialStatus
    trial_params: dict[str, Any]
    runtime_params: dict[str, Any]
    trial_attempt: int
    trial_started_at: str | None
    trial_finished_at: str | None
    execution_mode: ExecutionMode
    execution_status: ExecutionStatus
    external_job_id: str | None
    current_epoch: int
    total_epochs: int
    trial_checkpoint_uri: str | None
    trial_best_checkpoint_uri: str | None
    latest_metrics: dict[str, float]
    trial_best_metrics: dict[str, float]
    objective_value: float | None
    training_duration_seconds: float
    gpu_hours: float
    best_trial_number: int | None
    best_objective_value: float | None
    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    best_checkpoint_uri: str | None
    completed_trial_count: int
    failed_trial_count: int
    pruned_trial_count: int
    max_trials: int
    max_duration_seconds: float
    elapsed_seconds: float
    target_value: float | None
    patience: int
    min_delta: float
    no_improvement_count: int
    should_stop: bool
    stop_reason: StopReason | None
    error_type: ErrorType | None
    error_message: str | None
    error_node: str | None
    retryable: bool
    retry_count: int
    max_retries: int
    diagnosis: Diagnosis | None
    diagnosis_status: Literal["not_run", "completed", "failed"]
    final_model_uri: str | None
    final_report_uri: str | None
    events: Annotated[list[StateEvent], operator.add]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def event(node: str, name: str, message: str) -> StateEvent:
    return {"timestamp": now_iso(), "node": node, "event": name, "message": message}


def build_initial_state(config: AppConfig) -> TrainState:
    timestamp = now_iso()
    return {
        "schema_version": 1,
        "task_id": config.task.task_id,
        "workflow_status": "initializing",
        "created_at": timestamp,
        "updated_at": timestamp,
        "task_type": config.task.task_type,
        "base_checkpoint_uri": config.task.base_checkpoint_uri,
        "fixed_training_config": config.task.fixed_training_config,
        "random_seed": config.task.random_seed,
        "dataset_uri": config.task.dataset_uri,
        "dataset_version": None,
        "train_dataset_uri": None,
        "validation_dataset_uri": None,
        "test_dataset_uri": None,
        "runtime_dataset_uri": None,
        "runtime_dataset": {},
        "dataset_summary": {},
        "input_validation_passed": False,
        "study_name": config.optimization.study_name,
        "optuna_storage_key": config.optimization.storage_key,
        "objective_name": config.optimization.objective,
        "objective_direction": config.optimization.direction,
        "search_space": {k: v.model_dump() for k, v in config.optimization.search_space.items()},
        "search_space_version": 1,
        "trial_request_id": None,
        "trial_number": None,
        "trial_status": "none",
        "trial_params": {},
        "runtime_params": {},
        "trial_attempt": 0,
        "trial_started_at": None,
        "trial_finished_at": None,
        "execution_mode": config.execution.mode,
        "execution_status": "idle",
        "external_job_id": None,
        "current_epoch": 0,
        "total_epochs": config.training.epochs,
        "trial_checkpoint_uri": None,
        "trial_best_checkpoint_uri": None,
        "latest_metrics": {},
        "trial_best_metrics": {},
        "objective_value": None,
        "training_duration_seconds": 0,
        "gpu_hours": 0,
        "best_trial_number": None,
        "best_objective_value": None,
        "best_params": {},
        "best_metrics": {},
        "best_checkpoint_uri": None,
        "completed_trial_count": 0,
        "failed_trial_count": 0,
        "pruned_trial_count": 0,
        "max_trials": config.optimization.max_trials,
        "max_duration_seconds": config.optimization.max_duration_seconds,
        "elapsed_seconds": 0,
        "target_value": config.optimization.target_value,
        "patience": config.optimization.patience,
        "min_delta": config.optimization.min_delta,
        "no_improvement_count": 0,
        "should_stop": False,
        "stop_reason": None,
        "error_type": None,
        "error_message": None,
        "error_node": None,
        "retryable": False,
        "retry_count": 0,
        "max_retries": config.execution.max_retries,
        "diagnosis": None,
        "diagnosis_status": "not_run",
        "final_model_uri": None,
        "final_report_uri": None,
        "events": [event("init", "created", "initial state created")],
    }
