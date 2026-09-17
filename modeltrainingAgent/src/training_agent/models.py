from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    code: str
    message: str
    fatal: bool = True


class TrialSuggestion(BaseModel):
    request_id: str
    trial_number: int
    params: dict[str, Any]


class BestTrial(BaseModel):
    trial_number: int
    value: float
    params: dict[str, Any]


class TrialRecord(BaseModel):
    task_id: str
    trial_number: int
    request_id: str
    params: dict[str, Any]
    runtime_params: dict[str, Any]
    params_hash: str
    status: str = "suggested"


class ExecutionUpdate(BaseModel):
    task_id: str
    trial_number: int
    attempt: int
    status: str
    checkpoint_uri: str | None = None
    best_checkpoint_uri: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    current_epoch: int = 0
    duration_seconds: float = 0
    gpu_hours: float = 0
    error_type: str | None = None
    error_message: str | None = None


class TrainingContext(BaseModel):
    task_id: str
    trial_number: int
    attempt: int
    params: dict[str, Any]
    runtime_params: dict[str, Any]
    params_hash: str
    base_checkpoint_uri: str
    resume_checkpoint_uri: str | None
    train_dataset_uri: str
    validation_dataset_uri: str
    total_epochs: int
    artifact_dir: str
    fixed_training_config: dict[str, Any] = Field(default_factory=dict)
    behavior: Literal["completed", "oom", "interrupted", "nan", "fatal"] = "completed"


class TrainingResult(BaseModel):
    status: Literal["completed", "interrupted", "recoverable_error", "fatal_error"]
    metrics: dict[str, float] = Field(default_factory=dict)
    checkpoint_uri: str | None = None
    best_checkpoint_uri: str | None = None
    current_epoch: int = 0
    duration_seconds: float = 0
    gpu_hours: float = 0
    error_type: str | None = None
    error_message: str | None = None


class EvaluationContext(BaseModel):
    task_id: str
    trial_number: int
    checkpoint_uri: str
    validation_dataset_uri: str
    test_dataset_uri: str | None = None
    objective_name: str
    fixed_training_config: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    metrics: dict[str, float]


class RemoteTrainingRequest(BaseModel):
    request_id: str
    context: TrainingContext


class RemoteJobStatus(BaseModel):
    job_id: str
    status: Literal[
        "submitted", "running", "completed", "interrupted", "recoverable_error", "fatal_error"
    ]
    result: TrainingResult | None = None


class StructuredDiagnosis(BaseModel):
    problem: str = "none"
    confidence: float = 0.0
    explanation: str = "No external diagnosis service configured."
    suggested_actions: list[str] = Field(default_factory=list)
    suggested_search_space_changes: dict[str, Any] = Field(default_factory=dict)


class OptunaService(Protocol):
    def ensure_study(self) -> None: ...
    def ask(self, request_id: str, search_space: dict[str, Any]) -> TrialSuggestion: ...
    def complete(self, trial_number: int, value: float) -> None: ...
    def prune(self, trial_number: int, reason: str) -> None: ...
    def fail(self, trial_number: int, reason: str) -> None: ...
    def best_trial(self) -> BestTrial | None: ...
    def reconcile_running_trial(self, request_id: str) -> TrialSuggestion | None: ...


class TrialRepository(Protocol):
    def create_or_get_trial(self, record: TrialRecord) -> TrialRecord: ...
    def start_attempt(self, task_id: str, trial_number: int, attempt: int) -> None: ...
    def update_execution(self, update: ExecutionUpdate) -> None: ...
    def save_metrics(self, task_id: str, trial_number: int, metrics: dict[str, float]) -> None: ...
    def mark_completed(self, task_id: str, trial_number: int) -> None: ...
    def mark_failed(self, task_id: str, trial_number: int, reason: str) -> None: ...


class TrainerAdapter(Protocol):
    def validate(self, context: TrainingContext) -> list[ValidationIssue]: ...
    def train(self, context: TrainingContext) -> TrainingResult: ...
    def resume(self, context: TrainingContext) -> TrainingResult: ...
    def evaluate(self, context: EvaluationContext) -> EvaluationResult: ...


class RemoteExecutor(Protocol):
    def submit(self, request: RemoteTrainingRequest) -> str: ...
    def status(self, job_id: str) -> RemoteJobStatus: ...
    def cancel(self, job_id: str) -> None: ...


@dataclass(frozen=True)
class Services:
    optuna: OptunaService
    repository: TrialRepository
    artifacts: Any
    trainer: TrainerAdapter
    remote: RemoteExecutor | None = None
    diagnosis: Any = None
    config: Any = None
    remote_submissions: dict[str, str] = field(default_factory=dict)
