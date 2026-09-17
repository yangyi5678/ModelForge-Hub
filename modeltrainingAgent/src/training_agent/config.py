from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TaskType = Literal["classification", "detection", "segmentation", "llm_finetuning"]
ExecutionMode = Literal["local", "remote"]
ObjectiveDirection = Literal["minimize", "maximize"]


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    optuna_storage_main: str = "sqlite:///artifacts/optuna.db"
    langgraph_checkpoint_db: str = "sqlite:///artifacts/langgraph.db"
    oss_endpoint: str | None = None
    oss_access_key_id: str | None = None
    oss_access_key_secret: str | None = None


def load_env_file() -> EnvSettings:
    env = EnvSettings()
    mappings = {
        "OSS_ENDPOINT": env.oss_endpoint,
        "OSS_ACCESS_KEY_ID": env.oss_access_key_id,
        "OSS_ACCESS_KEY_SECRET": env.oss_access_key_secret,
    }
    for key, value in mappings.items():
        if value and key not in os.environ:
            os.environ[key] = value
    return env


class TaskConfig(BaseModel):
    task_id: str
    task_type: TaskType | None = None
    dataset_uri: str
    base_checkpoint_uri: str
    random_seed: int = 42
    fixed_training_config: dict[str, Any] = Field(default_factory=dict)


class ExecutionConfig(BaseModel):
    mode: ExecutionMode = "local"
    max_retries: int = 3
    mock_behavior: Literal["completed", "oom", "interrupted", "nan", "fatal"] = "completed"


class SearchParam(BaseModel):
    type: Literal["float", "int", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    step: float | int | None = None
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def validate_param(self) -> SearchParam:
        if self.type in {"float", "int"}:
            if self.low is None or self.high is None:
                raise ValueError("numeric search parameters require low and high")
            if not self.low < self.high:
                raise ValueError("search parameter low must be less than high")
            if self.log and self.low <= 0:
                raise ValueError("log search parameter low must be greater than 0")
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("categorical search parameter requires non-empty choices")
            normalized = [repr(choice) for choice in self.choices]
            if len(normalized) != len(set(normalized)):
                raise ValueError("categorical choices must be unique")
        return self


class OptimizationConfig(BaseModel):
    study_name: str
    storage_key: str = "OPTUNA_STORAGE_MAIN"
    sampler: Literal["tpe", "random"] = "tpe"
    direction: ObjectiveDirection
    objective: str
    target_value: float | None = None
    max_trials: int = 20
    max_duration_seconds: float = 3600
    patience: int = 6
    min_delta: float = 0.001
    search_space: dict[str, SearchParam]

    @field_validator("search_space")
    @classmethod
    def non_empty_space(cls, value: dict[str, SearchParam]) -> dict[str, SearchParam]:
        if not value:
            raise ValueError("search_space must not be empty")
        return value


class TrainingConfig(BaseModel):
    epochs: int = 10
    precision: str = "fp32"


class AppConfig(BaseModel):
    task: TaskConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    optimization: OptimizationConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    artifacts_dir: str = "artifacts"
    repository_uri: str = "artifacts/trials.sqlite"

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        load_env_file()
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.model_validate(data)

    def optuna_storage_url(self, env: EnvSettings | None = None) -> str:
        raw = os.getenv(self.optimization.storage_key)
        if raw:
            return raw
        env = env or EnvSettings()
        key = self.optimization.storage_key.lower()
        configured = getattr(env, key, None)
        if configured and self.optimization.storage_key in os.environ:
            return str(configured)
        return f"sqlite:///{Path(self.artifacts_dir) / 'optuna.db'}"
