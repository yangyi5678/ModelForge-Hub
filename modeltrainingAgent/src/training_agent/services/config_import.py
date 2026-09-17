from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from training_agent.config import AppConfig


class SearchParamDraft(BaseModel):
    type: Literal["float", "int", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    choices: list[Any] | None = None


class GeneratedTrainTaskConfig(BaseModel):
    task_id: str
    task_type: Literal["classification", "detection", "segmentation", "llm_finetuning"]
    dataset_uri: str
    base_checkpoint_uri: str
    random_seed: int = 42
    fixed_training_config: dict[str, Any]
    execution: dict[str, Any]
    optimization: dict[str, Any]
    training: dict[str, Any]
    artifacts_dir: str = "artifacts"
    repository_uri: str = "artifacts/trials.sqlite"

    def to_app_config_dict(self) -> dict[str, Any]:
        return {
            "task": {
                "task_id": self.task_id,
                "task_type": self.task_type,
                "dataset_uri": self.dataset_uri,
                "base_checkpoint_uri": self.base_checkpoint_uri,
                "random_seed": self.random_seed,
                "fixed_training_config": self.fixed_training_config,
            },
            "execution": self.execution,
            "optimization": self.optimization,
            "training": self.training,
            "artifacts_dir": self.artifacts_dir,
            "repository_uri": self.repository_uri,
        }


class ImportModelProjectConfigResult(BaseModel):
    status: Literal["completed"]
    agent_config_uri: str
    generated_config: dict[str, Any]
    structured_output: GeneratedTrainTaskConfig
    llm_used: bool = False


def import_model_project_config_generate_train_task_core(
    *,
    model_project_config_uri: str,
    project_type: Literal["vlm", "generic"] = "generic",
    task_id: str | None = None,
    output_uri: str | None = None,
    entrypoint_module: str | None = None,
    llm: Any | None = None,
) -> ImportModelProjectConfigResult:
    """Generate an Agent task config from a model-project config.

    If an LLM object with ``with_structured_output`` is supplied, it is asked to
    produce ``GeneratedTrainTaskConfig``. A deterministic structured fallback is
    used otherwise, and also validates the final output through Pydantic and
    ``AppConfig``.
    """

    config_path = _local_path_from_uri(model_project_config_uri)
    model_config = _read_yaml(config_path)
    default_draft = _deterministic_draft(
        model_config=model_config,
        model_project_config_uri=model_project_config_uri,
        project_type=project_type,
        task_id=task_id,
        entrypoint_module=entrypoint_module,
    )
    draft = _llm_draft_or_default(
        llm=llm,
        default_draft=default_draft,
        model_config=model_config,
        model_project_config_uri=model_project_config_uri,
        project_type=project_type,
    )
    app_config_dict = draft.to_app_config_dict()
    AppConfig.model_validate(app_config_dict)

    output_path = _output_path(output_uri, draft.task_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(app_config_dict, handle, sort_keys=False, allow_unicode=True)
    tmp.replace(output_path)

    return ImportModelProjectConfigResult(
        status="completed",
        agent_config_uri=str(output_path),
        generated_config=app_config_dict,
        structured_output=draft,
        llm_used=draft != default_draft,
    )


def _llm_draft_or_default(
    *,
    llm: Any | None,
    default_draft: GeneratedTrainTaskConfig,
    model_config: dict[str, Any],
    model_project_config_uri: str,
    project_type: str,
) -> GeneratedTrainTaskConfig:
    if llm is None or not hasattr(llm, "with_structured_output"):
        return default_draft
    prompt = (
        "Generate an Agent training task config from this model-project config. "
        "Preserve entrypoint URIs and use safe HPO search spaces. "
        f"project_type={project_type}; model_config_uri={model_project_config_uri}; "
        f"default_draft={default_draft.model_dump()}; model_config={model_config}"
    )
    structured_llm = llm.with_structured_output(GeneratedTrainTaskConfig)
    result = structured_llm.invoke(prompt)
    return GeneratedTrainTaskConfig.model_validate(result)


def _deterministic_draft(
    *,
    model_config: dict[str, Any],
    model_project_config_uri: str,
    project_type: str,
    task_id: str | None,
    entrypoint_module: str | None,
) -> GeneratedTrainTaskConfig:
    default_vlm_entrypoint = "vlm_project.adapters.training_agent_entrypoints"
    if project_type == "vlm":
        return _vlm_draft(
            model_config=model_config,
            model_project_config_uri=model_project_config_uri,
            task_id=task_id,
            entrypoint_module=entrypoint_module or default_vlm_entrypoint,
        )
    return _generic_draft(
        model_config=model_config,
        model_project_config_uri=model_project_config_uri,
        task_id=task_id,
        entrypoint_module=entrypoint_module or "vlm_project.adapters.training_agent_entrypoints",
    )


def _vlm_draft(
    *,
    model_config: dict[str, Any],
    model_project_config_uri: str,
    task_id: str | None,
    entrypoint_module: str,
) -> GeneratedTrainTaskConfig:
    training = model_config.get("training", {})
    model = model_config.get("model", {})
    generated_task_id = task_id or _slug_from_model_path(model.get("path")) or "vlm-hpo-001"
    learning_rate = float(training.get("learning_rate", 2e-4))
    batch_size = int(training.get("batch_size", 1))
    epochs = int(training.get("epochs", 3))
    return GeneratedTrainTaskConfig(
        task_id=generated_task_id,
        task_type="llm_finetuning",
        dataset_uri=model_project_config_uri,
        base_checkpoint_uri=str(model.get("path", "")),
        fixed_training_config={
            "model_config_uri": model_project_config_uri,
            "model_entrypoint": f"python://{entrypoint_module}:build_model",
            "loss_entrypoint": f"python://{entrypoint_module}:build_loss",
            "optimizer_entrypoint": f"python://{entrypoint_module}:build_optimizer",
            "scheduler_entrypoint": f"python://{entrypoint_module}:build_scheduler",
            "trainer_adapter": "vlm",
        },
        execution={"mode": "local", "max_retries": 3, "mock_behavior": "completed"},
        optimization={
            "study_name": generated_task_id,
            "storage_key": "OPTUNA_STORAGE_MAIN",
            "sampler": "tpe",
            "direction": "minimize",
            "objective": "val_loss",
            "target_value": None,
            "max_trials": 5,
            "max_duration_seconds": 3600,
            "patience": 3,
            "min_delta": 0.001,
            "search_space": {
                "learning_rate": {
                    "type": "float",
                    "low": max(1e-6, learning_rate / 10),
                    "high": max(learning_rate * 3, learning_rate + 1e-6),
                    "log": True,
                },
                "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
                "batch_size": {
                    "type": "categorical",
                    "choices": sorted({1, batch_size, max(1, batch_size * 2)}),
                },
            },
        },
        training={"epochs": epochs, "precision": str(model.get("dtype", "bfloat16"))},
    )


def _generic_draft(
    *,
    model_config: dict[str, Any],
    model_project_config_uri: str,
    task_id: str | None,
    entrypoint_module: str,
) -> GeneratedTrainTaskConfig:
    training = model_config.get("training", {})
    model = model_config.get("model", {})
    generated_task_id = task_id or "generic-hpo-001"
    return GeneratedTrainTaskConfig(
        task_id=generated_task_id,
        task_type="llm_finetuning",
        dataset_uri=model_project_config_uri,
        base_checkpoint_uri=str(model.get("path", "")),
        fixed_training_config={
            "model_config_uri": model_project_config_uri,
            "model_entrypoint": f"python://{entrypoint_module}:build_model",
            "loss_entrypoint": f"python://{entrypoint_module}:build_loss",
            "optimizer_entrypoint": f"python://{entrypoint_module}:build_optimizer",
            "scheduler_entrypoint": f"python://{entrypoint_module}:build_scheduler",
        },
        execution={"mode": "local", "max_retries": 3, "mock_behavior": "completed"},
        optimization={
            "study_name": generated_task_id,
            "storage_key": "OPTUNA_STORAGE_MAIN",
            "sampler": "tpe",
            "direction": "minimize",
            "objective": "val_loss",
            "max_trials": 5,
            "search_space": {
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
            },
        },
        training={"epochs": int(training.get("epochs", 3)), "precision": "fp32"},
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("model project config must be a YAML mapping")
    return data


def _local_path_from_uri(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(uri.removeprefix("file://"))
    return Path(uri)


def _output_path(output_uri: str | None, task_id: str) -> Path:
    if output_uri:
        return _local_path_from_uri(output_uri)
    return Path("modeltrainingAgent") / "configs" / "generated" / f"{task_id}.yaml"


def _slug_from_model_path(path: Any) -> str | None:
    if not path:
        return None
    name = Path(str(path)).name.lower().replace(".", "-").replace("_", "-")
    return f"{name}-hpo-001"
