from __future__ import annotations

from typing import Any

from training_agent.models import Services
from training_agent.state import TrainState, event, now_iso


def prepare_runtime(state: TrainState, services: Services) -> dict[str, Any]:
    services.optuna.ensure_study()
    task_id = state["task_id"]
    services.artifacts.task_dir(task_id)
    services.artifacts.write_json(f"{task_id}/task_config.json", {"task_id": task_id})
    train_dataset_uri = state["train_dataset_uri"] or state["dataset_uri"]
    validation_dataset_uri = state["validation_dataset_uri"] or state["dataset_uri"]
    test_dataset_uri = state["test_dataset_uri"] or validation_dataset_uri
    services.artifacts.write_json(
        f"{task_id}/dataset_manifest.json",
        {
            "dataset_uri": state["dataset_uri"],
            "dataset_version": state["dataset_version"],
            "train_dataset_uri": train_dataset_uri,
            "validation_dataset_uri": validation_dataset_uri,
            "test_dataset_uri": test_dataset_uri,
        },
    )
    return {
        "updated_at": now_iso(),
        "workflow_status": "optimizing",
        "train_dataset_uri": train_dataset_uri,
        "validation_dataset_uri": validation_dataset_uri,
        "test_dataset_uri": test_dataset_uri,
        "events": [event("prepare_runtime", "prepared", "runtime prepared")],
    }
