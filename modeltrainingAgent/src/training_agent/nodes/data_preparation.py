from __future__ import annotations

from pathlib import Path
from typing import Any

from training_agent.data_preparation.graph import build_data_preparation_graph
from training_agent.models import Services
from training_agent.state import TrainState, event, now_iso


def prepare_dataset_snapshot(state: TrainState, services: Services) -> dict[str, Any]:
    dataset_uri = state["dataset_uri"]
    if not _looks_like_snapshot_result(dataset_uri):
        return {
            "updated_at": now_iso(),
            "events": [event("prepare_dataset", "skipped", "dataset_uri is not a snapshot result")],
        }

    config = services.config
    fixed = state["fixed_training_config"]
    data_prep_cfg = fixed.get("data_preparation", {})
    if not isinstance(data_prep_cfg, dict):
        data_prep_cfg = {}
    work_dir = str(
        data_prep_cfg.get(
            "work_dir",
            Path(config.artifacts_dir) / state["task_id"] / "datasets",
        )
    )
    model_family = str(data_prep_cfg.get("model_family", fixed.get("trainer_adapter", "manifest")))

    subgraph = build_data_preparation_graph()
    prepared = subgraph.invoke(
        {
            "result_uri": dataset_uri,
            "work_dir": work_dir,
            "model_family": model_family,
            "events": [],
        }
    )
    if prepared.get("should_stop"):
        return {
            "updated_at": now_iso(),
            "workflow_status": "failed",
            "execution_status": "fatal_error",
            "error_type": "invalid_data",
            "error_message": prepared.get("error_message", "dataset preparation failed"),
            "error_node": prepared.get("error_node", "prepare_dataset"),
            "retryable": False,
            "stop_reason": "fatal_error",
            "should_stop": True,
            "events": prepared.get("events", [])
            + [event("prepare_dataset", "failed", "dataset preparation failed")],
        }

    return {
        "updated_at": now_iso(),
        "dataset_version": prepared.get("dataset_version"),
        "dataset_summary": prepared.get("dataset_summary", {}),
        "train_dataset_uri": prepared.get("train_dataset_uri"),
        "validation_dataset_uri": prepared.get("validation_dataset_uri"),
        "runtime_dataset_uri": prepared.get("runtime_dataset_uri"),
        "runtime_dataset": prepared.get("runtime_dataset", {}),
        "events": prepared.get("events", [])
        + [event("prepare_dataset", "prepared", "dataset snapshot prepared")],
    }


def _looks_like_snapshot_result(uri: str) -> bool:
    return uri.endswith("result.json")
