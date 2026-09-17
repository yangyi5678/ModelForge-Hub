from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from training_agent.state import StateEvent


class DataPreparationState(TypedDict, total=False):
    result_uri: str
    work_dir: str
    model_family: str
    snapshot: dict[str, Any]
    dataset_name: str
    version: str
    snapshot_id: str
    train_manifest_uri: str
    val_manifest_uri: str
    local_train_manifest_path: str
    local_val_manifest_path: str
    train_dataset_uri: str
    validation_dataset_uri: str
    local_data_root: str
    model_dataset_format: str
    runtime_dataset_uri: str
    runtime_dataset: dict[str, Any]
    dataset_version: str
    dataset_summary: dict[str, Any]
    error_type: str | None
    error_message: str | None
    error_node: str | None
    should_stop: bool
    events: Annotated[list[StateEvent], operator.add]
