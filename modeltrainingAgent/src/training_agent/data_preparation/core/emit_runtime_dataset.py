from __future__ import annotations

import json
from pathlib import Path

from training_agent.data_preparation.models import RuntimeDataset, SnapshotResult


def emit_runtime_dataset(
    *,
    snapshot: SnapshotResult,
    runtime_dataset_path: str | Path,
    local_train_manifest_path: str,
    local_val_manifest_path: str,
    train_dataset_uri: str,
    validation_dataset_uri: str,
    local_data_root: str,
) -> RuntimeDataset:
    runtime = RuntimeDataset(
        dataset_name=snapshot.dataset_name,
        version=snapshot.version,
        snapshot_id=snapshot.snapshot_id,
        result_uri=snapshot.result_uri,
        local_train_manifest_path=local_train_manifest_path,
        local_val_manifest_path=local_val_manifest_path,
        train_dataset_uri=train_dataset_uri,
        validation_dataset_uri=validation_dataset_uri,
        local_data_root=local_data_root,
        train_count=snapshot.train_count,
        val_count=snapshot.val_count,
    )
    path = Path(runtime_dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
    return runtime
