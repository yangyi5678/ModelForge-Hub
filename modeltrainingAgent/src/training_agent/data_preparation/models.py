from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SnapshotResult(BaseModel):
    dataset_name: str
    version: str
    snapshot_id: str
    status: str
    schema_version: int
    result_uri: str
    snapshot_uri: str | None = None
    source_uri: str | None = None
    source_sha256: str | None = None
    train_manifest_uri: str
    val_manifest_uri: str
    train_sha256: str | None = None
    val_sha256: str | None = None
    train_count: int = 0
    val_count: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)


class ManifestStats(BaseModel):
    path: str
    sample_count: int
    fields: list[str]
    empty_lines: int = 0


class RuntimeDataset(BaseModel):
    dataset_name: str
    version: str
    snapshot_id: str
    result_uri: str
    local_train_manifest_path: str
    local_val_manifest_path: str
    train_dataset_uri: str
    validation_dataset_uri: str
    local_data_root: str
    train_count: int
    val_count: int
    manifest_type: str = "sample_manifest"
    model_dataset_format: str = "manifest_passthrough"


class PreparedPaths(BaseModel):
    workspace: Path
    result_path: Path
    train_manifest_path: Path
    val_manifest_path: Path
    runtime_dataset_path: Path

