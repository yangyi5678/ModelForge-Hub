from __future__ import annotations

from pathlib import Path

from training_agent.data_preparation.models import PreparedPaths, SnapshotResult


def build_workspace(work_dir: str | Path, snapshot: SnapshotResult | None = None) -> Path:
    root = Path(work_dir)
    if snapshot is None:
        return root
    return root / snapshot.dataset_name / snapshot.version / snapshot.snapshot_id


def prepared_paths(work_dir: str | Path, snapshot: SnapshotResult) -> PreparedPaths:
    workspace = build_workspace(work_dir, snapshot)
    return PreparedPaths(
        workspace=workspace,
        result_path=workspace / "result.json",
        train_manifest_path=workspace / "train_manifest.jsonl",
        val_manifest_path=workspace / "val_manifest.jsonl",
        runtime_dataset_path=workspace / "runtime_dataset.json",
    )

