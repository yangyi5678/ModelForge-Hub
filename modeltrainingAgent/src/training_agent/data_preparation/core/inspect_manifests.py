from __future__ import annotations

from training_agent.data_preparation.models import ManifestStats, SnapshotResult
from training_agent.data_preparation.services.manifest_reader import inspect_jsonl


def inspect_manifests(
    snapshot: SnapshotResult,
    train_manifest_path: str,
    val_manifest_path: str,
) -> tuple[ManifestStats, ManifestStats, dict[str, object]]:
    train_stats = inspect_jsonl(train_manifest_path)
    val_stats = inspect_jsonl(val_manifest_path)
    if snapshot.train_count and train_stats.sample_count != snapshot.train_count:
        raise ValueError(
            "train manifest count mismatch: "
            f"expected {snapshot.train_count}, got {train_stats.sample_count}"
        )
    if snapshot.val_count and val_stats.sample_count != snapshot.val_count:
        raise ValueError(
            f"val manifest count mismatch: expected {snapshot.val_count}, got {val_stats.sample_count}"
        )
    fields = sorted(set(train_stats.fields) | set(val_stats.fields))
    summary: dict[str, object] = {
        "dataset_name": snapshot.dataset_name,
        "version": snapshot.version,
        "snapshot_id": snapshot.snapshot_id,
        "sample_count": train_stats.sample_count + val_stats.sample_count,
        "train_count": train_stats.sample_count,
        "val_count": val_stats.sample_count,
        "fields": fields,
        "manifest": {
            "train": train_stats.model_dump(),
            "val": val_stats.model_dump(),
        },
        "source_summary": snapshot.summary,
    }
    return train_stats, val_stats, summary

