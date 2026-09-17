from __future__ import annotations

import json

from training_agent.data_preparation.core.paths import prepared_paths
from training_agent.data_preparation.models import PreparedPaths, SnapshotResult
from training_agent.data_preparation.services.checksum import verify_sha256
from training_agent.data_preparation.services.object_store import ObjectStoreClient


def fetch_manifests(
    snapshot: SnapshotResult,
    work_dir: str,
    object_store: ObjectStoreClient,
) -> PreparedPaths:
    paths = prepared_paths(work_dir, snapshot)
    paths.workspace.mkdir(parents=True, exist_ok=True)
    paths.result_path.write_text(
        json.dumps(snapshot.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    object_store.download_file(snapshot.train_manifest_uri, paths.train_manifest_path)
    object_store.download_file(snapshot.val_manifest_uri, paths.val_manifest_path)
    verify_sha256(paths.train_manifest_path, snapshot.train_sha256)
    verify_sha256(paths.val_manifest_path, snapshot.val_sha256)
    return paths

