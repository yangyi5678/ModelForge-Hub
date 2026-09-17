from __future__ import annotations

from typing import Any

from training_agent.data_preparation.core.fetch_manifests import fetch_manifests
from training_agent.data_preparation.models import SnapshotResult
from training_agent.data_preparation.nodes.common import fail_state
from training_agent.data_preparation.services.object_store import ObjectStoreClient
from training_agent.data_preparation.state import DataPreparationState
from training_agent.state import event, now_iso


def fetch_manifests_node(state: DataPreparationState) -> dict[str, Any]:
    try:
        snapshot = SnapshotResult.model_validate(state["snapshot"])
        paths = fetch_manifests(snapshot, state["work_dir"], ObjectStoreClient())
        return {
            "updated_at": now_iso(),
            "local_train_manifest_path": str(paths.train_manifest_path),
            "local_val_manifest_path": str(paths.val_manifest_path),
            "events": [event("fetch_manifests", "downloaded", "downloaded train/val manifests")],
        }
    except Exception as exc:
        return fail_state("fetch_manifests", exc)

