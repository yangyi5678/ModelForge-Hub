from __future__ import annotations

from typing import Any

from training_agent.data_preparation.core.resolve_snapshot import resolve_snapshot
from training_agent.data_preparation.nodes.common import fail_state
from training_agent.data_preparation.services.object_store import ObjectStoreClient
from training_agent.data_preparation.state import DataPreparationState
from training_agent.state import event, now_iso


def resolve_snapshot_node(state: DataPreparationState) -> dict[str, Any]:
    try:
        snapshot = resolve_snapshot(state["result_uri"], ObjectStoreClient())
        return {
            "updated_at": now_iso(),
            "snapshot": snapshot.model_dump(),
            "dataset_name": snapshot.dataset_name,
            "version": snapshot.version,
            "snapshot_id": snapshot.snapshot_id,
            "train_manifest_uri": snapshot.train_manifest_uri,
            "val_manifest_uri": snapshot.val_manifest_uri,
            "dataset_version": snapshot.source_sha256 or snapshot.snapshot_id,
            "events": [
                event(
                    "resolve_snapshot",
                    "resolved",
                    f"resolved {snapshot.dataset_name}/{snapshot.version}/{snapshot.snapshot_id}",
                )
            ],
        }
    except Exception as exc:
        return fail_state("resolve_snapshot", exc)

