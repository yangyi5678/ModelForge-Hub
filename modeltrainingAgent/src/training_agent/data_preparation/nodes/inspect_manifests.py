from __future__ import annotations

from typing import Any

from training_agent.data_preparation.core.inspect_manifests import inspect_manifests
from training_agent.data_preparation.models import SnapshotResult
from training_agent.data_preparation.nodes.common import fail_state
from training_agent.data_preparation.state import DataPreparationState
from training_agent.state import event, now_iso


def inspect_manifests_node(state: DataPreparationState) -> dict[str, Any]:
    try:
        snapshot = SnapshotResult.model_validate(state["snapshot"])
        train_stats, val_stats, summary = inspect_manifests(
            snapshot,
            state["local_train_manifest_path"],
            state["local_val_manifest_path"],
        )
        return {
            "updated_at": now_iso(),
            "dataset_summary": summary,
            "events": [
                event(
                    "inspect_manifests",
                    "inspected",
                    f"train={train_stats.sample_count} val={val_stats.sample_count}",
                )
            ],
        }
    except Exception as exc:
        return fail_state("inspect_manifests", exc)

