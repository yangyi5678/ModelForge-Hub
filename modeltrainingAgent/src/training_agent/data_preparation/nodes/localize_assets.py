from __future__ import annotations

from typing import Any

from training_agent.data_preparation.core.localize_assets import localize_assets
from training_agent.data_preparation.core.paths import build_workspace
from training_agent.data_preparation.models import SnapshotResult
from training_agent.data_preparation.nodes.common import fail_state
from training_agent.data_preparation.services.object_store import ObjectStoreClient
from training_agent.data_preparation.state import DataPreparationState
from training_agent.state import event, now_iso


def localize_assets_node(state: DataPreparationState) -> dict[str, Any]:
    try:
        snapshot = SnapshotResult.model_validate(state["snapshot"])
        workspace = build_workspace(state["work_dir"], snapshot)
        result = localize_assets(
            train_manifest_path=state["local_train_manifest_path"],
            val_manifest_path=state["local_val_manifest_path"],
            workspace=str(workspace),
            object_store=ObjectStoreClient(),
        )
        return {
            "updated_at": now_iso(),
            **result,
            "events": [
                event(
                    "localize_assets",
                    "localized",
                    f"downloaded train={result['localized_train_count']} val={result['localized_val_count']}",
                )
            ],
        }
    except Exception as exc:
        return fail_state("localize_assets", exc)
