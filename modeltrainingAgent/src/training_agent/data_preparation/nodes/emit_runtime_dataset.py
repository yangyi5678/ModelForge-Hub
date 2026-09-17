from __future__ import annotations

from typing import Any

from training_agent.data_preparation.core.emit_runtime_dataset import emit_runtime_dataset
from training_agent.data_preparation.core.paths import prepared_paths
from training_agent.data_preparation.models import SnapshotResult
from training_agent.data_preparation.nodes.common import fail_state
from training_agent.data_preparation.state import DataPreparationState
from training_agent.state import event, now_iso


def emit_runtime_dataset_node(state: DataPreparationState) -> dict[str, Any]:
    try:
        snapshot = SnapshotResult.model_validate(state["snapshot"])
        paths = prepared_paths(state["work_dir"], snapshot)
        runtime = emit_runtime_dataset(
            snapshot=snapshot,
            runtime_dataset_path=paths.runtime_dataset_path,
            local_train_manifest_path=state["local_train_manifest_path"],
            local_val_manifest_path=state["local_val_manifest_path"],
            train_dataset_uri=state["train_dataset_uri"],
            validation_dataset_uri=state["validation_dataset_uri"],
            local_data_root=state["local_data_root"],
        )
        return {
            "updated_at": now_iso(),
            "runtime_dataset_uri": str(paths.runtime_dataset_path),
            "runtime_dataset": runtime.model_dump(),
            "events": [
                event("emit_runtime_dataset", "emitted", str(paths.runtime_dataset_path))
            ],
        }
    except Exception as exc:
        return fail_state("emit_runtime_dataset", exc)

