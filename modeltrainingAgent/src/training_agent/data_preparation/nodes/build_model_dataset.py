from __future__ import annotations

from typing import Any

from training_agent.data_preparation.core.build_model_dataset import build_model_dataset
from training_agent.data_preparation.nodes.common import fail_state
from training_agent.data_preparation.state import DataPreparationState
from training_agent.state import event, now_iso


def build_model_dataset_node(state: DataPreparationState) -> dict[str, Any]:
    try:
        result = build_model_dataset(
            train_dataset_uri=state["train_dataset_uri"],
            validation_dataset_uri=state["validation_dataset_uri"],
            model_family=state.get("model_family", "manifest"),
        )
        return {
            "updated_at": now_iso(),
            **result,
            "events": [
                event(
                    "build_model_dataset",
                    "built",
                    f"prepared {result['model_family']} dataset contract",
                )
            ],
        }
    except Exception as exc:
        return fail_state("build_model_dataset", exc)

