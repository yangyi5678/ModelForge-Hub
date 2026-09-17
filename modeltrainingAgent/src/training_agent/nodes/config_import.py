from __future__ import annotations

from typing import Any

from training_agent.services.config_import import (
    import_model_project_config_generate_train_task_core,
)
from training_agent.state import TrainState, event, now_iso


def import_model_project_config_node(state: TrainState) -> dict[str, Any]:
    fixed = dict(state.get("fixed_training_config", {}))
    model_config_uri = fixed.get("model_project_config_uri")
    if not model_config_uri:
        return {}

    result = import_model_project_config_generate_train_task_core(
        model_project_config_uri=model_config_uri,
        project_type=fixed.get("project_type", "generic"),
        task_id=fixed.get("generated_task_id") or state["task_id"],
        output_uri=fixed.get("agent_config_output_uri"),
        entrypoint_module=fixed.get("entrypoint_module"),
    )
    generated_config = result.generated_config
    generated_task = generated_config["task"]
    generated_optimization = generated_config["optimization"]
    generated_training = generated_config["training"]
    next_fixed = dict(generated_task.get("fixed_training_config", {}))
    next_fixed["generated_agent_config_uri"] = result.agent_config_uri

    return {
        "updated_at": now_iso(),
        "task_type": generated_task.get("task_type", state["task_type"]),
        "dataset_uri": generated_task.get("dataset_uri", state["dataset_uri"]),
        "base_checkpoint_uri": generated_task.get(
            "base_checkpoint_uri", state["base_checkpoint_uri"]
        ),
        "fixed_training_config": next_fixed,
        "study_name": generated_optimization.get("study_name", state["study_name"]),
        "objective_name": generated_optimization.get("objective", state["objective_name"]),
        "objective_direction": generated_optimization.get(
            "direction", state["objective_direction"]
        ),
        "search_space": generated_optimization.get("search_space", state["search_space"]),
        "max_trials": generated_optimization.get("max_trials", state["max_trials"]),
        "target_value": generated_optimization.get("target_value", state["target_value"]),
        "patience": generated_optimization.get("patience", state["patience"]),
        "min_delta": generated_optimization.get("min_delta", state["min_delta"]),
        "total_epochs": generated_training.get("epochs", state["total_epochs"]),
        "events": [
            event(
                "import_model_project_config",
                "generated",
                f"generated agent config at {result.agent_config_uri}",
            )
        ],
    }
