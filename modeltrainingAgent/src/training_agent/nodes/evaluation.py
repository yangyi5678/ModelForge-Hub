from __future__ import annotations

import math
from typing import Any

from training_agent.models import EvaluationContext, Services
from training_agent.state import TrainState, event, now_iso


def evaluate_trial(state: TrainState, services: Services) -> dict[str, Any]:
    if state["execution_status"] != "completed" or not state["trial_best_checkpoint_uri"]:
        raise ValueError("evaluate_trial requires completed execution and best checkpoint")
    assert state["trial_number"] is not None
    result = services.trainer.evaluate(
        EvaluationContext(
            task_id=state["task_id"],
            trial_number=state["trial_number"],
            checkpoint_uri=state["trial_best_checkpoint_uri"],
            validation_dataset_uri=state["validation_dataset_uri"] or state["dataset_uri"],
            objective_name=state["objective_name"],
            fixed_training_config=state["fixed_training_config"],
        )
    )
    value = result.metrics.get(state["objective_name"])
    if value is None or not math.isfinite(value):
        return {
            "execution_status": "fatal_error",
            "error_type": "unknown",
            "error_message": "objective metric is missing or non-finite",
            "error_node": "evaluate_trial",
            "stop_reason": "fatal_error",
            "should_stop": True,
        }
    services.repository.save_metrics(state["task_id"], state["trial_number"], result.metrics)
    return {
        "updated_at": now_iso(),
        "objective_value": value,
        "trial_best_metrics": result.metrics,
        "events": [event("evaluate_trial", "evaluated", f"objective={value}")],
    }
