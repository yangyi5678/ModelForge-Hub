from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training_agent.models import EvaluationContext, Services
from training_agent.state import TrainState, event, now_iso


def finalize(state: TrainState, services: Services) -> dict[str, Any]:
    best = services.optuna.best_trial()
    if best is None or not state["best_checkpoint_uri"]:
        return failure_report(
            {
                **state,
                "error_node": "finalize",
                "error_type": "unknown",
                "error_message": "no best trial available",
            },
            services,
        )
    checkpoint = Path(state["best_checkpoint_uri"])
    if not checkpoint.exists():
        return failure_report(
            {
                **state,
                "error_node": "finalize",
                "error_type": "unknown",
                "error_message": "best checkpoint is missing",
            },
            services,
        )
    test_metrics: dict[str, float] = {}
    if state["test_dataset_uri"]:
        result = services.trainer.evaluate(
            EvaluationContext(
                task_id=state["task_id"],
                trial_number=best.trial_number,
                checkpoint_uri=str(checkpoint),
                validation_dataset_uri=state["validation_dataset_uri"] or state["dataset_uri"],
                test_dataset_uri=state["test_dataset_uri"],
                objective_name=state["objective_name"],
                fixed_training_config=state["fixed_training_config"],
            )
        )
        test_metrics = {k: v for k, v in result.metrics.items() if k.startswith("test_")}
    model_ref = {
        "task_id": state["task_id"],
        "trial_number": best.trial_number,
        "checkpoint_uri": str(checkpoint),
        "params": best.params,
    }
    final_dir = services.artifacts.final_dir(state["task_id"])
    model_uri = services.artifacts.write_json(
        Path(state["task_id"]) / "final" / "best_model.ref.json", model_ref
    )
    report: dict[str, Any] = {
        "task": {
            "task_id": state["task_id"],
            "task_type": state["task_type"],
            "dataset_version": state["dataset_version"],
        },
        "study": {
            "study_name": state["study_name"],
            "objective_name": state["objective_name"],
            "objective_direction": state["objective_direction"],
            "search_space": state["search_space"],
            "search_space_version": state["search_space_version"],
        },
        "trials": getattr(services.optuna, "summaries", lambda: [])(),
        "best_trial": {
            "trial_number": best.trial_number,
            "value": best.value,
            "params": best.params,
            "validation_metrics": state["best_metrics"],
            "test_metrics": test_metrics,
        },
        "resources": {
            "elapsed_seconds": state["elapsed_seconds"],
            "gpu_hours": state["gpu_hours"],
        },
        "counts": {
            "completed": state["completed_trial_count"],
            "failed": state["failed_trial_count"],
            "pruned": state["pruned_trial_count"],
        },
        "stop_reason": state["stop_reason"],
        "reproducible_command": f"training-agent resume --task-id {state['task_id']}",
    }
    report_uri = services.artifacts.write_json(
        Path(state["task_id"]) / "final" / "report.json", report
    )
    getattr(services.repository, "set_final_report", lambda *_: None)(state["task_id"], report_uri)
    return {
        "updated_at": now_iso(),
        "workflow_status": "completed",
        "final_model_uri": model_uri,
        "final_report_uri": report_uri,
        "events": [event("finalize", "completed", str(final_dir))],
    }


def failure_report(state: TrainState | dict[str, Any], services: Services) -> dict[str, Any]:
    task_id = str(state["task_id"])
    report = {
        "task_id": task_id,
        "workflow_status": "failed",
        "error_node": state.get("error_node"),
        "error_type": state.get("error_type"),
        "error_message": state.get("error_message"),
        "trial_number": state.get("trial_number"),
        "checkpoint_uri": state.get("trial_checkpoint_uri"),
        "can_resume": bool(state.get("trial_checkpoint_uri")),
        "resume_command": f"training-agent resume --task-id {task_id}",
    }
    uri = services.artifacts.write_json(Path(task_id) / "final" / "failure_report.json", report)
    return {
        "updated_at": now_iso(),
        "workflow_status": "failed",
        "final_report_uri": uri,
        "should_stop": True,
        "stop_reason": "fatal_error",
        "events": [event("failure_report", "failed", json.dumps(report, sort_keys=True))],
    }
