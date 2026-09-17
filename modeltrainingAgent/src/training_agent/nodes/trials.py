from __future__ import annotations

from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

from training_agent.models import Services, TrialRecord
from training_agent.routes import decide_stop, improved
from training_agent.state import TrainState, event, now_iso
from training_agent.trainers.base import stable_params_hash


def suggest_trial(state: TrainState, services: Services) -> dict[str, Any]:
    if state["trial_number"] is not None:
        return {}
    next_index = state["completed_trial_count"] + state["failed_trial_count"]
    next_index += state["pruned_trial_count"] + 1
    request_id = state["trial_request_id"] or str(
        uuid5(NAMESPACE_URL, f"{state['task_id']}:{next_index}")
    )
    repo_trial = getattr(services.repository, "trial_by_request", lambda *_: None)(
        state["task_id"], request_id
    )
    if repo_trial is not None:
        suggestion = repo_trial
        trial_number = suggestion.trial_number
        params = suggestion.params
    else:
        optuna_trial = services.optuna.ask(request_id, state["search_space"])
        trial_number = optuna_trial.trial_number
        params = optuna_trial.params
    batch = int(params.get("batch_size", 16))
    runtime_params = {
        "micro_batch_size": min(batch, 16),
        "gradient_accumulation_steps": max(1, batch // min(batch, 16)),
        "num_workers": 0,
    }
    return {
        "updated_at": now_iso(),
        "trial_request_id": request_id,
        "trial_number": trial_number,
        "trial_status": "suggested",
        "trial_params": params,
        "runtime_params": runtime_params,
        "trial_attempt": 1,
        "execution_status": "idle",
        "external_job_id": None,
        "current_epoch": 0,
        "trial_checkpoint_uri": None,
        "trial_best_checkpoint_uri": None,
        "latest_metrics": {},
        "trial_best_metrics": {},
        "objective_value": None,
        "error_type": None,
        "error_message": None,
        "error_node": None,
        "retryable": False,
        "events": [event("suggest_trial", "suggested", f"trial {trial_number} suggested")],
    }


def validate_trial(state: TrainState, services: Services) -> dict[str, Any]:
    if state["trial_number"] is None:
        raise ValueError("validate_trial called without active trial")
    params = state["trial_params"]
    batch = int(params.get("batch_size", 1))
    runtime = dict(state["runtime_params"])
    if int(runtime.get("micro_batch_size", 1)) > batch:
        runtime["micro_batch_size"] = batch
    context = _context(state, services, resume=False)
    issues = services.trainer.validate(context)
    if issues:
        reason = "; ".join(issue.message for issue in issues)
        services.optuna.fail(state["trial_number"], reason)
        services.repository.mark_failed(state["task_id"], state["trial_number"], reason)
        return {
            "trial_status": "failed",
            "execution_status": "fatal_error",
            "error_type": "invalid_config",
            "error_message": reason,
            "error_node": "validate_trial",
            "stop_reason": "fatal_error",
            "should_stop": True,
            "events": [event("validate_trial", "failed", reason)],
        }
    return {
        "updated_at": now_iso(),
        "runtime_params": runtime,
        "trial_status": "validated",
        "events": [event("validate_trial", "validated", "trial validated")],
    }


def persist_trial(state: TrainState, services: Services) -> dict[str, Any]:
    assert state["trial_number"] is not None
    params_hash = stable_params_hash(state["trial_params"])
    record = TrialRecord(
        task_id=state["task_id"],
        trial_number=state["trial_number"],
        request_id=state["trial_request_id"] or "",
        params=state["trial_params"],
        runtime_params=state["runtime_params"],
        params_hash=params_hash,
        status="running",
    )
    services.repository.create_or_get_trial(record)
    services.repository.start_attempt(
        state["task_id"], state["trial_number"], state["trial_attempt"]
    )
    services.artifacts.write_json(
        f"{state['task_id']}/trials/trial_{state['trial_number']:04d}/trial_config.json",
        record.model_dump(),
    )
    return {
        "updated_at": now_iso(),
        "trial_status": "running",
        "trial_started_at": state["trial_started_at"] or now_iso(),
        "events": [event("persist_trial", "persisted", "trial persisted")],
    }


def resume_trial(state: TrainState, services: Services) -> dict[str, Any]:
    if not state["trial_checkpoint_uri"]:
        return {
            "execution_status": "fatal_error",
            "error_type": "unknown",
            "error_message": "cannot resume without checkpoint",
            "error_node": "resume_trial",
            "stop_reason": "fatal_error",
            "should_stop": True,
        }
    attempt = state["trial_attempt"] + 1
    services.repository.start_attempt(state["task_id"], state["trial_number"] or -1, attempt)
    return {
        "updated_at": now_iso(),
        "trial_attempt": attempt,
        "execution_status": "idle",
        "retry_count": state["retry_count"] + 1,
        "events": [event("resume_trial", "resuming", "resuming same trial")],
    }


def repair_trial(state: TrainState, services: Services) -> dict[str, Any]:
    if state["error_type"] != "oom" or state["retry_count"] >= state["max_retries"]:
        trial = state["trial_number"]
        if trial is not None:
            services.optuna.fail(trial, state["error_message"] or "unrecoverable error")
            services.repository.mark_failed(
                state["task_id"], trial, state["error_message"] or "failed"
            )
        return {
            "trial_status": "failed",
            "execution_status": "fatal_error",
            "stop_reason": "fatal_error",
            "should_stop": True,
            "events": [event("repair_trial", "failed", "repair retries exhausted")],
        }
    runtime = dict(state["runtime_params"])
    old_micro = int(runtime.get("micro_batch_size", 1))
    batch = int(state["trial_params"].get("batch_size", old_micro))
    new_micro = max(1, old_micro // 2)
    runtime["micro_batch_size"] = new_micro
    runtime["gradient_accumulation_steps"] = max(1, batch // new_micro)
    return {
        "updated_at": now_iso(),
        "runtime_params": runtime,
        "trial_attempt": state["trial_attempt"] + 1,
        "retry_count": state["retry_count"] + 1,
        "execution_status": "idle",
        "events": [event("repair_trial", "repaired", f"micro_batch_size {old_micro}->{new_micro}")],
    }


def complete_optuna_trial(state: TrainState, services: Services) -> dict[str, Any]:
    assert state["trial_number"] is not None
    assert state["objective_value"] is not None
    services.optuna.complete(state["trial_number"], state["objective_value"])
    services.repository.mark_completed(state["task_id"], state["trial_number"])
    best = services.optuna.best_trial()
    current_is_best = improved(
        state["objective_value"],
        state["best_objective_value"],
        state["objective_direction"],
        state["min_delta"],
    )
    no_improvement = 0 if current_is_best else state["no_improvement_count"] + 1
    counts: dict[str, int] = getattr(services.optuna, "counts", lambda: {})()
    elapsed = state["elapsed_seconds"] + state["training_duration_seconds"]
    update: dict[str, Any] = {
        "updated_at": now_iso(),
        "completed_trial_count": counts.get("completed", state["completed_trial_count"] + 1),
        "failed_trial_count": counts.get("failed", state["failed_trial_count"]),
        "pruned_trial_count": counts.get("pruned", state["pruned_trial_count"]),
        "no_improvement_count": no_improvement,
        "elapsed_seconds": elapsed,
        "trial_finished_at": now_iso(),
        "events": [event("complete_optuna_trial", "completed", "optuna trial completed")],
    }
    if best is not None:
        update.update(
            {
                "best_trial_number": best.trial_number,
                "best_objective_value": best.value,
                "best_params": best.params,
                "best_metrics": state["trial_best_metrics"]
                if best.trial_number == state["trial_number"]
                else state["best_metrics"],
                "best_checkpoint_uri": state["trial_best_checkpoint_uri"]
                if best.trial_number == state["trial_number"]
                else state["best_checkpoint_uri"],
            }
        )
    shadow = cast(TrainState, {**state, **update})
    should_stop, reason = decide_stop(shadow)
    update.update(
        {
            "should_stop": should_stop,
            "stop_reason": reason,
            "trial_request_id": None,
            "trial_number": None,
            "trial_status": "none",
            "trial_params": {},
            "runtime_params": {},
            "trial_attempt": 0,
            "execution_status": "idle",
            "external_job_id": None,
        }
    )
    return update


def _context(state: TrainState, services: Services, resume: bool) -> Any:
    from training_agent.nodes.execution import build_training_context

    return build_training_context(state, services, resume=resume)
