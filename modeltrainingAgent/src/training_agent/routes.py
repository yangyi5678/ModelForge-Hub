from __future__ import annotations

from typing import Literal

from training_agent.state import TrainState


def route_execution_mode(state: TrainState) -> Literal["local", "remote"]:
    return state["execution_mode"]


def route_after_execution(state: TrainState) -> str:
    match state["execution_status"]:
        case "completed":
            return "completed"
        case "interrupted":
            return "interrupted"
        case "recoverable_error":
            return "recoverable"
        case "fatal_error":
            return "fatal"
        case _:
            raise ValueError("execution has not reached a routable state")


def reached_target(value: float, target: float, direction: str) -> bool:
    if direction == "maximize":
        return value >= target
    return value <= target


def improved(new_value: float, old_value: float | None, direction: str, min_delta: float) -> bool:
    if old_value is None:
        return True
    if direction == "maximize":
        return new_value > old_value + min_delta
    return new_value < old_value - min_delta


def route_after_trial(state: TrainState) -> str:
    if state["execution_status"] == "fatal_error" or state["stop_reason"] == "fatal_error":
        return "fatal"
    if state["should_stop"]:
        return "target_or_budget_reached"
    return "continue"


def decide_stop(state: TrainState) -> tuple[bool, str | None]:
    value = state["best_objective_value"]
    target = state["target_value"]
    if state["stop_reason"] == "fatal_error":
        return True, "fatal_error"
    if (
        value is not None
        and target is not None
        and reached_target(value, target, state["objective_direction"])
    ):
        return True, "target_reached"
    if state["completed_trial_count"] >= state["max_trials"]:
        return True, "max_trials_reached"
    if state["elapsed_seconds"] >= state["max_duration_seconds"]:
        return True, "timeout_reached"
    if state["no_improvement_count"] >= state["patience"]:
        return True, "no_improvement"
    return False, None
