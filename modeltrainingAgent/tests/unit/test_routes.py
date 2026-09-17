from __future__ import annotations

from training_agent.routes import decide_stop, improved, reached_target, route_after_execution
from training_agent.state import build_initial_state


def test_target_direction() -> None:
    assert reached_target(0.9, 0.8, "maximize")
    assert not reached_target(0.7, 0.8, "maximize")
    assert reached_target(0.1, 0.2, "minimize")
    assert not reached_target(0.3, 0.2, "minimize")


def test_improvement_direction() -> None:
    assert improved(0.9, 0.8, "maximize", 0.01)
    assert not improved(0.805, 0.8, "maximize", 0.01)
    assert improved(0.1, 0.2, "minimize", 0.01)
    assert not improved(0.195, 0.2, "minimize", 0.01)


def test_route_after_execution() -> None:
    for status, route in [
        ("completed", "completed"),
        ("interrupted", "interrupted"),
        ("recoverable_error", "recoverable"),
        ("fatal_error", "fatal"),
    ]:
        assert route_after_execution({"execution_status": status}) == route  # type: ignore[arg-type]


def test_stop_priority(config_factory) -> None:
    state = build_initial_state(config_factory())
    state["stop_reason"] = "fatal_error"
    assert decide_stop(state) == (True, "fatal_error")
    state["stop_reason"] = None
    state["best_objective_value"] = 1.0
    state["target_value"] = 0.9
    state["completed_trial_count"] = 999
    assert decide_stop(state) == (True, "target_reached")
