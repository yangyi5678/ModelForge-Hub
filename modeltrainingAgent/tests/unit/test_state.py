from __future__ import annotations

from training_agent.state import build_initial_state


def test_initial_state_is_complete(config_factory) -> None:
    state = build_initial_state(config_factory())
    assert state["schema_version"] == 1
    assert state["trial_number"] is None
    assert state["execution_status"] == "idle"
    assert state["events"]
