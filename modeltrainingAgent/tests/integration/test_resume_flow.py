from __future__ import annotations

from training_agent.graph import build_graph, build_services
from training_agent.state import build_initial_state


def test_interrupted_trial_resumes_same_trial(config_factory) -> None:
    cfg = config_factory(behavior="interrupted", max_trials=1, target_value=0.999)
    services = build_services(cfg)
    final = build_graph(services).invoke(build_initial_state(cfg), config={"recursion_limit": 200})
    assert final["workflow_status"] == "completed"
    trials = services.repository.list_trials(cfg.task.task_id)
    assert len(trials) == 1
    assert trials[0]["trial_number"] == 0
    assert services.repository.attempt_count(cfg.task.task_id, 0) == 2
