from __future__ import annotations

from training_agent.graph import build_graph, build_services
from training_agent.state import build_initial_state


def test_oom_repair_creates_new_attempt(config_factory) -> None:
    cfg = config_factory(behavior="oom", max_trials=1, target_value=0.999)
    services = build_services(cfg)
    final = build_graph(services).invoke(build_initial_state(cfg), config={"recursion_limit": 200})
    assert final["workflow_status"] == "completed"
    assert services.repository.attempt_count(cfg.task.task_id, 0) == 2
    trial = services.repository.list_trials(cfg.task.task_id)[0]
    assert trial["status"] == "completed"
