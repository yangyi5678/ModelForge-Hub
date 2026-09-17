from __future__ import annotations

from training_agent.graph import build_graph, build_services
from training_agent.state import build_initial_state


def test_same_thread_id_can_rerun_without_second_study(config_factory) -> None:
    cfg = config_factory(task_id="recoverable", max_trials=1, target_value=0.999)
    services = build_services(cfg)
    graph = build_graph(services)
    thread = {"configurable": {"thread_id": cfg.task.task_id}, "recursion_limit": 200}
    first = graph.invoke(build_initial_state(cfg), config=thread)
    second = graph.invoke(build_initial_state(cfg), config=thread)
    assert first["workflow_status"] == "completed"
    assert second["workflow_status"] == "completed"
    assert len(services.optuna.summaries()) == 1
