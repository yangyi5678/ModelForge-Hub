from __future__ import annotations

from training_agent.graph import build_graph, build_services
from training_agent.state import build_initial_state


def test_remote_fake_submit_once(config_factory) -> None:
    cfg = config_factory(mode="remote", max_trials=1, target_value=0.999)
    services = build_services(cfg)
    final = build_graph(services).invoke(build_initial_state(cfg), config={"recursion_limit": 200})
    assert final["workflow_status"] == "completed"
    assert services.remote is not None
    assert len(services.remote.submissions) == 1
