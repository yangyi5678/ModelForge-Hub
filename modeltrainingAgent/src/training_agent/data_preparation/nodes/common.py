from __future__ import annotations

from typing import Any

from training_agent.state import event, now_iso


def fail_state(node: str, exc: Exception) -> dict[str, Any]:
    return {
        "updated_at": now_iso(),
        "error_type": "invalid_data",
        "error_message": str(exc),
        "error_node": node,
        "should_stop": True,
        "events": [event(node, "failed", str(exc))],
    }

