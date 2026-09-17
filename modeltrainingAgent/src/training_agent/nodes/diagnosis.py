from __future__ import annotations

from typing import Any

from training_agent.state import TrainState, event, now_iso


def diagnose_trial(state: TrainState, services: Any) -> dict[str, Any]:
    try:
        diagnosis = services.diagnosis.diagnose(state) if services.diagnosis else None
        payload = diagnosis.model_dump() if diagnosis else None
        status = "completed"
    except Exception as exc:
        payload = {
            "problem": "diagnosis_failed",
            "confidence": 0.0,
            "explanation": str(exc),
            "suggested_actions": [],
            "suggested_search_space_changes": {},
        }
        status = "failed"
    return {
        "updated_at": now_iso(),
        "diagnosis": payload,
        "diagnosis_status": status,
        "events": [event("diagnose_trial", status, "diagnosis processed")],
    }
