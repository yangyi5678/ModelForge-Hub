from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_params_hash(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def effective_batch_size(params: dict[str, Any], runtime_params: dict[str, Any]) -> int:
    batch_size = int(params.get("batch_size", 1))
    micro = int(runtime_params.get("micro_batch_size", batch_size))
    grad_accum = int(runtime_params.get("gradient_accumulation_steps", max(1, batch_size // micro)))
    return micro * grad_accum
