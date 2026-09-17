from __future__ import annotations

from training_agent.data_preparation.models import SnapshotResult
from training_agent.data_preparation.services.object_store import ObjectStoreClient


SUPPORTED_SCHEMA_VERSIONS = {1}


def resolve_snapshot(result_uri: str, object_store: ObjectStoreClient) -> SnapshotResult:
    payload = object_store.read_json(result_uri)
    payload.setdefault("result_uri", result_uri)
    snapshot = SnapshotResult.model_validate(payload)
    if snapshot.status != "completed":
        raise ValueError(f"dataset snapshot is not completed: {snapshot.status}")
    if snapshot.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported snapshot schema_version: {snapshot.schema_version}")
    return snapshot

