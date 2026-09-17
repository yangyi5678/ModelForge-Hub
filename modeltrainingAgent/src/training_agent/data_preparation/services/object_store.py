from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from training_agent.data_preparation.services.uri import parse_uri


class ObjectStoreClient:
    """Small object-store facade with OSS and local-file support."""

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = endpoint or os.getenv("OSS_ENDPOINT")

    def download_file(self, uri: str, local_path: str | Path) -> Path:
        parsed = parse_uri(uri)
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if parsed.scheme == "file":
            shutil.copyfile(parsed.key, destination)
            return destination
        if parsed.scheme == "oss":
            bucket = self._bucket(parsed.bucket)
            bucket.get_object_to_file(parsed.key, str(destination))
            return destination
        raise ValueError(f"unsupported URI scheme: {parsed.scheme}")

    def read_json(self, uri: str) -> dict[str, Any]:
        parsed = parse_uri(uri)
        if parsed.scheme == "file":
            with Path(parsed.key).open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        elif parsed.scheme == "oss":
            bucket = self._bucket(parsed.bucket)
            value = json.loads(bucket.get_object(parsed.key).read().decode("utf-8"))
        else:
            raise ValueError(f"unsupported URI scheme: {parsed.scheme}")
        if not isinstance(value, dict):
            raise ValueError(f"JSON object expected at {uri}")
        return value

    def _bucket(self, bucket_name: str | None) -> Any:
        if not bucket_name:
            raise ValueError("OSS bucket is required")
        if not self.endpoint:
            raise ValueError("OSS_ENDPOINT must be set for oss:// downloads")
        try:
            import oss2
        except ImportError as exc:
            raise RuntimeError("oss2 is required for oss:// downloads") from exc
        access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        if not access_key_id or not access_key_secret:
            raise ValueError("OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET must be set")
        auth = oss2.Auth(access_key_id, access_key_secret)
        return oss2.Bucket(auth, self.endpoint, bucket_name)

