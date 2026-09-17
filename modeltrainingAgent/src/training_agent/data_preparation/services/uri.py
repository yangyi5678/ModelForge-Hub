from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class ParsedURI:
    scheme: str
    bucket: str | None
    key: str


def parse_uri(uri: str) -> ParsedURI:
    parsed = urlparse(uri)
    if parsed.scheme == "oss":
        if not parsed.netloc or not parsed.path:
            raise ValueError(f"invalid OSS URI: {uri}")
        return ParsedURI("oss", parsed.netloc, parsed.path.lstrip("/"))
    if parsed.scheme == "file":
        return ParsedURI("file", None, parsed.path)
    if parsed.scheme == "":
        return ParsedURI("file", None, uri)
    raise ValueError(f"unsupported URI scheme: {parsed.scheme}")


def safe_local_name(uri: str) -> str:
    parsed = parse_uri(uri)
    name = Path(parsed.key).name
    if not name:
        raise ValueError(f"URI has no file name: {uri}")
    return name

