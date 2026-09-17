from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any


def load_python_entrypoint(uri: str) -> Callable[..., Any]:
    """Load a callable from a python://module.path:function URI."""

    prefix = "python://"
    if not uri.startswith(prefix):
        raise ValueError("entrypoint URI must start with python://")
    target = uri.removeprefix(prefix)
    module_name, separator, function_name = target.partition(":")
    if not module_name or separator != ":" or not function_name:
        raise ValueError("entrypoint URI must use python://module:function format")
    module = import_module(module_name)
    entrypoint = getattr(module, function_name, None)
    if not callable(entrypoint):
        raise ValueError(f"entrypoint is not callable: {uri}")
    return entrypoint
