"""DataSource registry — register, lookup, list all sources."""
from __future__ import annotations

from shared.data_sources.interface import DataSource

_registry: dict[str, DataSource] = {}


def register(name: str, source: DataSource) -> None:
    _registry[name] = source


def get(name: str) -> DataSource:
    if name not in _registry:
        raise KeyError(f"DataSource '{name}' not registered. Available: {list(_registry.keys())}")
    return _registry[name]


def list_all() -> list[str]:
    return list(_registry.keys())


def is_registered(name: str) -> bool:
    return name in _registry


def unregister(name: str) -> None:
    _registry.pop(name, None)


def clear() -> None:
    _registry.clear()
