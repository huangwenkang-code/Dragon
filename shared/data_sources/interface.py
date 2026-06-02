"""DataSource abstract base class and query/result types."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from shared.data_sources.constants import CACHE_TTL, RATE_LIMITS, QueryType


@dataclass
class RateLimit:
    max_per_second: int = 10
    burst: int = 20


@dataclass
class DataQuery:
    query_type: QueryType
    symbols: list[str] = field(default_factory=list)
    date: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataResult:
    source: str
    query_type: QueryType
    data: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None
    cached_at: float = 0.0
    error: str = ""

    @property
    def is_cached(self) -> bool:
        return self.cached_at > 0

    @property
    def is_error(self) -> bool:
        return bool(self.error)

    @property
    def count(self) -> int:
        return len(self.data)


class DataSource(ABC):
    """Abstract base for all data sources."""

    name: str = ""
    protocol: str = "http"

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection. Return True on success."""

    @abstractmethod
    async def query(self, query: DataQuery) -> DataResult:
        """Execute query and return standardized result."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Lightweight liveness check."""

    @abstractmethod
    def default_rate_limit(self) -> RateLimit:
        """Return rate limit for this source."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.name}) proto={self.protocol}>"
