"""Strategy registry — CRUD + DB persistence for trading strategies."""

from __future__ import annotations

from services.backtest.strategies import TradingStrategy, SYSTEM_STRATEGIES, STRATEGY_A, STRATEGY_B
from shared.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyRegistry:
    """In-memory + DB-backed strategy store.

    System strategies (A/B) are always present and cannot be deleted.
    Custom strategies are persisted to DB and loaded on startup.
    """

    def __init__(self):
        self._strategies: dict[str, TradingStrategy] = dict(SYSTEM_STRATEGIES)

    def list_all(self) -> list[TradingStrategy]:
        return list(self._strategies.values())

    def get(self, name: str) -> TradingStrategy | None:
        return self._strategies.get(name)

    def add(self, s: TradingStrategy):
        if s.name in self._strategies and self._strategies[s.name].is_system:
            raise ValueError(f"Cannot overwrite system strategy: {s.name}")
        self._strategies[s.name] = s

    def remove(self, name: str):
        s = self._strategies.get(name)
        if s and s.is_system:
            raise ValueError(f"Cannot delete system strategy: {name}")
        self._strategies.pop(name, None)

    async def save_to_db(self, session):
        """Persist all non-system strategies to backtest_strategies table."""
        from db.models import BacktestStrategy
        for s in self._strategies.values():
            if s.is_system:
                continue
            row = BacktestStrategy(
                name=s.name, description=s.description,
                config_json=s.to_dict(),
            )
            session.add(row)
        await session.flush()

    async def load_from_db(self, session):
        """Load custom strategies from DB, merge with system defaults."""
        from sqlalchemy import select
        from db.models import BacktestStrategy
        result = await session.execute(select(BacktestStrategy))
        for row in result.scalars().all():
            try:
                s = TradingStrategy.from_dict(row.config_json)
                s.is_system = False
                self._strategies[s.name] = s
            except Exception as e:
                logger.warning("Failed to load strategy %s: %s", row.name, e)


_registry: StrategyRegistry | None = None


def get_registry() -> StrategyRegistry:
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry
