"""Backtest data models — Position, Trade, Order, BacktestResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


def calc_commission(amount: float, rate: float, min_fee: float = 5.0) -> float:
    """Calculate commission with minimum fee floor.
    A-share: 0.025% rate, min ¥5 per trade.
    """
    return max(amount * rate, min_fee)


@dataclass
class Position:
    """A single holding during backtest."""
    stock_code: str
    stock_name: str
    entry_date: date
    entry_price: float
    entry_score: float           # leader_score at entry
    shares: int = 0
    cost: float = 0.0            # total cost (entry_price * shares)
    peak_score: float = 0.0      # highest leader_score since entry
    peak_price: float = 0.0      # highest price since entry
    days_held: int = 0
    score_history: list[float] = field(default_factory=list)  # daily scores since entry
    current_price: float = 0.0
    current_score: float = 0.0


@dataclass
class Trade:
    """A completed round-trip trade."""
    stock_code: str
    stock_name: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: int
    cost: float
    proceeds: float
    pnl: float                   # gross profit/loss (before fees)
    pnl_pct: float               # gross percentage return
    entry_commission: float = 0.0   # buy commission
    exit_commission: float = 0.0    # sell commission
    stamp_duty: float = 0.0        # sell stamp duty
    net_pnl: float = 0.0           # net P&L after all fees
    entry_score: float = 0.0       # leader_score at entry
    exit_score: float = 0.0        # leader_score at exit
    exit_reason: str = ""
    holding_days: int = 0
    cash_after_trade: float = 0.0  # available cash right after this trade closed


@dataclass
class Order:
    """A buy instruction produced by PositionAllocator."""
    stock_code: str
    stock_name: str
    score: float
    allocated_cash: float
    shares: int                  # rounded down to 100-share lots
    entry_price: float


@dataclass
class DailySnapshot:
    """Portfolio state at end of a trading day."""
    date: date
    equity: float                # total portfolio value
    cash: float                  # unused cash
    positions: list[Position]
    daily_return: float = 0.0    # day-over-day return
    regime: str = "?"            # BULL / CHOPPY / BEAR


@dataclass
class BacktestResult:
    """Full backtest output."""
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    total_commission: float = 0.0    # sum of all commissions
    total_stamp_duty: float = 0.0   # sum of all stamp duties
    trades: list[Trade] = field(default_factory=list)
    daily_snapshots: list[DailySnapshot] = field(default_factory=list)
    benchmark_return_pct: float = 0.0
