"""BacktestEngine — iterate daily PipelineRuns, simulate portfolio with pluggable rules.

V3: Index MA20 gate + quality factor filtering + ATR trailing stop.
"""

from __future__ import annotations

from datetime import date
from services.backtest.models import (
    Position, Trade, DailySnapshot, BacktestResult, calc_commission
)
from services.backtest.strategies import TradingStrategy
from services.backtest.episode import (
    TradeEpisode, EpisodeRecord,
    STATE_HOLDING, STATE_DECAYING, STATE_EXITED,
    run_state_machine,
    _is_accelerating,
)
from services.backtest.holding_scorer import compute_continuation, compute_decay
from services.backtest.market_regime import RegimeParams, REGIME_NORMAL
from shared.utils.logging import get_logger
from shared.utils.trading_day import is_trading_day

logger = get_logger(__name__)

# Config cache (refreshed on startup and via /admin/config/reload)
_CONFIG_CACHE: dict = {}

def _get_regime_multipliers() -> dict[str, float]:
    """Sync-safe regime multiplier lookup."""
    if not _CONFIG_CACHE:
        from shared.config_loader import _cache
        if _cache:
            _CONFIG_CACHE.update(_cache)
    if _CONFIG_CACHE:
        return {r: cfg.get("capital_multiplier", 0.85) for r, cfg in _CONFIG_CACHE.items()}
    return {"BULL": 1.0, "CHOPPY_UP": 0.90, "CHOPPY": 0.85, "CHOPPY_DOWN": 0.50, "BEAR": 0.50}

def _get_time_stop_days() -> int:
    if _CONFIG_CACHE:
        for cfg in _CONFIG_CACHE.values():
            return cfg.get("time_stop_days", 40)
    return 40

# Default params since regime detection is replaced by index MA20 gate
DEFAULT_PARAMS = RegimeParams(
    cont_weight=0.5, decay_weight=0.5, price_stop_pct=0.06,
    decay_trigger_days=3, gap_up_pct=0.04, time_stop_days=_get_time_stop_days(),
    min_cont_threshold=0.25, position_size_pct=0.20,
)


def _compute_atr(bars: list[dict], period: int = 14) -> float:
    """Compute Average True Range from OHLCV bars. Returns 0 if insufficient data."""
    if len(bars) < period + 1:
        return 0.0
    tr_list = []
    recent = bars[-(period + 1):]
    for i in range(1, len(recent)):
        h = recent[i].get("high", 0)
        l = recent[i].get("low", 0)
        prev_c = recent[i - 1].get("close", 0)
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr_list.append(tr)
    return sum(tr_list) / len(tr_list) if tr_list else 0.0


class BacktestContext:
    """Mutable state during a backtest run."""

    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
        self.current_date: date | None = None
        self.episodes: dict[str, TradeEpisode] = {}
        self.trades: list[Trade] = []
        self.snapshots: list[DailySnapshot] = []
        self.cash: float = strategy.initial_capital
        self.benchmark_value: float = 1.0
        self.episode_counter: int = 0
        # Index MA20 regime: BULL / CHOPPY / BEAR
        self.index_regime: str = "CHOPPY"
        self._regime_days: dict[str, int] = {"BULL": 0, "CHOPPY": 0, "BEAR": 0}


class BacktestEngine:
    """Iterate daily PipelineRun data, simulate trading with pluggable rules.

    Each day:
    1. Update prices from stock_daily_bars
    2. Daily rescoring — continuation/decay for ALL held episodes
    3. State machine transition
    4. Exit decision matrix (price stop + state exit + volume decay + time stop)
    5. Entry filtering (EntryRule chain + limit-up)
    6. Position limit enforcement
    7. Capital allocation (per-stock equal weight)
    8. Record daily snapshot + episode records
    """

    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
        self.context = BacktestContext(strategy)

    def run(self, daily_runs: list[dict]) -> BacktestResult:
        if not daily_runs:
            return self._empty_result()

        runs_sorted = sorted(daily_runs, key=lambda r: r["trade_date"])
        start_date = runs_sorted[0]["trade_date"]
        end_date = runs_sorted[-1]["trade_date"]

        # Candidates from day T are actionable on day T+1 (pipeline runs after close).
        pending_candidates: list[dict] = []

        skipped_count = 0
        for i, run in enumerate(runs_sorted):
            trade_date_str = run["trade_date"]
            if not is_trading_day(trade_date_str):
                skipped_count += 1
                continue
            self._process_day(run, pending_candidates)
            pending_candidates = run.get("leader_candidates", [])

        if skipped_count:
            logger.info("Skipped %d non-trading days (weekends/holidays)", skipped_count)

        # Force-close all remaining episodes at end of backtest
        if self.context.episodes:
            # Find last actual trading day for force-close
            last_trading_run = None
            for run in reversed(runs_sorted):
                if is_trading_day(run["trade_date"]):
                    last_trading_run = run
                    break
            if last_trading_run is None:
                last_trading_run = runs_sorted[-1]
            last_date = last_trading_run["trade_date"]
            last_prices = last_trading_run.get("prices", {})
            for code, ep in list(self.context.episodes.items()):
                price = last_prices.get(code)
                if not price or price <= 0:
                    last_rec = ep.daily_records[-1] if ep.daily_records else None
                    price = last_rec.price if last_rec and (last_rec.price or 0) > 0 else ep.entry_price
                if ep.daily_records:
                    ep.daily_records[-1].price = price
                self._close_episode(code, ep, "回测结束强制平仓", last_date)

        return self._build_result(start_date, end_date)

    def _process_day(self, run: dict, pending_candidates: list[dict]):
        trade_date = run["trade_date"]
        candidates = run.get("leader_candidates", [])
        prices = run.get("prices", {})
        prev_close = run.get("prev_close", {})

        cand_by_code = {c.get("stock_code"): c for c in candidates}

        # ── Step 1: Update prices ──
        for code, ep in self.context.episodes.items():
            price = prices.get(code)
            if price and price > 0:
                if ep.daily_records:
                    ep.daily_records[-1].price = price
                # Update peak_price for price trailing stop
                if price > ep.peak_price:
                    ep.peak_price = price

        # ── Step 2: Daily rescoring ──
        ohlcv_data = run.get("bars", {})
        index_bars = run.get("index_bars", [])

        # Index MA20 regime: BULL / CHOPPY / BEAR
        self._update_index_regime(index_bars, trade_date)

        for code, ep in self.context.episodes.items():
            stock_bars = ohlcv_data.get(code, [])
            is_in_pool = code in cand_by_code

            full_score = None
            if is_in_pool:
                full_score = cand_by_code[code].get("leader_score") or 0

            cont = compute_continuation(stock_bars, index_bars, REGIME_NORMAL)
            decay = compute_decay(stock_bars, REGIME_NORMAL)

            last_price = (
                prices.get(code)
                or (ep.daily_records[-1].price if ep.daily_records else ep.entry_price)
            )

            rec = EpisodeRecord(
                date=trade_date,
                state=ep.state,
                price=last_price,
                full_score=full_score,
                continuation_score=cont,
                decay_score=decay,
                volume=stock_bars[-1].get("volume", 0) if stock_bars else 0,
                pnl_pct=(last_price - ep.entry_price) / ep.entry_price if ep.entry_price > 0 else 0.0,
                is_in_candidate_pool=is_in_pool,
            )
            ep.add_record(rec)

        # ── Step 3: State machine ──
        for code, ep in self.context.episodes.items():
            stock_bars = ohlcv_data.get(code, [])
            new_state = run_state_machine(ep, REGIME_NORMAL, stock_bars, DEFAULT_PARAMS.decay_trigger_days)
            if new_state != ep.state:
                old_state = ep.state
                ep.transition_to(new_state)
                if ep.daily_records:
                    ep.daily_records[-1].state = new_state
                logger.info("[%s] %s state: %s → %s", trade_date, code, old_state, new_state)

        # ── Step 4: Exit decision matrix ──
        for code, ep in list(self.context.episodes.items()):
            exit_reason: str | None = None

            # 4a: ATR-based hard stop (dynamic, min 3%, max 8%)
            current_price = prices.get(code, 0)
            if current_price <= 0 and ep.daily_records:
                current_price = ep.daily_records[-1].price
            stock_bars_exit = ohlcv_data.get(code, [])
            atr = _compute_atr(stock_bars_exit, 14) if stock_bars_exit else 0.0
            if ep.entry_price > 0 and current_price > 0:
                # ATR-based stop level: 1.5x ATR, bounded [3%, 8%]
                if atr > 0:
                    atr_pct = (1.5 * atr) / ep.entry_price
                    stop_pct = max(0.03, min(0.08, atr_pct))
                else:
                    stop_pct = 0.05
                entry_dd = (current_price - ep.entry_price) / ep.entry_price
                # Day-1 grace: no hard stop on entry day (allow overnight gap breathing room)
                days_held = len(ep.daily_records)
                effective_stop = stop_pct if days_held > 1 else stop_pct * 1.5
                if entry_dd < -effective_stop:
                    exit_reason = f"硬止损({entry_dd:.1%}<-{effective_stop:.1%})"

            # 4b: ATR trailing stop (regime-adaptive)
            # Peak trail: 2.5x ATR, Entry floor: 2.0x ATR (wider to let winners run)
            current_price = prices.get(code, 0)
            if current_price <= 0 and ep.daily_records:
                current_price = ep.daily_records[-1].price
            stock_bars_exit = ohlcv_data.get(code, [])
            atr = _compute_atr(stock_bars_exit, 14)
            if not exit_reason and atr > 0 and ep.peak_price > 0 and current_price > 0:
                trail_stop = ep.peak_price - (2.5 * atr)
                floor_stop = ep.entry_price - (2.0 * atr)
                if current_price < trail_stop and current_price < floor_stop:
                    dd = (ep.peak_price - current_price) / ep.peak_price
                    exit_reason = f"ATR止损({dd:.1%} dd, ATR={atr:.2f})"
            elif not exit_reason and ep.peak_price > 0 and current_price > 0:
                dd = (ep.peak_price - current_price) / ep.peak_price
                if dd > DEFAULT_PARAMS.price_stop_pct:
                    exit_reason = f"价格止损({dd:.1%}>{DEFAULT_PARAMS.price_stop_pct:.0%})"

            # 4b: State exit — DECAYING sustained
            if not exit_reason and ep.state == STATE_DECAYING:
                recent = ep.daily_records[-DEFAULT_PARAMS.decay_trigger_days:]
                if len(recent) >= DEFAULT_PARAMS.decay_trigger_days:
                    if all((r.decay_score or 0) > (r.continuation_score or 0) for r in recent):
                        exit_reason = f"退潮状态持续{DEFAULT_PARAMS.decay_trigger_days}天"

            # 4c: Volume exhaustion
            if not exit_reason and ep.daily_records:
                last_rec = ep.daily_records[-1]
                if (last_rec.volume or 0) > 0:
                    recent_vols = [(r.volume or 0) for r in ep.daily_records[-6:] if (r.volume or 0) > 0]
                    if len(recent_vols) >= 6:
                        avg_vol = sum(recent_vols[:-1]) / (len(recent_vols) - 1)
                        if avg_vol > 0 and (last_rec.volume or 0) < avg_vol * 0.3:
                            exit_reason = "成交量枯竭"

            # 4d: Extreme decay — cont below threshold AND not in pool
            # Grace period: skip first 2 days (scorer needs multi-day data)
            if not exit_reason and len(ep.daily_records) >= 3:
                last_rec = ep.daily_records[-1]
                if ((last_rec.continuation_score or 0) < DEFAULT_PARAMS.min_cont_threshold
                        and not last_rec.is_in_candidate_pool):
                    exit_reason = f"续航极低({last_rec.continuation_score:.2f}<{DEFAULT_PARAMS.min_cont_threshold})"

            # 4e: Time stop — held too long without accelerating
            if not exit_reason and len(ep.daily_records) >= DEFAULT_PARAMS.time_stop_days:
                has_acc = any(r.state == "ACCELERATING" for r in ep.daily_records)
                if not has_acc:
                    exit_reason = f"时间止损({len(ep.daily_records)}天未加速)"

            if exit_reason:
                ep.exit_reason = exit_reason
                ep.exit_date = trade_date
                self._close_episode(code, ep, exit_reason, trade_date)

        # ── Step 5: Entry filtering ──
        ctx = {
            "date": trade_date,
            "prices": prices,
            "prev_close": prev_close,
            "prev_day_change": run.get("prev_day_change", {}),
            "sector_volume_pct": run.get("sector_volume_pct", {}),
            "avg_volume_20": run.get("avg_volume_20", {}),
            "turnover_pct": run.get("turnover_pct", {}),
            "today_volume": run.get("today_volume", {}),
        }

        regime = self.context.index_regime

        eligible = []
        for c in pending_candidates:
            code = c.get("stock_code", "")
            if code in self.context.episodes:
                continue
            if self.strategy.enable_limit_up_filter:
                if not self._check_limit_up(c, ctx):
                    continue
            passed = all(r.should_enter(c, ctx) for r in self.strategy.entry_rules)
            if passed:
                eligible.append(c)

        # ── Step 6: Index regime gate ──
        # BEAR: only stocks with LHB or DDE activity (smart money signal)
        if regime == "BEAR":
            eligible = [c for c in eligible
                        if (c.get("lhb_score") or 0) > 0 or (c.get("flow_score") or 0) > 0.1]

        # ── Step 7: Position limit enforcement ──
        current_count = len(self.context.episodes)
        slots = max(0, self.strategy.max_positions - current_count)
        if slots == 0:
            eligible = []
        elif slots < len(eligible):
            if run.get("score_sort_desc"):
                eligible.sort(key=lambda c: c.get("leader_score") or 0, reverse=True)
            else:
                eligible.sort(key=lambda c: abs((c.get("leader_score") or 0) - 0.58))
            eligible = eligible[:slots]

        # ── Step 8: Capital allocation ──
        if eligible:
            per_stock = self.strategy.initial_capital * DEFAULT_PARAMS.position_size_pct
            # Regime multiplier from config (with sync-safe defaults)
            _rmap = _get_regime_multipliers()
            regime_mult = _rmap.get(regime, 0.85)
            raw_cash = per_stock * len(eligible) * regime_mult
            # Score-weighted multiplier: higher score → more capital (0.50=0.7x, 0.58+=1.0x)
            weight_multipliers = []
            for c in eligible:
                s = c.get("leader_score") or 0.5
                if s >= 0.58:
                    weight_multipliers.append(1.0)
                else:
                    weight_multipliers.append(0.7 + 0.3 * (s - 0.50) / 0.08)
            avg_weight = sum(weight_multipliers) / len(weight_multipliers)
            alloc_cash = min(raw_cash * avg_weight, self.context.cash)
            orders = self.strategy.allocator.allocate(eligible, alloc_cash, prices, {"date": trade_date})
            for order in orders:
                self._open_episode(order, trade_date, prices)

        # ── Step 9: Record daily snapshot ──
        equity = self.context.cash + sum(
            (ep.daily_records[-1].price if ep.daily_records else ep.entry_price) * ep.shares
            for ep in self.context.episodes.values()
        )
        prev_equity = self.context.snapshots[-1].equity if self.context.snapshots else self.strategy.initial_capital
        daily_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0

        self.context.snapshots.append(DailySnapshot(
            date=trade_date,
            equity=equity,
            cash=self.context.cash,
            positions=[
                Position(
                    stock_code=ep.stock_code,
                    stock_name=ep.stock_name,
                    entry_date=ep.entry_date,
                    entry_price=ep.entry_price,
                    entry_score=ep.entry_score,
                    shares=ep.shares,
                    cost=ep.cost,
                    peak_score=ep.peak_score,
                    peak_price=ep.peak_price,
                    days_held=len(ep.daily_records),
                    score_history=[r.continuation_score for r in ep.daily_records],
                    current_price=ep.daily_records[-1].price if ep.daily_records else ep.entry_price,
                    current_score=ep.daily_records[-1].continuation_score if ep.daily_records else ep.entry_score,
                )
                for ep in self.context.episodes.values()
            ],
            daily_return=daily_return,
            regime=self.context.index_regime,
        ))
        self.context.current_date = trade_date

    def _update_index_regime(self, index_bars: list[dict], trade_date):
        """Classify market as BULL / CHOPPY / BEAR using index price vs MA20 + MA20 slope.

        BULL:   close > MA20 AND MA20 slope rising
        CHOPPY: close within MA20 ±2% OR MA20 slope flat
        BEAR:   close < MA20 AND MA20 slope falling

        Requires 2 consecutive days to change regime (prevents whipsaws).
        """
        if not index_bars or len(index_bars) < 26:
            return

        closes = [b.get("close") or 0 for b in index_bars if (b.get("close") or 0) > 0]
        if len(closes) < 26:
            return

        # MA20 up to yesterday
        ma20_today = sum(closes[-21:-1]) / 20
        # MA20 5 days ago
        ma20_5d_ago = sum(closes[-26:-6]) / 20
        today_close = closes[-1]

        # MA20 slope as % change over 5 days
        ma20_slope = (ma20_today - ma20_5d_ago) / ma20_5d_ago if ma20_5d_ago > 0 else 0
        price_vs_ma = (today_close - ma20_today) / ma20_today if ma20_today > 0 else 0

        # Determine raw regime
        if price_vs_ma > 0.02 and ma20_slope > 0.005:
            raw = "BULL"
        elif price_vs_ma < -0.02 and ma20_slope < -0.005:
            raw = "BEAR"
        else:
            raw = "CHOPPY"

        # 2-day confirmation
        for r in ["BULL", "CHOPPY", "BEAR"]:
            if r == raw:
                self.context._regime_days[r] += 1
            else:
                self.context._regime_days[r] = 0

        old_regime = self.context.index_regime
        if self.context._regime_days[raw] >= 2 and raw != old_regime:
            self.context.index_regime = raw
            logger.info("[%s] INDEX REGIME: %s → %s (price_vs_ma=%.1f%%, slope=%.2f%%)",
                        trade_date, old_regime, raw, price_vs_ma * 100, ma20_slope * 100)

    def _check_limit_up(self, candidate: dict, ctx: dict) -> bool:
        code = candidate.get("stock_code", "")
        prev_close = ctx.get("prev_close", {}).get(code)
        open_price = ctx.get("prices", {}).get(code)
        if not prev_close or not open_price or prev_close <= 0 or open_price <= 0:
            return True  # missing data → pass (can't confirm limit-up)
        gap = (open_price - prev_close) / prev_close
        return gap < 0.098

    def _open_episode(self, order: dict, trade_date: date, prices: dict):
        code = order["stock_code"]
        price = order["entry_price"]
        cost = order["allocated_cash"]
        entry_comm = calc_commission(cost, self.strategy.commission_rate, self.strategy.min_commission)

        self.context.episode_counter += 1
        date_str = str(trade_date).replace("-", "")
        ep = TradeEpisode(
            episode_id=f"EP{date_str}_{self.context.episode_counter:04d}",
            stock_code=code,
            stock_name=order.get("stock_name", ""),
            state=STATE_HOLDING,
            entry_date=trade_date,
            entry_price=price,
            entry_score=order.get("score", 0),
            shares=order["shares"],
            cost=cost,
            total_commission=entry_comm,
        )
        self.context.episodes[code] = ep
        self.context.cash -= (cost + entry_comm)

    def _close_episode(self, code: str, ep: TradeEpisode, reason: str, trade_date: date):
        sell_price = ep.daily_records[-1].price if ep.daily_records else ep.entry_price
        proceeds = sell_price * ep.shares
        pnl = proceeds - ep.cost
        pnl_pct = pnl / ep.cost if ep.cost > 0 else 0.0

        entry_comm = ep.total_commission  # accumulates initial + add commissions
        exit_comm = calc_commission(proceeds, self.strategy.commission_rate, self.strategy.min_commission)
        stamp = proceeds * self.strategy.stamp_duty_rate
        net_pnl = pnl - entry_comm - exit_comm - stamp

        exit_score = (
            ep.daily_records[-1].continuation_score if ep.daily_records else ep.entry_score
        )

        trade = Trade(
            stock_code=code,
            stock_name=ep.stock_name,
            entry_date=ep.entry_date,
            exit_date=trade_date,
            entry_price=ep.entry_price,
            exit_price=sell_price,
            shares=ep.shares,
            cost=ep.cost,
            proceeds=proceeds,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            entry_commission=round(entry_comm, 2),
            exit_commission=round(exit_comm, 2),
            stamp_duty=round(stamp, 2),
            net_pnl=round(net_pnl, 2),
            entry_score=ep.entry_score,
            exit_score=round(exit_score, 3),
            exit_reason=reason,
            holding_days=len(ep.daily_records),
        )
        self.context.trades.append(trade)
        self.context.cash += (proceeds - exit_comm - stamp)
        trade.cash_after_trade = round(self.context.cash, 2)
        del self.context.episodes[code]

    def _build_result(self, start_date: date, end_date: date) -> BacktestResult:
        final_equity = self.context.snapshots[-1].equity if self.context.snapshots else self.strategy.initial_capital
        total_return = (final_equity - self.strategy.initial_capital) / self.strategy.initial_capital
        max_dd = self._calc_max_drawdown()
        sharpe = self._calc_sharpe()
        wins = sum(1 for t in self.context.trades if t.net_pnl > 0)
        wr = wins / len(self.context.trades) if self.context.trades else 0.0

        total_comm = sum(t.entry_commission + t.exit_commission for t in self.context.trades)
        total_stamp = sum(t.stamp_duty for t in self.context.trades)

        return BacktestResult(
            strategy_name=self.strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.strategy.initial_capital,
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return * 100, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            win_rate=round(wr, 3),
            total_trades=len(self.context.trades),
            total_commission=round(total_comm, 2),
            total_stamp_duty=round(total_stamp, 2),
            trades=self.context.trades,
            daily_snapshots=self.context.snapshots,
        )

    def _calc_max_drawdown(self) -> float:
        peak = self.strategy.initial_capital
        max_dd = 0.0
        for snap in self.context.snapshots:
            if snap.equity > peak:
                peak = snap.equity
            dd = (peak - snap.equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _calc_sharpe(self) -> float:
        returns = [s.daily_return for s in self.context.snapshots]
        if len(returns) < 2:
            return 0.0
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1) if len(returns) > 1 else 1e-9
        std = var ** 0.5
        if std == 0:
            return 0.0
        return (mean_ret / std) * (252 ** 0.5)

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            strategy_name=self.strategy.name,
            start_date=date.today(),
            end_date=date.today(),
            initial_capital=self.strategy.initial_capital,
            final_equity=self.strategy.initial_capital,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
        )
