"""TradeEpisode — 一只票从发现到退出的完整生命周期."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# State constants
STATE_DISCOVERED = "DISCOVERED"
STATE_HOLDING = "HOLDING"
STATE_ACCELERATING = "ACCELERATING"
STATE_DISTRIBUTING = "DISTRIBUTING"
STATE_DECAYING = "DECAYING"
STATE_EXITED = "EXITED"

VALID_STATES = {
    STATE_DISCOVERED, STATE_HOLDING, STATE_ACCELERATING,
    STATE_DISTRIBUTING, STATE_DECAYING, STATE_EXITED,
}

VALID_TRANSITIONS: dict[str, set[str]] = {
    STATE_DISCOVERED:    {STATE_HOLDING, STATE_EXITED},
    STATE_HOLDING:       {STATE_ACCELERATING, STATE_DECAYING, STATE_DISTRIBUTING, STATE_EXITED},
    STATE_ACCELERATING:  {STATE_HOLDING, STATE_DISTRIBUTING, STATE_DECAYING, STATE_EXITED},
    STATE_DISTRIBUTING:  {STATE_DECAYING, STATE_EXITED},
    STATE_DECAYING:      {STATE_HOLDING, STATE_EXITED},
    STATE_EXITED:        set(),
}


@dataclass
class EpisodeRecord:
    """每日一条记录 — 替代旧 Position.score_history."""
    date: date
    state: str
    price: float
    full_score: float | None = None
    continuation_score: float = 0.0
    decay_score: float = 0.0
    volume: float = 0.0
    pnl_pct: float = 0.0
    is_in_candidate_pool: bool = False


@dataclass
class TradeEpisode:
    """一只票从发现到退出的完整生命周期."""
    episode_id: str
    stock_code: str
    stock_name: str
    state: str
    entry_date: date
    entry_price: float
    entry_score: float
    shares: int = 0
    cost: float = 0.0
    total_commission: float = 0.0

    daily_records: list[EpisodeRecord] = field(default_factory=list)
    peak_price: float = 0.0
    peak_score: float = 0.0
    max_floating_pnl: float = 0.0
    max_drawdown: float = 0.0

    exit_reason: str | None = None
    exit_date: date | None = None
    entry_reason: list[str] = field(default_factory=list)

    def add_record(self, record: EpisodeRecord):
        self.daily_records.append(record)
        if record.price > self.peak_price:
            self.peak_price = record.price
        if record.continuation_score > self.peak_score:
            self.peak_score = record.continuation_score
        if self.entry_price > 0:
            floating = (record.price - self.entry_price) / self.entry_price
            if floating > self.max_floating_pnl:
                self.max_floating_pnl = floating
        dd = (self.peak_price - record.price) / self.peak_price if self.peak_price > 0 else 0.0
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def transition_to(self, new_state: str) -> bool:
        if new_state in VALID_TRANSITIONS.get(self.state, set()):
            self.state = new_state
            return True
        return False


# ── State machine ──

def run_state_machine(
    episode: TradeEpisode,
    regime: str,
    lookback_bars: list[dict],
    decay_trigger_days: int = 2,
) -> str:
    """评估 episode 当前状态，返回新状态（可能不变）."""
    current = episode.state
    if current == STATE_EXITED:
        return STATE_EXITED

    if len(episode.daily_records) < 2:
        return current

    recent = episode.daily_records[-2:]
    cont_now = recent[-1].continuation_score
    decay_now = recent[-1].decay_score
    cont_prev = recent[0].continuation_score
    decay_prev = recent[0].decay_score

    if current == STATE_HOLDING:
        if decay_now > cont_now:
            if decay_trigger_days == 1:
                return STATE_DECAYING
            elif decay_prev > cont_prev:
                return STATE_DECAYING
        if _is_accelerating(episode, lookback_bars):
            return STATE_ACCELERATING

    elif current == STATE_ACCELERATING:
        if len(lookback_bars) >= 2:
            today_close = lookback_bars[-1].get("close") or 0
            prev_close = lookback_bars[-2].get("close") or 0
            if prev_close > 0:
                day_change = (today_close - prev_close) / prev_close
                if day_change < -0.05:
                    return STATE_DISTRIBUTING
        if len(lookback_bars) >= 1:
            bar = lookback_bars[-1]
            open_p = bar.get("open") or 0
            close_p = bar.get("close") or 0
            if open_p > 0 and (open_p - close_p) / open_p > 0.03:
                return STATE_DISTRIBUTING

    elif current == STATE_DISTRIBUTING:
        if decay_now > cont_now:
            if decay_trigger_days == 1:
                return STATE_DECAYING
            elif decay_prev > cont_prev:
                return STATE_DECAYING

    elif current == STATE_DECAYING:
        if cont_now > decay_now and _price_new_high(lookback_bars):
            return STATE_HOLDING

    return current


def _is_accelerating(episode: TradeEpisode, lookback_bars: list[dict]) -> bool:
    records = episode.daily_records[-3:]
    bars = lookback_bars[-3:] if len(lookback_bars) >= 3 else lookback_bars
    if len(records) < 3 or len(bars) < 3:
        return False
    prices = [(r.price or 0) for r in records]
    if not (prices[0] < prices[1] < prices[2]):
        return False
    avg_vol = sum(b.get("volume") or 0 for b in bars) / len(bars) if bars else 0
    today_vol = bars[-1].get("volume") or 0
    if today_vol <= 0 or avg_vol <= 0:
        return False
    return today_vol > avg_vol


def _price_new_high(lookback_bars: list[dict]) -> bool:
    if len(lookback_bars) < 5:
        return True
    today = lookback_bars[-1].get("close") or 0
    prev = [b.get("close") or 0 for b in lookback_bars[-6:-1]]
    if not prev or today <= 0:
        return False
    return today > max(prev)
