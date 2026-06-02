"""Deep analysis of backtest trades."""
import json, sys
from collections import defaultdict, Counter

with open(sys.argv[1], encoding='utf-8') as f:
    data = json.load(f)

trades = data['trades']

def p(s=""): print(s)

p("=" * 80)
p(f"BACKTEST ANALYSIS | Return={data['total_return_pct']:.2f}% WR={data['win_rate']*100:.1f}% Trades={data['total_trades']} DD={data['max_drawdown_pct']:.2f}% Sharpe={data['sharpe_ratio']:.3f}")

# ═════════════ 1. EXIT REASON ═════════════
p(); p("--- 1. Exit Reason Breakdown ---")
exits = defaultdict(lambda: {"cnt":0, "win":0, "pnl":0.0, "days":[]})
for t in trades:
    er = t['exit_reason']
    exits[er]['cnt'] += 1
    exits[er]['pnl'] += t['net_pnl']
    exits[er]['days'].append(t['holding_days'])
    if t['net_pnl'] > 0:
        exits[er]['win'] += 1

print(f"{'Exit Reason':<35} {'N':>4} {'WR':>6} {'PnL':>12} {'AvgPnL':>10} {'AvgD':>5}")
print("-" * 74)
for k, v in sorted(exits.items(), key=lambda x: x[1]['pnl'], reverse=True):
    wr = v['win']/v['cnt']*100 if v['cnt'] else 0
    avg_pnl = v['pnl']/v['cnt']
    avg_day = sum(v['days'])/len(v['days'])
    print(f"{k:<35} {v['cnt']:>4} {wr:>5.1f}% {v['pnl']:>+12,.0f} {avg_pnl:>+10,.0f} {avg_day:>4.1f}d")

# ═════════════ 2. HOLDING DAYS ═════════════
p(); p("--- 2. Holding Days vs Win Rate ---")
buckets = [(1,1), (2,3), (4,7), (8,15), (16,99)]
print(f"{'Days':<10} {'N':>5} {'%':>5} {'WR':>6} {'PnL':>12} {'AvgPnL':>10}")
print("-" * 50)
for lo, hi in buckets:
    group = [t for t in trades if lo <= t['holding_days'] <= hi]
    if not group: continue
    wins = sum(1 for t in group if t['net_pnl'] > 0)
    pnl = sum(t['net_pnl'] for t in group)
    label = f"{lo}d" if lo==hi else f"{lo}-{hi}d"
    print(f"{label:<10} {len(group):>5} {len(group)/len(trades)*100:>4.1f}% {wins/len(group)*100:>5.1f}% {pnl:>+12,.0f} {pnl/len(group):>+10,.0f}")

# ═════════════ 3. SCORE BANDS ═════════════
p(); p("--- 3. Entry Score Bands ---")
score_bands = [(0.50, 0.52), (0.52, 0.54), (0.54, 0.56), (0.56, 0.58), (0.58, 0.60), (0.60, 0.65)]
print(f"{'Score':<10} {'N':>5} {'WR':>6} {'PnL':>12} {'AvgPnL':>10} {'AvgD':>5}")
print("-" * 50)
for lo, hi in score_bands:
    group = [t for t in trades if lo <= t['entry_score'] < hi]
    if not group: continue
    wins = sum(1 for t in group if t['net_pnl'] > 0)
    pnl = sum(t['net_pnl'] for t in group)
    avg_d = sum(t['holding_days'] for t in group) / len(group)
    print(f"{lo:.2f}-{hi:.2f}  {len(group):>5} {wins/len(group)*100:>5.1f}% {pnl:>+12,.0f} {pnl/len(group):>+10,.0f} {avg_d:>4.1f}d")

# ═════════════ 4. HARD STOP DEEP DIVE ═════════════
p(); p("--- 4. Hard Stop (-5%) Deep Dive ---")
hard_stops = [t for t in trades if 'hard' in t.get('exit_reason','').lower()
              or 'HardStop' in t.get('exit_reason','')
              or '硬止损' in t.get('exit_reason','')
              or '止损' in t.get('exit_reason','')]

# If keyword match fails, take all negative exit reasons with PnL < -5000
if not hard_stops:
    hard_stops = [t for t in trades if t['net_pnl'] < -5000 and t['holding_days'] <= 3]

if hard_stops:
    hs_pnl = sum(t['net_pnl'] for t in hard_stops)
    p(f"  Total hard stops: {len(hard_stops)} ({len(hard_stops)/len(trades)*100:.0f}% of trades)")
    p(f"  Total loss: {hs_pnl:,.0f}")
    p(f"  If eliminated entirely: +{abs(hs_pnl):,.0f} net profit")
    p(f"  If 50% reduced: +{abs(hs_pnl)*0.5:,.0f} net profit")
    p(f"  Avg holding: {sum(t['holding_days'] for t in hard_stops)/len(hard_stops):.1f}d")

    p(f"\n  By holding days:")
    hs_days = Counter(t['holding_days'] for t in hard_stops)
    for d in sorted(hs_days):
        pnl_d = sum(t['net_pnl'] for t in hard_stops if t['holding_days']==d)
        p(f"    {d}d: {hs_days[d]} trades, loss={pnl_d:,.0f}")

    p(f"\n  By entry score:")
    hs_score = defaultdict(lambda: {"cnt":0, "pnl":0.0})
    for t in hard_stops:
        b = f"{t['entry_score']:.2f}"
        hs_score[b]['cnt'] += 1
        hs_score[b]['pnl'] += t['net_pnl']
    for b in sorted(hs_score):
        p(f"    score={b}: {hs_score[b]['cnt']} trades, loss={hs_score[b]['pnl']:,.0f}")
else:
    p("  (No hard stops identified — showing all negative exit types)")
    for k, v in sorted(exits.items(), key=lambda x: x[1]['pnl']):
        if v['pnl'] < 0:
            p(f"    {k}: {v['cnt']}t, WR={v['win']/v['cnt']*100:.0f}%, PnL={v['pnl']:,.0f}")

# ═════════════ 5. WINNER vs LOSER ═════════════
p(); p("--- 5. Winner vs Loser Profile ---")
winners = [t for t in trades if t['net_pnl'] > 0]
losers = [t for t in trades if t['net_pnl'] <= 0]

for label, group in [("Winners", winners), ("Losers", losers)]:
    if not group: continue
    avg_s = sum(t['entry_score'] for t in group)/len(group)
    avg_d = sum(t['holding_days'] for t in group)/len(group)
    avg_pnl = sum(t['pnl_pct'] for t in group)/len(group)
    avg_exit = sum(t['exit_score'] for t in group)/len(group)
    drop = avg_s - avg_exit
    p(f"  {label} ({len(group)}): entry_score={avg_s:.3f} exit_score={avg_exit:.3f} drop={drop:.3f} days={avg_d:.1f} avg_pnl%={avg_pnl:+.2f}%")

# ═════════════ 6. MONTHLY ═════════════
p(); p("--- 6. Monthly Breakdown ---")
monthly = defaultdict(lambda: {"cnt":0, "win":0, "pnl":0.0})
for t in trades:
    m = t['entry_date'][:7]
    monthly[m]['cnt'] += 1
    monthly[m]['pnl'] += t['net_pnl']
    if t['net_pnl'] > 0:
        monthly[m]['win'] += 1

cum = 0
print(f"{'Month':<10} {'N':>5} {'WR':>6} {'PnL':>12} {'Cum':>12}")
print("-" * 47)
for m in sorted(monthly):
    v = monthly[m]
    wr = v['win']/v['cnt']*100
    cum += v['pnl']
    print(f"{m:<10} {v['cnt']:>5} {wr:>5.1f}% {v['pnl']:>+12,.0f} {cum:>+12,.0f}")

# ═════════════ 7. TOP/BOTTOM ═════════════
p(); p("--- 7. Top/Worst Trades ---")
sorted_t = sorted(trades, key=lambda t: t['pnl_pct'])
p("  WORST 5:")
for t in sorted_t[:5]:
    p(f"    {t['stock_code']} {t['stock_name']:<8} {t['entry_date']} score={t['entry_score']:.3f} "
      f"{t['holding_days']}d PnL={t['net_pnl']:+,.0f} ({t['pnl_pct']:+.1f}%) [{t['exit_reason']}]")
p("  BEST 5:")
for t in sorted_t[-5:]:
    p(f"    {t['stock_code']} {t['stock_name']:<8} {t['entry_date']} score={t['entry_score']:.3f} "
      f"{t['holding_days']}d PnL={t['net_pnl']:+,.0f} ({t['pnl_pct']:+.1f}%) [{t['exit_reason']}]")

# ═════════════ 8. RECOMMENDATIONS ═════════════
p(); p("=" * 80)
p("  8. DATA-DRIVEN RECOMMENDATIONS")
p("=" * 80)

# Hard stop analysis
total_hs_loss = sum(t['net_pnl'] for t in hard_stops) if hard_stops else 0
one_day = [t for t in trades if t['holding_days'] <= 1]
one_day_loss = sum(t['net_pnl'] for t in one_day)

p(f"\n  [P0 - CRITICAL] Replace fixed -5% hard stop with ATR trailing stop")
p(f"    Hard stops: {len(hard_stops)} trades, total loss: {total_hs_loss:,.0f}")
p(f"    {len([t for t in hard_stops if t['holding_days']<=1])} of these happen within 1 day (gap down overnight)")
p(f"    Solution: ATR-based stop (e.g., 2x ATR(14)) + no stop on entry day")
est = abs(total_hs_loss) * 0.5
p(f"    Estimated gain: +{est:,.0f} ({est/1000000*100:.1f}% return)")

low_s = [t for t in trades if t['entry_score'] < 0.55]
high_s = [t for t in trades if t['entry_score'] >= 0.55]
low_wr = sum(1 for t in low_s if t['net_pnl']>0)/len(low_s)*100 if low_s else 0
high_wr = sum(1 for t in high_s if t['net_pnl']>0)/len(high_s)*100 if high_s else 0
low_pnl = sum(t['net_pnl'] for t in low_s)
high_pnl = sum(t['net_pnl'] for t in high_s)

p(f"\n  [P1] Raise entry score threshold: 0.50 -> 0.55")
p(f"    0.50-0.55: {len(low_s)} trades, WR={low_wr:.0f}%, PnL={low_pnl:,.0f}")
p(f"    0.55+:     {len(high_s)} trades, WR={high_wr:.0f}%, PnL={high_pnl:,.0f}")

long_holds = [t for t in trades if t['holding_days'] >= 8]
lw = sum(1 for t in long_holds if t['net_pnl']>0)
p(f"\n  [P2] Aggressively extend winning holds")
p(f"    8+ day holds: {len(long_holds)} trades, WR={lw/len(long_holds)*100:.0f}%, PnL={sum(t['net_pnl'] for t in long_holds):,.0f}")
p(f"    Strategy: slower decay rate for profitable positions, remove time stop")

p(f"\n  [P3] Add overnight gap-down protection")
p(f"    {len([t for t in trades if t['holding_days']<=1 and t['net_pnl']<0])} trades exit day-1 at a loss")
p(f"    Filter: skip entry if prev close near limit-up (suggesting gap-down risk next day)")

# Print total potential
p(f"\n{'='*80}")
p(f"  SUMMARY: 26.91% is the baseline. P0+P1 alone could add 5-10% return.")
p(f"  Focus: FIX hard stop mechanism + RAISE entry bar slightly.")
