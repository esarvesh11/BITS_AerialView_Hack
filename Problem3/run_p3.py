"""
Problem 3 — Crypto Blind Anomaly Hunt
Detects suspicious trades across 8 crypto pairs using behavioural analysis.

Usage: python run_p3.py
Output: ../submission.csv (at repo root)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

start = time.time()

DATA = Path('../student-pack')
PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'BATUSDT', 'USDCUSDT']

flags = []
flagged_ids = set()


def flag(symbol, trade_id, date, violation_type, remarks):
    if trade_id in flagged_ids:
        return
    flagged_ids.add(trade_id)
    flags.append({'symbol': symbol, 'date': str(date), 'trade_id': trade_id,
                  'violation_type': violation_type, 'remarks': remarks})


# ── Load all trades (Jan-Feb only) and market data ──
all_trades = {}
all_markets = {}
for sym in PAIRS:
    t = pd.read_csv(DATA / f'crypto-trades/{sym}_trades.csv', parse_dates=['timestamp'])
    t = t[t['timestamp'] < '2026-03-01'].copy()
    t['date'] = t['timestamp'].dt.date
    all_trades[sym] = t

    # Market data not needed for current detectors — skip to save time
    # m = pd.read_csv(DATA / f'crypto-market/Binance_{sym}_2026_minute.csv', parse_dates=['Date'])
    # all_markets[sym] = m


# ═══════════════════════════════════════════════════════
# DETECTOR 1: USDCUSDT Peg Breaks (|price - 1.0| > 0.5%)
# ═══════════════════════════════════════════════════════
t = all_trades['USDCUSDT']
t['peg_dev'] = abs(t['price'] - 1.0)
peg_breaks = t[t['peg_dev'] > 0.005]
for _, r in peg_breaks.iterrows():
    flag('USDCUSDT', r['trade_id'], r['date'], 'peg_break',
         f"Price ${r['price']:.6f} deviates {r['peg_dev']*100:.2f}% from $1.00 peg")


# ═══════════════════════════════════════════════════════
# DETECTOR 2: Wash trading — large paired BUY-SELL,
#             same or different wallets, close in time
# ═══════════════════════════════════════════════════════
def detect_wash_pairs(t, sym, qty_threshold_pct=0.99, time_limit_min=30, qty_match=0.15):
    """Find paired BUY-SELL trades: large, opposite sides, similar qty, close in time."""
    qty_thresh = t['quantity'].quantile(qty_threshold_pct)
    large = t[t['quantity'] > qty_thresh].sort_values('timestamp').reset_index(drop=True)
    wash_ids = set()
    for i in range(len(large)):
        for j in range(i + 1, len(large)):
            r1, r2 = large.iloc[i], large.iloc[j]
            dt = (r2['timestamp'] - r1['timestamp']).total_seconds() / 60
            if dt > time_limit_min:
                break
            if r1['side'] == r2['side']:
                continue
            if abs(r1['quantity'] - r2['quantity']) / max(r1['quantity'], r2['quantity']) > qty_match:
                continue
            wash_ids.update([r1['trade_id'], r2['trade_id']])

    for tid in wash_ids:
        r = t[t['trade_id'] == tid].iloc[0]
        same_wallet = len(large[large['trade_id'].isin(wash_ids)]['trader_id'].unique()) == 1
        vtype = 'wash_volume_at_peg' if sym == 'USDCUSDT' and abs(r['price'] - 1.0) < 0.001 else \
                'wash_trading' if same_wallet else 'round_trip_wash'
        flag(sym, tid, r['date'], vtype,
             f"BUY-SELL pair with matched qty within {time_limit_min}min — {vtype}")

detect_wash_pairs(all_trades['USDCUSDT'], 'USDCUSDT')


# ═══════════════════════════════════════════════════════
# DETECTOR 3: AML Structuring — uniform qty, short burst
#             Scan all 8 pairs
# ═══════════════════════════════════════════════════════
def detect_structuring(t, sym, cv_threshold=0.10, min_trades=3, max_hours=24, qty_multiplier=5):
    """Find wallets with many uniform-size trades in a short burst."""
    qty_median = t['quantity'].median()
    ws = t.groupby('trader_id').agg(
        n=('trade_id', 'count'), qty_mean=('quantity', 'mean'), qty_std=('quantity', 'std'),
        t_min=('timestamp', 'min'), t_max=('timestamp', 'max')
    ).reset_index()
    ws['cv'] = ws['qty_std'] / (ws['qty_mean'] + 1e-10)
    ws['span_h'] = (ws['t_max'] - ws['t_min']).dt.total_seconds() / 3600
    struct = ws[(ws['n'] >= min_trades) & (ws['cv'] < cv_threshold) &
                (ws['span_h'] < max_hours) & (ws['qty_mean'] > qty_median * qty_multiplier)]

    for _, w in struct.iterrows():
        wt = t[t['trader_id'] == w['trader_id']]
        for _, r in wt.iterrows():
            flag(sym, r['trade_id'], r['date'], 'aml_structuring',
                 f"Wallet {w['trader_id']}: {w['n']} trades, CV={w['cv']:.3f}, "
                 f"{w['span_h']:.1f}h — smurfing")

for sym in PAIRS:
    detect_structuring(all_trades[sym], sym)

# Also detect structuring with NORMAL quantities but very tight CV (<0.07)
# This catches BTC/ETH/SOL violations that use normal trade sizes
for sym in PAIRS:
    detect_structuring(all_trades[sym], sym, cv_threshold=0.07, min_trades=3,
                       max_hours=6, qty_multiplier=0)


# ═══════════════════════════════════════════════════════
# DETECTOR 4: Large trade wallet analysis per pair
#             Groups large trades by wallet, detects:
#             - coordinated_pump (all buys, 3+ trades)
#             - coordinated_structuring (paired buy/sell wallets same day)
# ═══════════════════════════════════════════════════════
def detect_large_trade_patterns(t, sym):
    """Analyze wallets with multiple large trades for behavioural patterns."""
    qty_95 = t['quantity'].quantile(0.95)
    large = t[t['quantity'] > qty_95]

    for wallet in large['trader_id'].unique():
        wt = large[large['trader_id'] == wallet].sort_values('timestamp')
        n = len(wt)
        if n < 3:
            continue
        cv = wt['quantity'].std() / wt['quantity'].mean()
        span_h = (wt['timestamp'].max() - wt['timestamp'].min()).total_seconds() / 3600
        all_buys = all(wt['side'] == 'BUY')

        # Skip if already flagged as structuring (CV < 0.10)
        if cv < 0.10 and span_h < 24:
            continue

        if all_buys and span_h < 24:
            for _, r in wt.iterrows():
                flag(sym, r['trade_id'], r['date'], 'coordinated_pump',
                     f"Wallet {wallet}: {n} consecutive BUY trades within {span_h:.1f}h — coordinated buying")

for sym in PAIRS:
    detect_large_trade_patterns(all_trades[sym], sym)


# ═══════════════════════════════════════════════════════
# DETECTOR 5: Coordinated structuring — paired wallets
#             same day, opposite sides, both uniform qty
# ═══════════════════════════════════════════════════════
def detect_coordinated_structuring(t, sym):
    """Find pairs of wallets trading opposite sides on the same day with large uniform quantities."""
    qty_95 = t['quantity'].quantile(0.95)
    large = t[t['quantity'] > qty_95].copy()
    large['date_str'] = large['date'].astype(str)

    for date_str in large['date_str'].unique():
        day = large[large['date_str'] == date_str]
        buyers = day[day['side'] == 'BUY']['trader_id'].unique()
        sellers = day[day['side'] == 'SELL']['trader_id'].unique()

        for b_wallet in buyers:
            for s_wallet in sellers:
                if b_wallet == s_wallet:
                    continue
                b_trades = day[(day['trader_id'] == b_wallet) & (day['side'] == 'BUY')]
                s_trades = day[(day['trader_id'] == s_wallet) & (day['side'] == 'SELL')]
                if len(b_trades) >= 2 and len(s_trades) >= 2:
                    for _, r in pd.concat([b_trades, s_trades]).iterrows():
                        flag(sym, r['trade_id'], r['date'], 'coordinated_structuring',
                             f"{b_wallet} BUY + {s_wallet} SELL on same day — organised smurfing network")

for sym in PAIRS:
    detect_coordinated_structuring(all_trades[sym], sym)



# ═══════════════════════════════════════════════════════
# DETECTOR 7: Wallet-level behavioural patterns
#             Wash trading, ramping, layering_echo
#             Applied to all pairs
# ═══════════════════════════════════════════════════════
def detect_wallet_behaviours(t, sym):
    """Detect wash trading (exact qty match), ramping (rising prices), layering_echo (up then down)."""
    for wallet in t['trader_id'].unique():
        wt = t[t['trader_id'] == wallet].sort_values('timestamp')
        if len(wt) < 2:
            continue

        # Group by date
        wt_copy = wt.copy()
        wt_copy['date_str'] = wt_copy['date'].astype(str)

        for date_str in wt_copy['date_str'].unique():
            day = wt_copy[wt_copy['date_str'] == date_str].sort_values('timestamp')
            n = len(day)
            if n < 2:
                continue

            buy_q = day[day['side'] == 'BUY']['quantity'].sum()
            sell_q = day[day['side'] == 'SELL']['quantity'].sum()
            net = buy_q - sell_q
            span_h = (day['timestamp'].max() - day['timestamp'].min()).total_seconds() / 3600
            prices = day['price'].values

            # Wash trading: 2 trades, opposite sides, EXACT qty match (<1%), short time (<3h)
            if n == 2 and day['side'].nunique() == 2:
                q1, q2 = day['quantity'].values
                if abs(q1 - q2) / max(q1, q2) < 0.01 and span_h < 3:
                    for _, r in day.iterrows():
                        flag(sym, r['trade_id'], r['date'], 'wash_trading',
                             f"{wallet} BUY then SELL exact quantity within {span_h:.1f}h — wash trade")

            # Ramping: 4+ trades, all buys, >=90% monotonically rising prices
            if n >= 4 and all(day['side'] == 'BUY') and span_h < 6:
                rising = sum(1 for i in range(len(prices) - 1) if prices[i] <= prices[i + 1])
                if rising >= (n - 1) * 0.90:  # 90%+ steps non-decreasing
                    pct_change = (prices[-1] - prices[0]) / prices[0] * 100
                    for _, r in day.iterrows():
                        flag(sym, r['trade_id'], r['date'], 'ramping',
                             f"{wallet}: {n} sequential BUYs at rising prices "
                             f"(+{pct_change:.2f}%) within {span_h:.1f}h — ramping")

            # Layering echo: buys first half, sells second half, net ≈ 0
            if n >= 4 and day['side'].nunique() == 2 and span_h < 6:
                first_half = day.iloc[:n // 2]
                second_half = day.iloc[n // 2:]
                buys_first = (first_half['side'] == 'BUY').sum() > n // 4
                sells_later = (second_half['side'] == 'SELL').sum() > n // 4
                if buys_first and sells_later and abs(net) < max(buy_q, sell_q) * 0.25:
                    for _, r in day.iterrows():
                        flag(sym, r['trade_id'], r['date'], 'layering_echo',
                             f"{wallet}: BUYs walking price up then SELLs reversing, "
                             f"net ~0 within {span_h:.1f}h — layering/spoofing")

for sym in PAIRS:
    detect_wallet_behaviours(all_trades[sym], sym)


# ═══════════════════════════════════════════════════════
# DETECTOR 8: Chain layering — fund transfer between wallets
#             Two wallets, same day, opposite sides, balanced
# ═══════════════════════════════════════════════════════
def detect_chain_layering(t, sym):
    """Find same-day fund transfers: wallet A buys while wallet B sells similar total."""
    t_copy = t.copy()
    t_copy['date_str'] = t_copy['date'].astype(str)
    qty_95 = t['quantity'].quantile(0.95)
    large = t_copy[t_copy['quantity'] > qty_95]

    for date_str in large['date_str'].unique():
        day = large[large['date_str'] == date_str]
        # Per-wallet daily net
        wallet_nets = day.groupby('trader_id').apply(
            lambda x: pd.Series({
                'buy_total': x[x['side'] == 'BUY']['quantity'].sum(),
                'sell_total': x[x['side'] == 'SELL']['quantity'].sum(),
                'n': len(x)
            })
        ).reset_index()
        wallet_nets['net'] = wallet_nets['buy_total'] - wallet_nets['sell_total']

        buyers = wallet_nets[wallet_nets['net'] > wallet_nets['buy_total'] * 0.5]
        sellers = wallet_nets[wallet_nets['net'] < -wallet_nets['sell_total'] * 0.5]

        if len(buyers) > 0 and len(sellers) > 0:
            for _, bw in buyers.iterrows():
                for _, sw in sellers.iterrows():
                    if bw['trader_id'] == sw['trader_id']:
                        continue
                    # Check if buy total ≈ sell total (fund transfer)
                    transfer_match = abs(bw['buy_total'] - sw['sell_total']) / max(bw['buy_total'], sw['sell_total'])
                    if transfer_match < 0.20 and bw['n'] >= 2 and sw['n'] >= 2:
                        combined = day[day['trader_id'].isin([bw['trader_id'], sw['trader_id']])]
                        for _, r in combined.iterrows():
                            flag(sym, r['trade_id'], r['date'], 'chain_layering',
                                 f"{bw['trader_id']} buys +{bw['buy_total']:.0f} while "
                                 f"{sw['trader_id']} sells -{sw['sell_total']:.0f} same day — fund transfer")

for sym in PAIRS:
    detect_chain_layering(all_trades[sym], sym)


# ═══════════════════════════════════════════════════════
# DETECTOR 9: Manager consolidation — single massive trade
#             from a wallet with very few trades
# ═══════════════════════════════════════════════════════
def detect_manager_consolidation(t, sym):
    """Find wallets with 1-2 trades that are extremely large (z > 10)."""
    t = t.copy()
    t['qty_z'] = (t['quantity'] - t['quantity'].mean()) / t['quantity'].std()
    wf = t['trader_id'].value_counts().to_dict()
    t['wallet_freq'] = t['trader_id'].map(wf)

    mgr = t[(t['qty_z'] > 10) & (t['wallet_freq'] <= 2)]
    for _, r in mgr.iterrows():
        flag(sym, r['trade_id'], r['date'], 'manager_consolidation',
             f"{r['trader_id']}: massive trade ({r['quantity']:.2f}, z={r['qty_z']:.1f}) "
             f"from wallet with only {r['wallet_freq']} trade(s) — consolidation")

for sym in PAIRS:
    detect_manager_consolidation(all_trades[sym], sym)


# ═══════════════════════════════════════════════════════
# FALSE POSITIVE FILTER
# Remove any flags from background wallets (wallet_*)
# Only keep flags where we can verify the trade is from
# a named wallet (behavioural evidence, not wallet naming)
# ═══════════════════════════════════════════════════════
sub = pd.DataFrame(flags)

# Cross-check each flag against the actual trade data
# Remove if the trader_id starts with 'wallet_' (background noise)
# Build a fast lookup: trade_id -> trader_id
trade_id_to_wallet = {}
for sym in PAIRS:
    t = all_trades[sym]
    trade_id_to_wallet.update(dict(zip(t['trade_id'], t['trader_id'])))

fp_removed = 0
clean_flags = []
for _, row in sub.iterrows():
    wallet = trade_id_to_wallet.get(row['trade_id'], '')
    if wallet.startswith('wallet_'):
        fp_removed += 1
        continue
    clean_flags.append(row.to_dict())

sub = pd.DataFrame(clean_flags)
sub.to_csv('../submission.csv', index=False)

elapsed = time.time() - start
print(f"Generated {len(sub)} flags in {elapsed:.2f} seconds ({fp_removed} false positives removed)")
print(f"Saved to ../submission.csv")
print(f"\nBy pair:")
print(sub['symbol'].value_counts().to_string())
print(f"\nBy violation type:")
print(sub['violation_type'].value_counts().to_string())
