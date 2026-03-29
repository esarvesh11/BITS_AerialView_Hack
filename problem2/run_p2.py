"""
Problem 2 — Insider / pre-announcement signal (equity).

Implements: SEC submissions 8-K, efts.sec.gov 8-K search (+ merger/acquisition pass),
15-day volume/return baselines, cumulative abnormal return (CAR) T-5..T-1,
abnormal returns on T-1/T-2, trade-level BUY/SELL vs trader history, p2_signals.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_P2 = Path(__file__).resolve().parent
REPO_ROOT = _P2.parent
OUT_CSV = REPO_ROOT / "p2_signals.csv"
if str(_P2) not in sys.path:
    sys.path.insert(0, str(_P2))

from cik_map import cik_for_ticker, load_ticker_cik_map  # noqa: E402
from edgar_efts import search_8k_keyword, search_8k_ticker  # noqa: E402
from edgar_filings import FilingRow, archives_url, fetch_material_filings, headline_for_filing  # noqa: E402
from load_data import load_all  # noqa: E402

# Prep-guide NLP buckets (order: M&A first for enforcement-style priority)
EVENT_KEYWORDS = {
    "merger": [
        "merger",
        "acquisition",
        "acquired",
        "takeover",
        "combine",
        "business combination",
        "asset purchase",
    ],
    "earnings": [
        "earnings",
        "revenue",
        "quarterly results",
        "eps",
        "guidance",
        "financial results",
        "2.02",
    ],
    "leadership": [
        "ceo",
        "chief executive",
        "cfo",
        "resign",
        "appoint",
        "director",
        "board",
        "5.02",
        "officer",
    ],
    "restatement": [
        "restate",
        "restatement",
        "correction",
        "material weakness",
        "4.02",
    ],
}


def classify_event_headline(headline: str) -> str:
    h = headline.lower()
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(kw in h for kw in keywords):
            return event_type
    return "other"


def earliest_filing_with_history(oh_first: pd.Timestamp) -> str:
    return str((oh_first + pd.offsets.BDay(12)).date())


def enrich_ohlcv(oh: pd.DataFrame) -> pd.DataFrame:
    oh = oh.sort_values(["sec_id", "trade_date"]).copy()
    oh["ret"] = oh.groupby("sec_id")["close"].pct_change()
    oh["vol_mean15"] = oh.groupby("sec_id")["volume"].transform(
        lambda s: s.shift(1).rolling(15, min_periods=8).mean()
    )
    oh["vol_std15"] = oh.groupby("sec_id")["volume"].transform(
        lambda s: s.shift(1).rolling(15, min_periods=8).std()
    )
    oh["vol_std15"] = oh["vol_std15"].replace(0, np.nan)
    oh["vol_z"] = (oh["volume"] - oh["vol_mean15"]) / oh["vol_std15"]
    return oh


def pre_window_trading_days(
    oh: pd.DataFrame, sec_id: int, filing_cal: pd.Timestamp
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, List[pd.Timestamp]]]:
    dts = oh.loc[oh["sec_id"] == sec_id, "trade_date"].sort_values()
    prior = dts[dts < filing_cal.normalize()].tolist()
    if len(prior) < 6:
        return None
    t1 = prior[-1]
    t2 = prior[-2]
    t5 = prior[-5]
    window = prior[-5:]
    return t5, t1, t2, window


def sigma_mu_before_T5(oh: pd.DataFrame, sec_id: int, t5: pd.Timestamp) -> Tuple[float, float]:
    sl = oh[oh["sec_id"] == sec_id].sort_values("trade_date").reset_index(drop=True)
    m = sl[sl["trade_date"].dt.normalize() == pd.Timestamp(t5).normalize()]
    if m.empty:
        return 0.0, 0.0
    j = int(m.index[0])
    if j < 2:
        return 0.0, 0.0
    base = sl.iloc[max(0, j - 15) : j]
    r = base["ret"].dropna()
    if r.empty:
        return 0.0, 0.0
    return float(r.mean()), float(r.std(ddof=1)) if len(r) > 1 else 0.0


def ret_on_date(oh: pd.DataFrame, sec_id: int, d: pd.Timestamp) -> float:
    m = oh[
        (oh["sec_id"] == sec_id)
        & (oh["trade_date"].dt.normalize() == pd.Timestamp(d).normalize())
    ]
    if m.empty:
        return float("nan")
    return float(m.iloc[0]["ret"])


def car_and_abnormal_returns(
    oh: pd.DataFrame, sec_id: int, window: List[pd.Timestamp], mu: float
) -> Tuple[float, float, float, List[float]]:
    """
    ``window`` = [T-5, T-4, T-3, T-2, T-1] ascending.
    CAR = sum((r_i - mu)) for daily returns on T-4..T-1 (4 days).
    Returns CAR, abnormal_ret_T-1, abnormal_ret_T-2, list of 4 daily rets.
    """
    if len(window) != 5:
        return float("nan"), float("nan"), float("nan"), []
    rets: List[float] = []
    for d in window[1:]:
        rv = ret_on_date(oh, sec_id, pd.Timestamp(d))
        rets.append(rv)
    if len(rets) != 4 or not all(np.isfinite(rets)):
        return float("nan"), float("nan"), float("nan"), rets
    ar = [r - mu for r in rets]
    CAR = float(sum(ar))
    # rets order: T-4, T-3, T-2, T-1
    abn_t1 = rets[3] - mu
    abn_t2 = rets[2] - mu
    return CAR, abn_t1, abn_t2, rets


def cumulative_simple_T5_T1(oh: pd.DataFrame, sec_id: int, t5: pd.Timestamp, t1: pd.Timestamp) -> float:
    sl = oh[(oh["sec_id"] == sec_id)].sort_values("trade_date")
    c5 = sl.loc[sl["trade_date"].dt.normalize() == t5.normalize(), "close"]
    c1 = sl.loc[sl["trade_date"].dt.normalize() == t1.normalize(), "close"]
    if c5.empty or c1.empty:
        return float("nan")
    return float(c1.iloc[0] / c5.iloc[0] - 1.0)


def vz_on(oh: pd.DataFrame, sec_id: int, d: pd.Timestamp) -> float:
    m = oh[
        (oh["sec_id"] == sec_id)
        & (oh["trade_date"].dt.normalize() == pd.Timestamp(d).normalize())
    ]
    return float(m["vol_z"].iloc[0]) if not m.empty else float("nan")


def trade_evidence_side(
    trades: pd.DataFrame,
    sec_id: int,
    window_start: pd.Timestamp,
    filing_dt: pd.Timestamp,
    side: str,
) -> Tuple[bool, str]:
    t = trades[(trades["sec_id"] == sec_id)].copy()
    t = t[t["order_status"].str.upper() == "FILLED"]
    t = t[t["side"].str.upper() == side.upper()]
    if t.empty:
        return False, ""
    pre_all = t[t["timestamp"] < filing_dt]
    in_win = t[
        (t["timestamp"] >= pd.Timestamp(window_start).normalize())
        & (t["timestamp"] < filing_dt)
    ]
    if in_win.empty:
        return False, ""

    strong = []
    for tr_id, grp in in_win.groupby("trader_id"):
        base = pre_all[pre_all["trader_id"] == tr_id]
        med = float(base["quantity"].median()) if len(base) > 0 else 0.0
        std = float(base["quantity"].std(ddof=1)) if len(base) > 2 else 0.0
        thr = max(med * 4.0, med + 3.0 * std if np.isfinite(std) else med * 4.0, 1.0)
        mx = float(grp["quantity"].max())
        if mx >= thr and mx >= med * 2.5:
            strong.append(f"{tr_id} {side} max_qty≈{mx:.0f} vs typical≈{med:.0f}")
    if not strong:
        return False, ""
    return True, "; ".join(strong[:2])


def trades_evidence(
    trades: pd.DataFrame, sec_id: int, window_start: pd.Timestamp, filing_dt: pd.Timestamp
) -> Tuple[bool, str]:
    parts = []
    hit = False
    for side in ("BUY", "SELL"):
        ok, msg = trade_evidence_side(trades, sec_id, window_start, filing_dt, side)
        if ok:
            hit = True
            parts.append(msg)
    return hit, " | ".join(parts) if parts else ""


def earliest_suspicious_day(
    oh: pd.DataFrame,
    sec_id: int,
    window: List[pd.Timestamp],
    trades: pd.DataFrame,
    filing_dt: pd.Timestamp,
    vol_thr: float = 2.0,
) -> pd.Timestamp:
    t5 = pd.Timestamp(window[0]).normalize()
    candidates = [t5]
    for d in window:
        z = vz_on(oh, sec_id, pd.Timestamp(d))
        if np.isfinite(z) and z > vol_thr:
            candidates.append(pd.Timestamp(d).normalize())
    t = trades[
        (trades["sec_id"] == sec_id)
        & (trades["order_status"].str.upper() == "FILLED")
        & (trades["timestamp"] >= t5)
        & (trades["timestamp"] < filing_dt)
    ]
    if not t.empty:
        candidates.append(pd.Timestamp(t["timestamp"].min()).normalize())
    return min(candidates)


def compute_signal_for_filing(
    oh: pd.DataFrame,
    trades: pd.DataFrame,
    name_by_sec: Dict[int, str],
    fr: FilingRow,
) -> Tuple[int, str, str]:
    filing_dt = pd.Timestamp(fr.filing_date)
    cal = pre_window_trading_days(oh, fr.sec_id, filing_dt)
    if cal is None:
        return 0, "", (
            f"Insufficient OHLCV history before filing {fr.filing_date} for sec_id {fr.sec_id}; "
            f"cannot form T-5..T-1 window."
        )
    t5, t1, t2, window = cal
    mu, sigma = sigma_mu_before_T5(oh, fr.sec_id, pd.Timestamp(t5))

    CAR, abn_t1, abn_t2, seg_rets = car_and_abnormal_returns(oh, fr.sec_id, window, mu)
    n_days = 4
    car_threshold = 2.0 * np.sqrt(n_days) * sigma if sigma > 1e-12 else np.inf
    car_ok = np.isfinite(CAR) and sigma > 1e-12 and CAR > car_threshold

    vol_z1 = vz_on(oh, fr.sec_id, pd.Timestamp(t1))
    vol_z2 = vz_on(oh, fr.sec_id, pd.Timestamp(t2))
    vol_ok = (np.isfinite(vol_z1) and vol_z1 > 3.0) or (np.isfinite(vol_z2) and vol_z2 > 3.0)

    pre_drift_simple = cumulative_simple_T5_T1(oh, fr.sec_id, pd.Timestamp(t5), pd.Timestamp(t1))
    # Simple cumulative return vs 2-sigma 4-day null (brief alternative)
    simple_ok = (
        np.isfinite(pre_drift_simple)
        and sigma > 1e-12
        and (pre_drift_simple - n_days * mu) > car_threshold
    )

    drift_ok = car_ok or simple_ok

    trade_hit, trade_txt = trades_evidence(trades, fr.sec_id, pd.Timestamp(t5), filing_dt)
    pre_flag = 1 if (drift_ok or vol_ok or trade_hit) else 0

    sus = earliest_suspicious_day(oh, fr.sec_id, window, trades, filing_dt)
    suspicious_start = str(sus.date())

    seg_str = ",".join(f"{100 * r:.2f}%" for r in seg_rets) if seg_rets else "n/a"
    parts = [
        f"Pre-filing window ending {t1.date()} (T-1) before public filing {fr.filing_date}; "
        f"T-5..T-1 daily returns (T-4..T-1): {seg_str}. "
        f"15-day baseline before T-5: mean daily return μ={100*mu:.4f}%, σ={100*sigma:.4f}%. "
        f"Cumulative abnormal return (CAR) over T-4..T-1 = sum(r-μ) = {100*CAR:.3f}% "
        f"(flag if CAR > 2·√4·σ ≈ {100*car_threshold:.3f}%). "
        f"Abnormal return T-1 = {100*abn_t1:.3f}%, T-2 = {100*abn_t2:.3f}% vs μ. "
        f"Simple cumulative return close(T-1)/close(T-5)-1 = {100*pre_drift_simple:.2f}%. "
        f"Volume z (15 prior trading days mean/std): T-1={vol_z1:.2f}, T-2={vol_z2:.2f} (strong if >3)."
    ]
    if trade_hit:
        parts.append(f"Trade-level (FILLED BUY/SELL vs own history on this sec_id): {trade_txt}")
    else:
        parts.append(
            "Trade-level: no outsized FILLED BUY/SELL vs trader baselines in T-5..T-1 before filing."
        )
    if pre_flag == 0:
        parts.append(
            "No pre-announcement signal: volume, CAR/simple drift, and trade checks below thresholds — clean."
        )
    return pre_flag, suspicious_start, " ".join(parts)


def merge_filings(
    submissions_rows: List[FilingRow],
    efts_ticker_rows: List[FilingRow],
    efts_merger_rows: List[FilingRow],
) -> List[FilingRow]:
    seen: set = set()
    out: List[FilingRow] = []
    for fr in submissions_rows + efts_ticker_rows + efts_merger_rows:
        key = (fr.accession.replace("-", ""), fr.filing_date, fr.sec_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(fr)
    return out


def main() -> None:
    t0 = time.perf_counter()
    mapping = load_ticker_cik_map()
    oh_raw, trades = load_all()
    meta = oh_raw[["sec_id", "ticker", "name"]].drop_duplicates("sec_id")
    name_by_sec = dict(zip(meta["sec_id"], meta["name"]))
    tick_map = meta.set_index("sec_id")["ticker"].to_dict()
    ticker_to_sec_id = {str(v).upper(): int(k) for k, v in tick_map.items()}

    oh = enrich_ohlcv(oh_raw)
    dmin = oh["trade_date"].min()
    dmax = oh["trade_date"].max()
    start_filings = str((dmin - pd.Timedelta(days=7)).date())
    end_filings = str(dmax.date())
    filing_floor = earliest_filing_with_history(pd.Timestamp(dmin))

    filings_sub: List[FilingRow] = []
    filings_efts_t: List[FilingRow] = []
    for sid, ticker in tick_map.items():
        cik = cik_for_ticker(str(ticker), mapping)
        if cik is not None:
            filings_sub.extend(
                fetch_material_filings(
                    str(ticker), int(sid), int(cik), start_filings, end_filings, sleep_s=0.22
                )
            )
        filings_efts_t.extend(
            search_8k_ticker(str(ticker), int(sid), start_filings, end_filings, sleep_s=0.28)
        )

    filings_m_a: List[FilingRow] = []
    filings_m_a.extend(
        search_8k_keyword("merger", ticker_to_sec_id, start_filings, end_filings, sleep_s=0.28)
    )
    filings_m_a.extend(
        search_8k_keyword("acquisition", ticker_to_sec_id, start_filings, end_filings, sleep_s=0.28)
    )

    uniq = merge_filings(filings_sub, filings_efts_t, filings_m_a)

    rows = []
    for fr in uniq:
        if fr.filing_date < filing_floor:
            continue
        nm = name_by_sec.get(fr.sec_id, fr.ticker)
        hl = headline_for_filing(nm, fr)
        evt = classify_event_headline(hl + " " + fr.items)
        flag, sus_start, remarks = compute_signal_for_filing(oh, trades, name_by_sec, fr)
        if not sus_start.strip() and "cannot form" in remarks.lower():
            continue
        # Brief-aligned filing URL (Archives); efts search URL noted for traceability in README
        src = archives_url(fr.cik, fr.accession, fr.primary_document)
        rows.append(
            {
                "sec_id": fr.sec_id,
                "event_date": fr.filing_date,
                "event_type": evt,
                "headline": hl,
                "source_url": src,
                "pre_drift_flag": int(flag),
                "suspicious_window_start": sus_start,
                "remarks": remarks,
            }
        )

    total = time.perf_counter() - t0
    cols_merged = [
        "sec_id",
        "event_date",
        "event_type",
        "headline",
        "source_url",
        "pre_drift_flag",
        "suspicious_window_start",
        "remarks",
        "time_to_run",
    ]
    if not rows:
        df = pd.DataFrame(columns=cols_merged)
    else:
        df = pd.DataFrame(rows)
        df = df.sort_values(["event_date", "sec_id", "headline"]).reset_index(drop=True)
        df["time_to_run"] = round(total, 2)
        df = df[cols_merged]
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUT_CSV} in {total:.2f}s")


if __name__ == "__main__":
    main()
