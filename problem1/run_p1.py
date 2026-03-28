"""
Problem 1 — Order book concentration (equity).

Aligned with brief: per-ticker baselines, 10m OBI stats, spread vs long lookback,
trade pressure vs depth, cross-level shape, clustering of episode types → p1_alerts.csv.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PROBLEM_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBLEM_DIR.parent
OUT_CSV = REPO_ROOT / "p1_alerts.csv"

BID_SIZE_COLS = [f"bid_size_level{i:02d}" for i in range(1, 11)]
ASK_SIZE_COLS = [f"ask_size_level{i:02d}" for i in range(1, 11)]

FEATURE_NAMES_FOR_CLUSTER = [
    "obi_mean",
    "spread_ratio_30d_mean",
    "bid_conc_mean",
    "bid_herf_mean",
    "buy_to_bid_mean",
    "obi_std10_mean",
]


def _nmean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.mean(a))


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _herfindahl(sizes: np.ndarray) -> np.ndarray:
    """Row-wise Herfindahl of nonnegative sizes (concentration of depth across levels)."""
    s = np.asarray(sizes, dtype=float)
    row_sum = s.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(row_sum > 0, s / row_sum, 0.0)
    return (p * p).sum(axis=1)


def enrich_market(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sec_id", "timestamp"]).copy()
    for c in BID_SIZE_COLS + ASK_SIZE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    bid_m = df[BID_SIZE_COLS].to_numpy(dtype=float)
    ask_m = df[ASK_SIZE_COLS].to_numpy(dtype=float)
    total_bid = bid_m.sum(axis=1)
    total_ask = ask_m.sum(axis=1)
    depth = total_bid + total_ask
    df["total_bid"] = total_bid
    df["total_ask"] = total_ask
    with np.errstate(divide="ignore", invalid="ignore"):
        df["obi"] = np.where(depth > 0, (total_bid - total_ask) / depth, np.nan)
    bp = pd.to_numeric(df["bid_price_level01"], errors="coerce")
    ap = pd.to_numeric(df["ask_price_level01"], errors="coerce")
    spread = ap - bp
    df["spread_bps"] = _safe_div(spread, bp) * 10000.0
    df["bid_concentration"] = _safe_div(df["bid_size_level01"], pd.Series(total_bid, index=df.index))
    df["ask_concentration"] = _safe_div(df["ask_size_level01"], pd.Series(total_ask, index=df.index))
    df["depth_ratio_l1"] = _safe_div(df["bid_size_level01"], df["ask_size_level01"])
    df["bid_herfindahl"] = _herfindahl(bid_m)
    df["ask_herfindahl"] = _herfindahl(ask_m)
    df["trade_date"] = df["timestamp"].dt.date.astype(str)
    df["md_date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    return df


def add_daily_spread_baseline(m: pd.DataFrame) -> pd.DataFrame:
    """
    Per sec_id: daily median spread_bps, then rolling mean of *prior* daily medians
    (up to 30 trading days in sample — uses all available history in CSV).
    Minute-level ``spread_ratio_to_30d`` compares current bar to that baseline.
    """
    daily = (
        m.groupby(["sec_id", "md_date"], sort=False)["spread_bps"]
        .median()
        .reset_index(name="daily_med_spread")
    )
    daily = daily.sort_values(["sec_id", "md_date"])
    daily["spread_30d_baseline"] = daily.groupby("sec_id")["daily_med_spread"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()
    )
    # First days: backfill baseline with expanding mean of prior daily medians
    m = m.merge(
        daily[["sec_id", "md_date", "spread_30d_baseline"]],
        on=["sec_id", "md_date"],
        how="left",
    )
    # Fallback: ticker-wide median spread if no prior day
    med_ticker = m.groupby("sec_id")["spread_bps"].transform("median")
    base = m["spread_30d_baseline"].fillna(med_ticker)
    m["spread_ratio_to_30d"] = m["spread_bps"] / (base + 1e-6)
    return m


def join_minute_buy_volume(m: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """BUY quantity per (sec_id, minute) → buy_to_bid_depth ratio."""
    t = trades.copy()
    t["ts_min"] = t["timestamp"].dt.floor("min")
    buys = t[t["side"].str.upper() == "BUY"]
    bv = buys.groupby(["sec_id", "ts_min"], sort=False)["quantity"].sum().reset_index(name="buy_vol_min")
    m = m.copy()
    m["ts_min"] = m["timestamp"].dt.floor("min")
    m = m.merge(bv, on=["sec_id", "ts_min"], how="left")
    m["buy_vol_min"] = m["buy_vol_min"].fillna(0.0)
    m["buy_to_bid_depth"] = m["buy_vol_min"] / (m["total_bid"] + 1e-6)
    return m


def rolling_z(
    s: pd.Series,
    win: int = 45,
    min_p: int = 20,
    min_std: float = 1e-6,
) -> pd.Series:
    m = s.rolling(win, min_periods=min_p).mean()
    sd = s.rolling(win, min_periods=min_p).std().clip(lower=min_std)
    z = (s - m) / sd
    return z.clip(-12, 12)


def _run_streak(cond: pd.Series, min_len: int) -> pd.Series:
    x = cond.astype(int)
    run_id = (x != x.shift()).cumsum()
    run_len = x.groupby(run_id).transform("sum")
    return (x == 1) & (run_len >= min_len)


def add_group_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("sec_id", sort=False):
        g = g.copy()
        # Rolling 10-minute OBI mean and std (brief: clustering feature)
        g["obi_ma10"] = g["obi"].rolling(10, min_periods=5).mean()
        g["obi_std10"] = g["obi"].rolling(10, min_periods=5).std()
        z_obi = rolling_z(g["obi"], min_std=0.05)
        z_sp = rolling_z(g["spread_bps"], min_std=0.8)
        z_bc = rolling_z(g["bid_concentration"], min_std=0.04)
        z_bh = rolling_z(g["bid_herfindahl"], min_std=0.02)
        z_buy = rolling_z(g["buy_to_bid_depth"], min_std=1e-4)
        g["obi_z"] = z_obi
        g["spread_z"] = z_sp
        g["bid_conc_z"] = z_bc
        g["bid_herf_z"] = z_bh
        g["buy_pressure_z"] = z_buy
        g["mask_obi_streak"] = _run_streak(g["obi"].abs() > 0.72, 10)
        g["mask_bconc_streak"] = _run_streak(g["bid_concentration"] > 0.85, 8)
        g["mask_aconc_streak"] = _run_streak(g["ask_concentration"] > 0.85, 8)
        # Per-ticker, per-day median OBI — deviation (brief: per-day baseline)
        day_med_obi = g.groupby(g["timestamp"].dt.date)["obi"].transform("median")
        g["obi_vs_day_median"] = g["obi"] - day_med_obi
        out.append(g)
    return pd.concat(out, ignore_index=True)


@dataclass
class EpisodeAlert:
    sec_id: int
    trade_date: str
    time_window_start: str
    signal_types: List[str]
    severity: str
    remarks: str
    features: Dict[str, float] = field(default_factory=dict)
    cluster_tag: str = "unclustered"


def _episode_feature_dict(part: pd.DataFrame) -> Dict[str, float]:
    return {
        "obi_mean": _nmean(part["obi"].to_numpy(dtype=float)),
        "spread_ratio_30d_mean": _nmean(part["spread_ratio_to_30d"].to_numpy(dtype=float)),
        "bid_conc_mean": _nmean(part["bid_concentration"].to_numpy(dtype=float)),
        "bid_herf_mean": _nmean(part["bid_herfindahl"].to_numpy(dtype=float)),
        "buy_to_bid_mean": _nmean(part["buy_to_bid_depth"].to_numpy(dtype=float)),
        "obi_std10_mean": _nmean(part["obi_std10"].to_numpy(dtype=float)),
    }


def market_episodes(df: pd.DataFrame) -> List[EpisodeAlert]:
    alerts: List[EpisodeAlert] = []
    df = df.sort_values(["sec_id", "timestamp"])

    dr = df["depth_ratio_l1"].replace([np.inf, -np.inf], np.nan)
    mask_obi = df["mask_obi_streak"]
    mask_obi_z = (
        (df["obi_z"].abs() > 3.5)
        & df["obi_z"].notna()
        & (df["obi"].abs() > 0.35)
    )
    mask_sp = (
        (df["spread_z"] > 3.8)
        & df["spread_z"].notna()
        & (df["spread_bps"] > 6)
    )
    mask_sp_long = (df["spread_ratio_to_30d"] > 2.5) & (df["spread_bps"] > 5)
    mask_bconc = df["mask_bconc_streak"]
    mask_aconc = df["mask_aconc_streak"]
    mask_depth = ((dr > 15) | (dr < (1 / 15))) & dr.notna() & (df["obi"].abs() > 0.4)
    mask_herf = (df["bid_herf_z"].abs() > 3.0) & df["bid_herf_z"].notna()
    mask_buy = (df["buy_pressure_z"] > 3.5) & (df["buy_to_bid_depth"] > 0.02)

    df["_flag"] = (
        mask_obi
        | mask_obi_z
        | mask_sp
        | mask_sp_long
        | mask_bconc
        | mask_aconc
        | mask_depth
        | mask_herf
        | mask_buy
    )

    for sid, g in df[df["_flag"]].groupby("sec_id", sort=False):
        g = g.sort_values("timestamp")
        start = 0
        while start < len(g):
            end = start
            while end + 1 < len(g):
                gap_m = (
                    g["timestamp"].iloc[end + 1] - g["timestamp"].iloc[end]
                ).total_seconds() / 60.0
                span_m = (
                    g["timestamp"].iloc[end + 1] - g["timestamp"].iloc[start]
                ).total_seconds() / 60.0
                if gap_m > 3:
                    break
                if span_m > 45:
                    break
                end += 1
            part = g.iloc[start : end + 1]
            start = end + 1
            if part.empty or part["obi"].notna().sum() < 3:
                continue

            types: List[str] = []
            score = 0
            if (part["obi"].abs() > 0.72).sum() >= 10:
                types.append("order_book_imbalance")
                score += 3
            if (part["obi_z"].abs() > 3.2).any() and (part["obi"].abs() > 0.35).any():
                types.append("obi_local_z_spike")
                score += 2
            if (part["spread_z"] > 3.8).any() and (part["spread_bps"] > 6).any():
                types.append("abnormal_spread_bps")
                score += 2
            if (part["spread_ratio_to_30d"] > 2.5).any() and (part["spread_bps"] > 5).any():
                types.append("spread_vs_30d_daily_baseline")
                score += 2
            if (part["bid_concentration"] > 0.85).sum() >= 8:
                types.append("bid_L1_concentration")
                score += 2
            if (part["ask_concentration"] > 0.85).sum() >= 8:
                types.append("ask_L1_concentration")
                score += 2
            drp = part["depth_ratio_l1"].replace([np.inf, -np.inf], np.nan)
            if ((drp > 15) | (drp < (1 / 15))).any() and (part["obi"].abs() > 0.4).any():
                types.append("L1_depth_asymmetry")
                score += 1
            if (part["bid_herf_z"].abs() > 3.0).any():
                types.append("cross_level_bid_shape_anomaly")
                score += 2
            if (part["buy_pressure_z"] > 3.0).any():
                types.append("aggressive_buy_vs_bid_depth")
                score += 2

            if score < 6 or not types:
                continue

            if score >= 8:
                sev = "HIGH"
            elif score >= 6:
                sev = "MEDIUM"
            else:
                sev = "LOW"

            t0 = part["timestamp"].iloc[0]
            t_end = part["timestamp"].iloc[-1]
            mins = (t_end - t0).total_seconds() / 60.0 + 1
            feat = _episode_feature_dict(part)
            obi_m = feat["obi_mean"]
            obi_mx = float(np.nanmax(np.abs(part["obi"].to_numpy(dtype=float))))
            sp_m = _nmean(part["spread_bps"].to_numpy(dtype=float))
            sp_z = float(part["spread_z"].max()) if part["spread_z"].notna().any() else float("nan")
            bc_m = feat["bid_conc_mean"]
            sr30 = feat["spread_ratio_30d_mean"]
            o10 = feat["obi_std10_mean"]
            buy_r = feat["buy_to_bid_mean"]
            remarks = (
                f"Episode ~{mins:.0f}min (per-ticker rolling baselines): mean OBI={obi_m:.3f}, "
                f"max|OBI|={obi_mx:.3f}; 10m OBI std avg={o10:.4f}; "
                f"spread_bps mean={sp_m:.2f}, ratio vs prior-daily-30d-mean={sr30:.2f}x; "
                f"bid L1 share avg={bc_m:.2f}; "
                f"BUY volume / bid depth (minute) avg={buy_r:.4f}. "
                f"Signals: {', '.join(types)}. "
                f"Cross-level bid Herfindahl avg={feat['bid_herf_mean']:.3f} (shape vs typical depth stack)."
            )
            if np.isfinite(sp_z):
                remarks += f" Max local spread z={sp_z:.2f}."

            alerts.append(
                EpisodeAlert(
                    sec_id=int(sid),
                    trade_date=str(t0.date()),
                    time_window_start=t0.strftime("%H:%M:%S"),
                    signal_types=types,
                    severity=sev,
                    remarks=remarks,
                    features=feat,
                )
            )
    return alerts


def cluster_episode_types(episodes: List[EpisodeAlert]) -> List[EpisodeAlert]:
    """KMeans on episode feature vectors → descriptive cluster tag prepended to anomaly_type."""
    if len(episodes) < 6:
        for e in episodes:
            e.cluster_tag = "few_episodes_no_cluster"
        return episodes

    X = np.array([[e.features.get(k, 0.0) for k in FEATURE_NAMES_FOR_CLUSTER] for e in episodes])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = max(3, min(6, len(episodes) // 3))
    k = min(k, len(episodes))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)

    centroids = km.cluster_centers_
    names = []
    for ci in range(k):
        j = int(np.argmax(np.abs(centroids[ci])))
        names.append(f"type_{FEATURE_NAMES_FOR_CLUSTER[j]}")

    for e, lab in zip(episodes, labels):
        e.cluster_tag = names[int(lab)]
    return episodes


def episodes_to_alerts(episodes: List[EpisodeAlert]) -> List[Tuple[str, str, str, str, str, str]]:
    """Returns list of (sec_id, date, time, anomaly_type, severity, remarks)."""
    out = []
    for e in episodes:
        atype = f"{e.cluster_tag}|" + "+".join(sorted(set(e.signal_types)))
        out.append(
            (
                str(e.sec_id),
                e.trade_date,
                e.time_window_start,
                atype,
                e.severity,
                e.remarks,
            )
        )
    return out


def isolation_top_outliers(df: pd.DataFrame, max_alerts: int = 6) -> List[Tuple[str, str, str, str, str, str]]:
    feat_cols = [
        "obi",
        "spread_bps",
        "spread_ratio_to_30d",
        "bid_concentration",
        "bid_herfindahl",
        "buy_to_bid_depth",
    ]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(X) < 200:
        return []
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.008,
        random_state=42,
        n_jobs=1,
    )
    pred = iso.fit_predict(Xs)
    score = iso.score_samples(Xs)
    idx = X.index[pred == -1]
    sub = df.loc[idx].copy()
    sub["_iso"] = score[pred == -1]
    sub = sub.nsmallest(max_alerts, "_iso")
    rows = []
    for _, row in sub.iterrows():
        rows.append(
            (
                str(int(row["sec_id"])),
                str(row["timestamp"].date()),
                row["timestamp"].strftime("%H:%M:%S"),
                "isolation_forest_rare_joint_microstructure",
                "MEDIUM",
                (
                    f"Rare joint state vs population: OBI={row['obi']:.3f}, "
                    f"spread_bps={row['spread_bps']:.2f}, "
                    f"spread×vs 30d daily baseline={row['spread_ratio_to_30d']:.2f}, "
                    f"bid_L1_share={row['bid_concentration']:.2f}, "
                    f"bid_HHI={row['bid_herfindahl']:.3f}, "
                    f"BUY/bid_depth={row['buy_to_bid_depth']:.4f}."
                ),
            )
        )
    return rows


def trade_cancel_alerts(trades: pd.DataFrame, market_days: pd.Series) -> List[Tuple[str, str, str, str, str, str]]:
    can = trades[trades["order_status"].str.upper() == "CANCELLED"].copy()
    if can.empty:
        return []
    rows: List[Tuple[str, str, str, str, str, str]] = []
    for sid, g in can.groupby("sec_id"):
        q85 = g["quantity"].quantile(0.85)
        g = g[g["quantity"] >= max(float(q85), 1.0)]
        g = g.sort_values("timestamp").reset_index(drop=True)
        for (_tr, _side), h in g.groupby(["trader_id", "side"]):
            h = h.sort_values("timestamp").reset_index(drop=True)
            if len(h) < 6:
                continue
            i = 0
            while i < len(h):
                t0 = h.at[i, "timestamp"]
                win = h[
                    (h["timestamp"] >= t0)
                    & (h["timestamp"] <= t0 + pd.Timedelta(minutes=12))
                ]
                if len(win) >= 6:
                    qty_sum = win["quantity"].sum()
                    rows.append(
                        (
                            str(int(sid)),
                            str(pd.Timestamp(t0).date()),
                            pd.Timestamp(t0).strftime("%H:%M:%S"),
                            "unusual_cancel_pattern",
                            "MEDIUM",
                            (
                                f"{len(win)} CANCELLED rows (large size band) from same trader/side within "
                                f"~12 minutes; combined qty ~{qty_sum:.0f} — consistent with layered quote "
                                f"tests / spoofing echo when book shows one-sided pressure."
                            ),
                        )
                    )
                    end_t = t0 + pd.Timedelta(minutes=12)
                    nxt = h[h["timestamp"] > end_t]
                    if nxt.empty:
                        break
                    i = int(nxt.index[0])
                else:
                    i += 1
    days_set = set(market_days.astype(str).unique()) if len(market_days) else set()
    if days_set:
        rows = [r for r in rows if r[1] in days_set]
    return rows


def dedupe(rows: List[Tuple[str, str, str, str, str, str]]) -> List[Tuple[str, str, str, str, str, str]]:
    seen = set()
    out = []
    for r in sorted(rows, key=lambda x: (x[0], x[1], x[2])):
        key = (r[0], r[1], r[2], r[3])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_pipeline(
    market: pd.DataFrame, trades: pd.DataFrame, ohlcv: pd.DataFrame, time_to_run: float
) -> pd.DataFrame:
    del ohlcv  # reserved if we join fundamentals later; baselines use market daily history
    m = enrich_market(market)
    m = add_daily_spread_baseline(m)
    m = join_minute_buy_volume(m, trades)
    m = add_group_rolling_features(m)

    episodes = market_episodes(m)
    episodes = cluster_episode_types(episodes)
    raw_rows = episodes_to_alerts(episodes)

    raw_rows.extend(isolation_top_outliers(m, max_alerts=6))
    raw_rows.extend(trade_cancel_alerts(trades, m["trade_date"]))

    raw_rows = dedupe(raw_rows)
    sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    raw_rows = sorted(
        raw_rows,
        key=lambda r: (sev_rank.get(r[4], 3), r[1], r[2]),
    )
    max_rows = 40
    if len(raw_rows) > max_rows:
        raw_rows = raw_rows[:max_rows]

    cols = [
        "alert_id",
        "sec_id",
        "trade_date",
        "time_window_start",
        "anomaly_type",
        "severity",
        "remarks",
        "time_to_run",
    ]
    out = []
    for i, r in enumerate(raw_rows, start=1):
        out.append(
            {
                "alert_id": i,
                "sec_id": int(r[0]),
                "trade_date": r[1],
                "time_window_start": r[2],
                "anomaly_type": r[3],
                "severity": r[4],
                "remarks": r[5],
                "time_to_run": round(time_to_run, 2),
            }
        )
    return pd.DataFrame(out, columns=cols)


def main() -> None:
    t0 = time.perf_counter()
    from load_data import load_all

    market, ohlcv, trades = load_all()
    df = build_pipeline(market, trades, ohlcv, 0.0)
    total = time.perf_counter() - t0
    if len(df) > 0:
        df["time_to_run"] = round(total, 2)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} alerts to {OUT_CSV} (time_to_run={total:.2f}s)")


if __name__ == "__main__":
    main()
