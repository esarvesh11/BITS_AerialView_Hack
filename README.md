# BITS Aerial View — Hackathon Work

Short notes on approach and outputs per problem. Judges may use this for borderline scoring.

---

## Problem 1 — Order book concentration (equity, bonus)

### Approach (matches brief: baselines, sustained patterns, clustering)

We treat **Problem 1** as surveillance on **1-minute L2 snapshots** (top 10 bid/ask levels) plus **OHLCV** (calendar context) and **event-week trades**. Everything is **normalized per ticker** (and per day where noted) before comparing names across the book.

1. **Core features (per minute, per `sec_id`)**  
   - **OBI** from total bid/ask depth across 10 levels.  
   - **Spread (bps)** from best bid/ask.  
   - **L1 concentration**: fraction of each side’s depth at level 1.  
   - **Rolling 10-minute mean and std of OBI** (`obi_ma10`, `obi_std10`) — sustained pressure vs one-off spikes.  
   - **Spread vs “30-day” baseline**: daily median spread from `market_data`, then a rolling mean of **prior** daily medians (up to 30 trading days in the CSV). Minute bars use **ratio to that baseline** so a \$0.50 spread is judged against *that* name’s history, not a global constant.  
   - **Cross-level shape**: **Herfindahl index** (HHI) of bid sizes across levels 1–10 — captures whether depth is stacked unusually at the top vs distributed; z-scored per ticker.  
   - **Trade pressure**: **BUY volume per minute / total bid depth** (aggressive buy into the bid stack), z-scored per ticker.  
   - **Per-day OBI median deviation** within each ticker-day (compliance-style “session” context).  
   - **Short-horizon z-scores** (rolling window, with **minimum σ** floors) for OBI, spread, concentration, HHI, buy-pressure.

2. **Episodes (sustained behaviour, not single minutes)**  
   Candidate minutes are merged into **episodes** (≤ **45** minutes clock span, ≤ **3** minute gaps). Each episode must clear a **multi-signal score** (several of: imbalance streaks, local OBI/spread z-spikes, spread ratio vs prior-daily baseline, L1 concentration streaks, L1 depth asymmetry, HHI anomaly, aggressive buy vs depth).

3. **Clustering (group by type of stress)**  
   Each episode gets a **feature vector** (mean OBI, mean spread ratio to 30d baseline, mean L1 bid share, mean bid HHI, mean buy/bid depth, mean 10m OBI volatility). **KMeans** (small *k*, scaled features) assigns a **cluster tag** (named from the strongest coordinate of the cluster centroid, e.g. `type_spread_ratio_30d_mean`). The CSV **`anomaly_type`** is `cluster_tag|signal1+signal2+…` so analysts see both the **family** of behaviour and the **concrete** triggers.

4. **Other detectors**  
   - **IsolationForest** on minute-level joint features (including spread ratio and HHI) for rare joint states.  
   - **Cancel bursts**: many large-band **CANCELLED** orders same trader/side in ~12 minutes (`unusual_cancel_pattern`, as in the problem example).

5. **Output control**  
   Drop **LOW**, **dedupe**, cap at **40** rows, sort by severity and time. **Remarks** spell out numbers (OBI, spread ratio, HHI, BUY/bid depth) for partial credit.

### Outputs

| File | Description |
|------|-------------|
| `p1_alerts.csv` | Root of this repo — submission format for Problem 1. |

### How to run

From the repository root (with `pandas`, `numpy`, `scikit-learn` installed):

```bash
python problem1/run_p1.py
```

Data is read from `problem1/dataset/` (`market_data.csv`, `ohlcv.csv`, `trade_data.csv`). Runtime is written into the `time_to_run` column (seconds).

---

## Problem 2 — Insider / pre-announcement signal (equity, bonus)

### Approach (spec + prep guide)

1. **EDGAR 8-K**  
   - **Submissions**: `data.sec.gov/submissions/CIK##########.json` — **8-K only**.  
   - **efts**: `efts.sec.gov/LATEST/search-index` with **`forms=8-K`**, **`q="{TICKER}"`**, filtered on **`(TICKER)`** in `display_names`.  
   - **M&A discovery**: `q=merger` and `q=acquisition` with the same form/date filter, universe via **`(TICKER)`** in hit text.  
   - **CIK** from cached **`company_tickers.json`**. **`source_url`** = Archives filing (primary doc from submissions when present, else **`index.htm`**).

2. **Calendar**  
   - **T−5…T−1** = last five OHLCV dates strictly before the filing date; **15-day** volume and return baselines end before **T−5**. Early filings without history are skipped.

3. **Signals**  
   - **Volume z** on **T−1** / **T−2** vs 15-day shifted mean/std; **z > 3** flags.  
   - **CAR**: Σ(**r**−**μ**) over **T−4…T−1** daily returns; flag if **CAR > 2·√4·σ**; **abnormal returns on T−1 and T−2** vs μ in **remarks**; backup check on simple **close(T−1)/close(T−5)−1**.  
   - **Trades**: outsized **FILLED** **BUY** and **SELL** vs each **trader_id**’s prior activity on that **`sec_id`**.

4. **`event_type`**  
   - Prep keyword buckets: **merger**, **earnings**, **leadership**, **restatement**, **other** (headline + items).

5. **`pre_drift_flag` / `suspicious_window_start`**  
   - **1** if drift (CAR/simple), volume, or trade rule fires. **suspicious_window_start** = earliest of **T−5**, high-volume day (**z>2**), or first trade in the window.

6. **Runtime**  
   - Many SEC round-trips (~**25–60s** typical); under **5 min** bonus target.

### Output

| File | Description |
|------|-------------|
| `p2_signals.csv` | Repo root — columns per problem statement. |

### How to run

Requires **network** and **`requests`**:

```bash
python problem2/run_p2.py
```

Data: `problem2/dataset/` if present, else **`problem1/dataset/`** (`ohlcv.csv`, `trade_data.csv`).

---

## Problem 3 — Crypto blind anomaly hunt (compulsory)

Work in this repo includes exploratory analysis and ranked suspicious trades; the graded list is intended as **`submission.csv`** at the repo root with columns `symbol`, `date`, `trade_id`, optional `violation_type`, and optional `remarks`. See `Problem3/phase1_usdcusdt.ipynb` for an example focused on **USDCUSDT** peg and related patterns. Extend the same ideas across the remaining pairs with pair-specific baselines (liquidity differs a lot by symbol).

---

## Repository layout (relevant paths)

```
problem1/
  dataset/           # market_data.csv, ohlcv.csv, trade_data.csv
  load_data.py
  run_p1.py          # → ../p1_alerts.csv
problem2/
  cache/             # company_tickers.json (downloaded once)
  cik_map.py
  edgar_filings.py   # submissions 8-K
  edgar_efts.py      # efts 8-K + merger/acquisition pass
  load_data.py
  run_p2.py          # → ../p2_signals.csv
p1_alerts.csv
p2_signals.csv
Problem3/submission.csv   # Problem 3 (example)
requirements.txt
```

---

## Dependencies

See `requirements.txt`. **pandas**, **numpy**, **scikit-learn** (P1), **requests** (P2 EDGAR).
