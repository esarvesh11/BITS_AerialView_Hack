"""Load OHLCV and trades for Problem 2 (prefers ``problem2/dataset``, else ``problem1/dataset``)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
P1_DATA = HERE.parent / "problem1" / "dataset"


def data_dir() -> Path:
    p2d = HERE / "dataset"
    if (p2d / "ohlcv.csv").exists():
        return p2d
    return P1_DATA


def load_ohlcv() -> pd.DataFrame:
    return pd.read_csv(data_dir() / "ohlcv.csv", parse_dates=["trade_date"])


def load_trades() -> pd.DataFrame:
    return pd.read_csv(data_dir() / "trade_data.csv", parse_dates=["timestamp"])


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_ohlcv(), load_trades()
