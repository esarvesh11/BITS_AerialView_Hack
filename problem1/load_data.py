"""Load Problem 1 CSVs from ``problem1/dataset/``."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "dataset"

DEFAULT_MARKET = DATA_DIR / "market_data.csv"
DEFAULT_OHLCV = DATA_DIR / "ohlcv.csv"
DEFAULT_TRADES = DATA_DIR / "trade_data.csv"


def load_market_data(path: Optional[Path] = None) -> pd.DataFrame:
    p = Path(path) if path is not None else DEFAULT_MARKET
    df = pd.read_csv(p, parse_dates=["timestamp"])
    return df


def load_ohlcv(path: Optional[Path] = None) -> pd.DataFrame:
    p = Path(path) if path is not None else DEFAULT_OHLCV
    df = pd.read_csv(p, parse_dates=["trade_date"])
    return df


def load_trade_data(path: Optional[Path] = None) -> pd.DataFrame:
    p = Path(path) if path is not None else DEFAULT_TRADES
    df = pd.read_csv(p, parse_dates=["timestamp"])
    return df


def load_all(
    market_path: Optional[Path] = None,
    ohlcv_path: Optional[Path] = None,
    trades_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_market_data(market_path),
        load_ohlcv(ohlcv_path),
        load_trade_data(trades_path),
    )
