"""Ticker → CIK using SEC ``company_tickers`` (cached)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import requests

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "company_tickers.json"
SEC_HEADERS = {
    "User-Agent": "BITS_AerialViewHackathon research (student project) you@example.edu",
}


def load_ticker_cik_map() -> Dict[str, int]:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS,
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(raw, f)
    return {v["ticker"]: int(v["cik_str"]) for v in raw.values()}


def cik_for_ticker(ticker: str, mapping: Dict[str, int]) -> Optional[int]:
    return mapping.get(ticker.upper())


def cik_10(cik: int) -> str:
    return str(cik).zfill(10)
