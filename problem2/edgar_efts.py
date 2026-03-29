"""
SEC full-text search (efts.sec.gov) for 8-K — per brief / edgar_starter_snippet.

Supplements ``data.sec.gov/submissions`` with the same 8-K universe and merger-focused discovery.
"""

from __future__ import annotations

import re
from time import sleep
from typing import Dict, List, Set

import requests

from cik_map import SEC_HEADERS
from edgar_filings import FilingRow

EFT_URL = "https://efts.sec.gov/LATEST/search-index"

# Ticker pattern inside display_names: (DE), (GOOGL), etc.
_RE_TICKER = re.compile(r"\(([A-Z]{1,6})\)")


def _cik_int(src: dict) -> int:
    ciks = src.get("ciks") or ["0"]
    return int(str(ciks[0]).lstrip("0") or "0")


def _hit_matches_ticker(src: dict, ticker: str) -> bool:
    t = ticker.upper()
    for d in src.get("display_names") or []:
        if f"({t})" in (d or "").upper():
            return True
    return False


def _tickers_from_hit(src: dict) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for d in src.get("display_names") or []:
        for m in _RE_TICKER.finditer(d or ""):
            tk = m.group(1)
            if tk not in seen:
                seen.add(tk)
                out.append(tk)
    return out


def _items_str(src: dict) -> str:
    it = src.get("items") or []
    if isinstance(it, list):
        return ",".join(str(x) for x in it)
    return str(it) if it is not None else ""


def _hit_to_filing_row(hit: dict, sec_id: int, ticker: str) -> FilingRow:
    src = hit.get("_source", {})
    fd = str(src.get("file_date", ""))[:10]
    adsh = src.get("adsh", "")
    if not adsh and hit.get("_id"):
        # fallback: first segment before colon
        _id = hit.get("_id", "")
        adsh = _id.split(":")[0] if ":" in _id else _id
    cik = _cik_int(src)
    it = _items_str(src)
    desc = (src.get("file_description") or src.get("primaryDocDescription") or "").strip()
    if desc:
        it = f"{it}; {desc}" if it else desc
    return FilingRow(
        ticker=ticker,
        sec_id=sec_id,
        cik=cik,
        form="8-K",
        filing_date=fd,
        accession=adsh,
        primary_document="index.htm",
        items=it,
        report_date=str(src.get("period_ending") or fd)[:10],
    )


def search_8k_ticker(
    ticker: str,
    sec_id: int,
    startdt: str,
    enddt: str,
    sleep_s: float = 0.28,
) -> List[FilingRow]:
    """``q`` quoted ticker, ``forms=8-K`` — filter hits to display_names containing (TICKER)."""
    sleep(sleep_s)
    r = requests.get(
        EFT_URL,
        params={
            "q": f'"{ticker}"',
            "forms": "8-K",
            "dateRange": "custom",
            "startdt": startdt,
            "enddt": enddt,
        },
        headers=SEC_HEADERS,
        timeout=40,
    )
    r.raise_for_status()
    hits = r.json().get("hits", {}).get("hits", [])
    out: List[FilingRow] = []
    for h in hits:
        src = h.get("_source", {})
        if not _hit_matches_ticker(src, ticker):
            continue
        fd = str(src.get("file_date", ""))[:10]
        if fd < startdt or fd > enddt:
            continue
        out.append(_hit_to_filing_row(h, sec_id, ticker))
    return out


def search_8k_keyword(
    keyword: str,
    ticker_to_sec_id: Dict[str, int],
    startdt: str,
    enddt: str,
    sleep_s: float = 0.28,
) -> List[FilingRow]:
    """
    M&A-focused discovery: ``q=merger`` or ``acquisition`` with ``forms=8-K``;
    map hits to our universe via ``(TICKER)`` in display_names.
    """
    sleep(sleep_s)
    r = requests.get(
        EFT_URL,
        params={
            "q": keyword,
            "forms": "8-K",
            "dateRange": "custom",
            "startdt": startdt,
            "enddt": enddt,
        },
        headers=SEC_HEADERS,
        timeout=40,
    )
    r.raise_for_status()
    hits = r.json().get("hits", {}).get("hits", [])
    out: List[FilingRow] = []
    for h in hits:
        src = h.get("_source", {})
        fd = str(src.get("file_date", ""))[:10]
        if fd < startdt or fd > enddt:
            continue
        for tk in _tickers_from_hit(src):
            sid = ticker_to_sec_id.get(tk.upper())
            if sid is None:
                continue
            if not _hit_matches_ticker(src, tk):
                continue
            out.append(_hit_to_filing_row(h, sid, tk.upper()))
    return out


def efts_search_url_example(ticker: str, startdt: str, enddt: str) -> str:
    """Illustrative URL like the problem statement example (search index)."""
    return (
        f"{EFT_URL}?q=%22{ticker}%22&forms=8-K&dateRange=custom"
        f"&startdt={startdt}&enddt={enddt}"
    )
