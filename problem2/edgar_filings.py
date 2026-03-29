"""Fetch 8-K metadata from ``data.sec.gov/submissions`` (per brief: material 8-K events)."""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import List, Optional

import requests

from cik_map import SEC_HEADERS, cik_10

SUB_BASE = "https://data.sec.gov/submissions/CIK{}.json"


@dataclass
class FilingRow:
    ticker: str
    sec_id: int
    cik: int
    form: str
    filing_date: str
    accession: str
    primary_document: str
    items: str
    report_date: str


def archives_url(cik: int, accession: str, primary_document: str) -> str:
    acc = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{primary_document}"


def fetch_material_filings(
    ticker: str,
    sec_id: int,
    cik: int,
    start_date: str,
    end_date: str,
    sleep_s: float = 0.25,
) -> List[FilingRow]:
    """Recent 8-K filings only (Problem 2 spec)."""
    url = SUB_BASE.format(cik_10(cik))
    sleep(sleep_s)
    r = requests.get(url, headers=SEC_HEADERS, timeout=45)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    if not forms:
        return []
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    pdocs = recent.get("primaryDocument", [])
    items_list = recent.get("items", [])
    rep_dates = recent.get("reportDate", [])

    out: List[FilingRow] = []
    n = len(forms)
    seen = set()
    for i in range(n):
        form = forms[i]
        if form != "8-K":
            continue
        fd = dates[i]
        if fd < start_date or fd > end_date:
            continue
        acc = accs[i]
        key = (acc, form)
        if key in seen:
            continue
        seen.add(key)
        items = items_list[i] if i < len(items_list) else ""
        if isinstance(items, list):
            items = ",".join(str(x) for x in items)
        elif items is None:
            items = ""
        rd = rep_dates[i] if i < len(rep_dates) else fd
        pdoc = pdocs[i] if i < len(pdocs) else ""
        out.append(
            FilingRow(
                ticker=ticker,
                sec_id=sec_id,
                cik=cik,
                form=form,
                filing_date=fd,
                accession=acc,
                primary_document=pdoc or "primary_doc.htm",
                items=str(items) if items is not None else "",
                report_date=rd or fd,
            )
        )
    return out


def headline_for_filing(name: str, fr: FilingRow) -> str:
    it = fr.items.strip() or "see filing"
    return f"{name} {fr.form} ({fr.filing_date}) — items {it}"
