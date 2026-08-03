"""nse_scraper.py

Fetches corporate announcements from BSE India's public API.
No API key required.

BSE provides a public REST endpoint that returns JSON announcements
for a given scrip code.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

logger = logging.getLogger("news_mcp.bse")

# BSE public API for announcements
_BSE_ANN_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
_BSE_SEARCH_URL = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    ),
    "Referer": "https://www.bseindia.com/",
    "Accept": "application/json, text/plain, */*",
}

# Announcement types to flag as significant
SIGNIFICANT_TYPES = {
    "Board Meeting Results", "Outcome Of Board Meeting", "Financial Results",
    "Shareholding Pattern", "Cessation", "Appointment",
    "Change in Management", "Auditors Report", "Credit Rating",
    "Pledging of Shares", "Release of Pledge", "Fund Raising",
    "Merger/Amalgamation", "Acquisition", "Divestiture", "Legal Proceedings",
    "Buyback", "Dividend", "Bonus", "Stock Split", "Rights Issue",
}


def _symbol_to_bse_code(symbol: str) -> str | None:
    """
    Try to extract a BSE scrip code from an NSE Fyers symbol.
    The Fyers symbol looks like 'NSE:RELIANCE-EQ'.
    We return None if we can't determine the scrip code — the caller
    will fall back to a name-based search.
    """
    # TODO: A proper symbol→BSEcode mapping table should be maintained.
    # For now, return None and rely on name search.
    return None


def _search_bse_scrip(company_name: str) -> str | None:
    """Search BSE for a scrip code by company name. Returns the first match."""
    try:
        resp = httpx.get(
            _BSE_SEARCH_URL,
            params={"Type": "EQ", "flag": "0"},
            headers=_HEADERS,
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        scrips = data if isinstance(data, list) else data.get("Table", [])
        name_lower = company_name.lower()
        for scrip in scrips:
            sname = (scrip.get("Issuer_Name") or scrip.get("LONG_NAME") or "").lower()
            if name_lower in sname or sname in name_lower:
                return str(scrip.get("SCRIP_CD") or scrip.get("scripcode") or "")
    except Exception as exc:
        logger.warning("BSE scrip search failed: %s", exc)
    return None


def fetch_corporate_disclosures(
    symbol: str,
    company_name: str = "",
    days: int = 30,
) -> list[dict]:
    """
    Fetch recent corporate announcements from BSE India.

    Args:
        symbol: Fyers symbol like 'NSE:RELIANCE-EQ'.
        company_name: Human name like 'Reliance Industries' (helps BSE lookup).
        days: How many days back to look.

    Returns:
        List of Disclosure dicts.
    """
    # Derive a search name from symbol if company_name not given
    if not company_name:
        # e.g. "NSE:RELIANCE-EQ" → "RELIANCE"
        company_name = symbol.split(":")[1].split("-")[0].replace("_", " ")

    scrip_code = _symbol_to_bse_code(symbol) or _search_bse_scrip(company_name)
    if not scrip_code:
        logger.info("Could not resolve BSE scrip code for %s, skipping disclosures", symbol)
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")
    to_date = datetime.now(timezone.utc).strftime("%Y%m%d")

    try:
        resp = httpx.get(
            _BSE_ANN_URL,
            params={
                "pageno": 1,
                "strCat": "-1",
                "strPrevDate": from_date,
                "strScrip": scrip_code,
                "strSearch": "P",
                "strToDate": to_date,
                "strType": "C",
                "subcategory": "-1",
            },
            headers=_HEADERS,
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning("BSE API returned %d for scrip %s", resp.status_code, scrip_code)
            return []

        data = resp.json()
        announcements = data.get("Table", [])
    except Exception as exc:
        logger.warning("BSE API fetch failed: %s", exc)
        return []

    disclosures = []
    for ann in announcements:
        ann_type = ann.get("CATEGORYNAME") or ann.get("TypeOfAnn") or "Unknown"
        subject = ann.get("HEADLINE") or ann.get("SLONGNAME") or ""
        date_str = ann.get("NEWS_DT") or ann.get("DT_TM") or ""
        attach_url = ann.get("ATTACHMENTNAME") or ""

        disclosures.append({
            "type": ann_type,
            "subject": subject,
            "date": date_str,
            "attachment_url": f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attach_url}"
            if attach_url else "",
            "is_significant": any(
                sig.lower() in ann_type.lower() or sig.lower() in subject.lower()
                for sig in SIGNIFICANT_TYPES
            ),
        })

    return disclosures
