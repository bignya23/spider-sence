"""Fetch screener.in company data, parse HTML, and export to Excel + JSONL.

This module was moved under the MCP server folder so the screener MCP is
self-contained.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import sys
import time
from pathlib import Path
from typing import Any

import json
import math
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from orchestrator.output_paths import ensure_category_directory, get_companies_root, normalise_symbol


BASE_URL = "https://www.screener.in"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "companies"


def fetch_company_page(
    session: requests.Session,
    symbol: str,
    *,
    consolidated: bool = False,
) -> str:
    symbol = symbol.strip().upper()
    path = f"/company/{symbol}/consolidated/" if consolidated else f"/company/{symbol}/"
    url = f"{BASE_URL}{path}"
    last_exc = None
    for attempt in range(1, 4):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as he:
            if response.status_code == 404:
                logging.debug("Received 404 for %s", url)
                raise ValueError(f"Symbol not found or page missing: {symbol} (url: {url})") from he
            last_exc = he
            logging.warning(
                "HTTP error fetching %s (attempt %s): %s", url, attempt, he
            )
        except requests.exceptions.RequestException as rexc:
            last_exc = rexc
            logging.warning(
                "Network error fetching %s (attempt %s): %s", url, attempt, rexc
            )
        time.sleep(1 * attempt)
    raise RuntimeError(f"Failed to fetch company page for {symbol} after retries") from last_exc


def fetch_peers_html(session: requests.Session, company_id: str | int, symbol: str) -> str:
    url = f"{BASE_URL}/api/company/{company_id}/peers/"
    headers = {
        **DEFAULT_HEADERS,
        "Accept": "text/html",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE_URL}/company/{symbol}/",
    }
    last_exc = None
    for attempt in range(1, 4):
        try:
            response = session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            logging.warning("Error fetching peers for %s (attempt %s): %s", symbol, attempt, exc)
        time.sleep(0.5 * attempt)
    raise RuntimeError(f"Failed to fetch peers for company {company_id}") from last_exc


def _parse_data_table(table: Tag) -> tuple[list[str], list[list[str]]]:
    header_row = None
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
    if header_row is None:
        header_row = table.find("tr")

    headers = [
        cell.get_text(strip=True)
        for cell in header_row.find_all(["th", "td"])
    ] if header_row else []

    tbody = table.find("tbody")
    body_rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]

    rows: list[list[str]] = []
    for row in body_rows:
        cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
        if cells:
            rows.append(cells)

    return headers, rows


def _parse_key_value_table(table: Tag) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
        if len(cells) >= 2:
            pairs.append((cells[0], cells[1]))
    return pairs


def _section_table(section_id: str, soup: BeautifulSoup) -> tuple[list[str], list[list[str]]]:
    section = soup.find("section", id=section_id)
    if not section:
        return [], []

    table = section.find("table", class_="data-table")
    if not table:
        return [], []

    return _parse_data_table(table)


def _section_tables(section_id: str, soup: BeautifulSoup) -> list[tuple[list[str], list[list[str]]]]:
    section = soup.find("section", id=section_id)
    if not section:
        return []
    return [_parse_data_table(table) for table in section.find_all("table", class_="data-table")]


def _growth_tables(section_id: str, soup: BeautifulSoup) -> list[dict[str, Any]]:
    section = soup.find("section", id=section_id)
    if not section:
        return []

    growth_blocks: list[dict[str, Any]] = []
    for table in section.find_all("table"):
        if "data-table" in (table.get("class") or []):
            continue
        rows = table.find_all("tr")
        if not rows:
            continue
        title_cells = [cell.get_text(strip=True) for cell in rows[0].find_all(["td", "th"])]
        if len(title_cells) != 1:
            continue
        growth_blocks.append({"metric": title_cells[0], "periods": _parse_key_value_table(table)})
    return growth_blocks


def _parse_documents(soup: BeautifulSoup) -> dict[str, list[dict[str, str]]]:
    section = soup.find("section", id="documents")
    if not section:
        return {}
    documents: dict[str, list[dict[str, str]]] = {}
    for heading in section.find_all("h3"):
        category = heading.get_text(strip=True)
        listing = heading.find_next("ul")
        if not listing:
            continue
        items: list[dict[str, str]] = []
        for item in listing.find_all("li"):
            link = item.find("a")
            items.append({"title": item.get_text(strip=True), "url": link["href"] if link and link.has_attr("href") else ""})
        documents[category] = items
    return documents


def parse_company_html(html: str, *, peers_html: str | None = None) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    company_name = soup.find("h1")
    about_div = soup.select_one(".company-profile .about, .about")
    key_points_div = soup.select_one(".company-profile .sub:not(.about)")

    key_ratios: dict[str, str] = {}
    ratios_list = soup.find("ul", id="top-ratios")
    if ratios_list:
        for item in ratios_list.find_all("li"):
            name = item.find("span", class_="name")
            value = item.find("span", class_="number") or item.find("span", class_="value")
            if name and value:
                key_ratios[name.get_text(strip=True)] = value.get_text(strip=True)

    analysis = soup.find("section", id="analysis")
    pros: list[str] = []
    cons: list[str] = []
    if analysis:
        pros_block = analysis.find("div", class_="pros")
        cons_block = analysis.find("div", class_="cons")
        if pros_block:
            pros = [item.get_text(strip=True) for item in pros_block.find_all("li")]
        if cons_block:
            cons = [item.get_text(strip=True) for item in cons_block.find_all("li")]

    shareholding_tables = _section_tables("shareholding", soup)

    peers_headers: list[str] = []
    peers_rows: list[list[str]] = []
    if peers_html:
        peers_table = BeautifulSoup(peers_html, "html.parser").find("table", class_="data-table")
        if peers_table:
            peers_headers, peers_rows = _parse_data_table(peers_table)

    company_id_tag = soup.find(attrs={"data-company-id": True})
    company_id = company_id_tag.get("data-company-id") if company_id_tag else None

    return {
        "company_id": company_id,
        "company_name": company_name.get_text(strip=True) if company_name else "",
        "about": about_div.get_text(strip=True) if about_div else "",
        "key_points": key_points_div.get_text(strip=True) if key_points_div else "",
        "key_ratios": key_ratios,
        "pros": pros,
        "cons": cons,
        "quarterly_results": dict(zip(["headers", "rows"], _section_table("quarters", soup))),
        "profit_loss": dict(zip(["headers", "rows"], _section_table("profit-loss", soup))),
        "growth_metrics": _growth_tables("profit-loss", soup),
        "balance_sheet": dict(zip(["headers", "rows"], _section_table("balance-sheet", soup))),
        "cash_flow": dict(zip(["headers", "rows"], _section_table("cash-flow", soup))),
        "ratios": dict(zip(["headers", "rows"], _section_table("ratios", soup))),
        "shareholding": {
            "quarterly": {"headers": shareholding_tables[0][0] if shareholding_tables else [], "rows": shareholding_tables[0][1] if shareholding_tables else []},
            "yearly": {"headers": shareholding_tables[1][0] if len(shareholding_tables) > 1 else [], "rows": shareholding_tables[1][1] if len(shareholding_tables) > 1 else []},
        },
        "peers": {"headers": peers_headers, "rows": peers_rows},
        "documents": _parse_documents(soup),
    }


def _table_to_dataframe(headers: list[str], rows: list[list[str]]) -> pd.DataFrame:
    if not headers and not rows:
        return pd.DataFrame()
    metric_header = headers[0] if headers else "Metric"
    period_headers = headers[1:] if len(headers) > 1 else []
    data: dict[str, list[str]] = {metric_header: []}
    for header in period_headers:
        data[header] = []
    for row in rows:
        data[metric_header].append(row[0] if row else "")
        for index, header in enumerate(period_headers, start=1):
            data[header].append(row[index] if index < len(row) else "")
    return pd.DataFrame(data)


def export_to_excel(data: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overview_rows = [
        {"Field": "Company", "Value": data.get("company_name", "")},
        {"Field": "Company ID", "Value": data.get("company_id", "")},
        {"Field": "About", "Value": data.get("about", "")},
        {"Field": "Key Points", "Value": data.get("key_points", "")},
        {"Field": "Exported At (UTC)", "Value": datetime.now(timezone.utc).isoformat()},
    ]
    overview_rows.extend({"Field": name, "Value": value} for name, value in data.get("key_ratios", {}).items())

    pros_cons_rows: list[dict[str, str | int]] = []
    for index, item in enumerate(data.get("pros", []), start=1):
        pros_cons_rows.append({"Type": "Pros", "No.": index, "Text": item})
    for index, item in enumerate(data.get("cons", []), start=1):
        pros_cons_rows.append({"Type": "Cons", "No.": index, "Text": item})

    growth_rows = [
        {"Metric": block["metric"], "Period": period, "Value": value}
        for block in data.get("growth_metrics", [])
        for period, value in block.get("periods", [])
    ]

    document_rows = [
        {"Category": category, "Title": item.get("title", ""), "URL": item.get("url", "")}
        for category, items in data.get("documents", {}).items()
        for item in items
    ]

    sheets: list[tuple[str, pd.DataFrame]] = [
        ("Overview", pd.DataFrame(overview_rows)),
        ("Pros_Cons", pd.DataFrame(pros_cons_rows)),
        ("Quarterly_Results", _table_to_dataframe(**data["quarterly_results"])),
        ("Profit_Loss", _table_to_dataframe(**data["profit_loss"])),
        ("Growth_Metrics", pd.DataFrame(growth_rows)),
        ("Balance_Sheet", _table_to_dataframe(**data["balance_sheet"])),
        ("Cash_Flow", _table_to_dataframe(**data["cash_flow"])),
        ("Ratios", _table_to_dataframe(**data["ratios"])),
        ("Shareholding_Q", _table_to_dataframe(**data["shareholding"]["quarterly"])),
        ("Shareholding_Y", _table_to_dataframe(**data["shareholding"]["yearly"])),
        ("Peers", _table_to_dataframe(**data["peers"])),
        ("Documents", pd.DataFrame(document_rows)),
    ]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets:
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return output_path


def export_company(symbol: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    html = fetch_company_page(session, symbol)
    data = parse_company_html(html)

    company_id = data.get("company_id")
    if company_id:
        try:
            peers_html = fetch_peers_html(session, company_id, symbol)
            data = parse_company_html(html, peers_html=peers_html)
        except Exception:
            logging.info("Continuing without peers for %s", symbol)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = normalise_symbol(symbol)
    output_path = output_dir / f"{safe_symbol}.xlsx"
    return export_to_excel(data, output_path)


def _parse_number_for_export(x: Any) -> float | None:
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        if math.isfinite(x):
            return float(x)
        return None
    s = str(x).strip()
    if s == "" or s in {"-", "—", "NA", "N/A"}:
        return None
    if s.endswith("%"):
        try:
            return float(s.strip("%").replace(",", ""))
        except Exception:
            return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s2 = re.sub(r"[^0-9.\-]", "", s)
    if s2 in {"", ".", "-"}:
        return None
    try:
        val = float(s2)
        return -val if neg else val
    except Exception:
        return None


def write_jsonl_from_excel(excel_path: Path) -> Path:
    if not excel_path.exists():
        raise FileNotFoundError(excel_path)

    wb = pd.read_excel(excel_path, sheet_name=None)
    out_dir = excel_path.parent
    symbol = excel_path.stem
    jsonl_path = out_dir / f"{symbol.lower()}.jsonl"
    entry_id = 0

    with jsonl_path.open("w", encoding="utf8") as fh:
        for sheet_name, df in wb.items():
            df = pd.DataFrame(df)
            df.columns = [str(c).strip() for c in df.columns]
            cols = list(df.columns)
            if len(cols) >= 2:
                metric_col = cols[0]
                period_cols = cols[1:]
                melt = df.melt(id_vars=[metric_col], value_vars=period_cols, var_name="period", value_name="value")
                for _, row in melt.dropna(subset=["value"]).iterrows():
                    val = row["value"]
                    num = _parse_number_for_export(val)
                    text = f"{sheet_name} | {row[metric_col]} | {row['period']} = {num if num is not None else val}"
                    meta = {
                        "sheet": sheet_name,
                        "symbol": symbol,
                        "period": str(row["period"]),
                        "metric": str(row[metric_col]),
                    }
                    obj = {"id": f"{symbol}_{entry_id}", "text": text, "metadata": meta}
                    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    entry_id += 1
            else:
                for _, row in df.iterrows():
                    parts = [f"{k}: {v}" for k, v in row.items() if pd.notna(v) and v != ""]
                    if not parts:
                        continue
                    text = f"{sheet_name} | " + "; ".join(parts)
                    meta = {"sheet": sheet_name, "symbol": symbol}
                    obj = {"id": f"{symbol}_{entry_id}", "text": text, "metadata": meta}
                    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    entry_id += 1

    return jsonl_path
