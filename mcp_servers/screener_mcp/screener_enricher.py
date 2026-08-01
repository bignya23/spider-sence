#!/usr/bin/env python3
"""Cleaner/Clerk: compute derived fundamentals from JSONL and save JSON payload.

Moved under `mcp_servers/screener_mcp/` so the Screener MCP server is
self-contained.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import re
import math
from typing import Any, Dict, Optional
import difflib

from orchestrator.output_paths import get_category_directory, get_companies_root, normalise_symbol


def parse_number(s: Any) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        if math.isfinite(s):
            return float(s)
        return None
    text = str(s).strip()
    if text == "" or text.upper() in {"NA", "N/A", "-"}:
        return None
    if text.endswith("%"):
        try:
            return float(text.rstrip("%").replace(",", ""))
        except Exception:
            return None
    neg = False
    if text.startswith("(") and text.endswith(")"):
        neg = True
        text = text[1:-1]
    m = re.search(r"-?[0-9,.]+(?:\.[0-9]+)?", text)
    if not m:
        return None
    tok = m.group(0).replace(",", "")
    try:
        val = float(tok)
        return -val if neg else val
    except Exception:
        return None


def parse_qualitative_metric(text: str) -> Dict[str, Any]:
    period_years = None
    value_pct = None
    m = re.search(r"(\d+)\s*(?:-?year|yr|yrs)\b", text, re.IGNORECASE)
    if m:
        try:
            period_years = int(m.group(1))
        except Exception:
            period_years = None
    v = re.search(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if v:
        try:
            value_pct = float(v.group(1))
        except Exception:
            value_pct = None
    return {"text": text, "period_years": period_years, "value_pct": value_pct}


def parse_high_low(raw: Any) -> tuple[Optional[float], Optional[float]]:
    if raw is None:
        return None, None
    text = str(raw)
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", text.replace(",", ""))
    try:
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        if len(nums) == 1:
            return float(nums[0]), None
    except Exception:
        return None, None
    return None, None


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                logging.warning("Skipping invalid JSON line: %s", line[:200])
    return items


def build_lookup(items: list[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    overview: Dict[str, Any] = {}

    for it in items:
        meta = it.get("metadata", {}) or {}
        sheet = meta.get("sheet") or (meta.get("sheet", "") if meta else "")
        text = it.get("text", "")
        metric = meta.get("metric")
        period = meta.get("period")

        if not metric or not period:
            parts = text.split("|")
            if len(parts) >= 3:
                metric = metric or parts[1].strip()
                last = parts[-1].strip()
                if "=" in last:
                    left, right = last.split("=", 1)
                    if re.search(r"[A-Za-z]{3}\s?\d{4}|TTM|Mar|Jun|Dec", left):
                        period = period or left.strip()
                else:
                    period = period or parts[2].strip()

        val = None
        if "value" in meta:
            val = meta.get("value")
        if val is None:
            m = re.search(r"=\s*([^;]+)$", text)
            if m:
                val = m.group(1).strip()
            else:
                m2 = re.search(r"([^:;]+):\s*([^;]+)", text)
                if m2:
                    val = m2.group(2).strip()

        num = parse_number(val) if val is not None else None

        if sheet and metric:
            lookup.setdefault(sheet, {}).setdefault(metric, {})[period or ""] = num if num is not None else val
        else:
            if sheet == "Overview" or "Overview" in (sheet or ""):
                pairs = re.findall(r"([^:;]+):\s*([^;\n]+)", text)
                for k, v in pairs:
                    overview[k.strip()] = parse_number(v.strip()) if parse_number(v.strip()) is not None else v.strip()

    lookup["_overview"] = overview
    lookup["_qualitative"] = {"pros": [], "cons": []}

    for it in items:
        meta = it.get("metadata", {}) or {}
        if meta.get("sheet") == "Pros_Cons" and meta.get("period") == "Text":
            val = it.get("text", "").split("=")[-1].strip()
            if val and parse_number(val) is not None and re.fullmatch(r"[\d.,\s%+-]+", val):
                continue
            parsed_q = parse_qualitative_metric(val)
            if meta.get("metric") == "Pros":
                lookup["_qualitative"]["pros"].append(parsed_q)
            elif meta.get("metric") == "Cons":
                lookup["_qualitative"]["cons"].append(parsed_q)

    return lookup


def _normalize_period(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    s = str(p).strip()
    s = s.replace("-", " ").replace("/", " ")
    if s.upper() in {"TTM", "TTM."}:
        return "TTM"
    m = re.search(r"([A-Za-z]{3,9})\s*(\d{4})", s)
    if m:
        mon = m.group(1)[:3].title()
        yr = m.group(2)
        return f"{mon} {yr}"
    m2 = re.search(r"(\d{4})", s)
    if m2:
        return f"{m2.group(1)}"
    return s


def get_value(lookup: Dict[str, Any], sheet: str, metric: str, period: Optional[str]) -> Optional[float]:
    s = lookup.get(sheet, {})
    if not s:
        return None

    m = s.get(metric)
    if not m:
        keys = list(s.keys())
        if keys:
            matches = difflib.get_close_matches(metric, keys, n=1, cutoff=0.6)
            if matches:
                logging.info("Fuzzy-matched metric %r -> %r (sheet=%s)", metric, matches[0], sheet)
                m = s.get(matches[0])
        if not m:
            for k in keys:
                if metric.lower() in k.lower():
                    logging.info("Substring-matched metric %r -> %r (sheet=%s)", metric, k, sheet)
                    m = s.get(k)
                    break
    if not m:
        return None

    if period is None:
        for p, v in reversed(list(m.items())):
            if v is not None:
                return v if isinstance(v, (int, float)) else parse_number(v)
        return None

    norm = _normalize_period(period)

    val = m.get(period) if period in m else None
    if val is None and norm is not None:
        for p_key in m.keys():
            if _normalize_period(p_key) == norm:
                val = m.get(p_key)
                break

    if val is None:
        for p_key, v in m.items():
            if p_key and norm and norm.lower() in str(p_key).lower():
                return v if isinstance(v, (int, float)) else parse_number(v)

        period_keys = [str(k) for k in m.keys()]
        matches = difflib.get_close_matches(str(period), period_keys, n=1, cutoff=0.6)
        if matches:
            logging.info("Fuzzy-matched period %r -> %r for metric %r (sheet=%s)", period, matches[0], metric, sheet)
            val = m.get(matches[0])

    return val if isinstance(val, (int, float)) else parse_number(val)


def compute_derivatives(lookup: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    # helper
    def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
        try:
            if a is None or b is None:
                return None
            if b == 0:
                return None
            return float(a) / float(b)
        except Exception:
            return None

    # Pillar A
    sales_ttm = get_value(lookup, "Profit_Loss", "Sales+", "TTM")
    sales_2019 = get_value(lookup, "Profit_Loss", "Sales+", "Mar 2019")
    net_ttm = get_value(lookup, "Profit_Loss", "Net Profit+", "TTM")
    net_2019 = get_value(lookup, "Profit_Loss", "Net Profit+", "Mar 2019")

    def cagr(start, end, years: float):
        try:
            if start is None or end is None:
                return None
            if start <= 0 or end <= 0:
                return None
            return (end / start) ** (1.0 / years) - 1.0
        except Exception:
            return None

    # compute elapsed years from Mar 31, 2019 to today (approximate)
    try:
        start_date = datetime(2019, 3, 31).date()
        years_elapsed = (datetime.utcnow().date() - start_date).days / 365.25
        if years_elapsed <= 0:
            years_elapsed = 7.5
    except Exception:
        years_elapsed = 7.5

    sales_cagr = cagr(sales_2019, sales_ttm, years_elapsed)
    net_cagr = cagr(net_2019, net_ttm, years_elapsed)

    sales_pct = round(sales_cagr * 100, 2) if sales_cagr is not None else None
    net_pct = round(net_cagr * 100, 2) if net_cagr is not None else None
    growth_alignment_status = None
    if sales_pct is not None and net_pct is not None:
        if net_pct is not None and net_pct < 0:
            growth_alignment_status = "PROFIT_DECLINING"
        elif sales_pct > net_pct * 1.1:
            growth_alignment_status = "SALES_OUTPACING_PROFIT"
        elif net_pct > sales_pct * 1.1:
            growth_alignment_status = "PROFIT_OUTPACING_SALES"
        else:
            growth_alignment_status = "IN_LINE"

    # Pillar B
    # stock PE from overview
    overview = lookup.get("_overview", {})
    pe = None
    # try direct overview fields
    for key in ("Stock P/E", "P/E", "Stock P/E (TTM)", "Stock P/E (Value)"):
        if key in overview:
            parsed = parse_number(overview[key])
            pe = parsed if parsed is not None else overview[key]
            break
    # warn if exact 2019 baseline not found for CAGR inputs
    if "Mar 2019" not in lookup.get("Profit_Loss", {}).get("Sales+", {}):
        logging.warning(
            "Exact period Mar 2019 not found for Sales+; CAGR may use fuzzy-matched period"
        )
    # fallback to sheet search
    if pe is None:
        pe = get_value(lookup, "Overview", "Stock P/E", "Value")

    dividend_yield = None
    for key in ("Dividend yield", "Dividend Yield", "Dividend yield %", "Dividend"):
        if key in overview:
            parsed = parse_number(overview[key])
            dividend_yield = parsed if parsed is not None else overview[key]
            break
    # fallback to sheet search
    if dividend_yield is None:
        dividend_yield = get_value(lookup, "Overview", "Dividend Yield", "Value")

    # Raw baseline inputs (absolute data points for audit trail)
    current_price = get_value(lookup, "Overview", "Current Price", "Value")
    raw_high_low = lookup.get("_overview", {}).get("High / Low") or get_value(
        lookup, "Overview", "High / Low", "Value"
    )
    week_52_high, week_52_low = parse_high_low(raw_high_low)
    market_cap = get_value(lookup, "Overview", "Market Cap", "Value")
    book_value = get_value(lookup, "Overview", "Book Value", "Value")
    face_value = get_value(lookup, "Overview", "Face Value", "Value")

    peg_proxy = None
    try:
        if pe is not None and isinstance(pe, (int, float)) and pe > 0 and net_pct is not None and net_pct != 0:
            peg_proxy = float(pe) / float(net_pct)
    except Exception:
        peg_proxy = None

    pricing_premium_bracket = None
    if peg_proxy is not None:
        if peg_proxy < 0:
            pricing_premium_bracket = "NEGATIVE_GROWTH_DISTORTION"
        elif peg_proxy > 20:
            pricing_premium_bracket = "EXTREME_GROWTH_PREMIUM"
        elif peg_proxy > 10:
            pricing_premium_bracket = "HIGH_GROWTH_PREMIUM"
        elif peg_proxy > 5:
            pricing_premium_bracket = "MODERATE_GROWTH_PREMIUM"
        else:
            pricing_premium_bracket = "FAIR_VALUE"
    elif pe is not None and isinstance(pe, (int, float)) and pe <= 0:
        pricing_premium_bracket = "LOSS_MAKING_UNDEFINED"

    # Pillar C
    borrowings = get_value(lookup, "Balance_Sheet", "Borrowings+", "Mar 2026")
    reserves = get_value(lookup, "Balance_Sheet", "Reserves", "Mar 2026")
    if borrowings is None:
        debt_to_reserves = None
    else:
        if reserves in (None, 0):
            debt_to_reserves = None
        else:
            try:
                debt_to_reserves = float(borrowings) / float(reserves)
            except Exception:
                debt_to_reserves = None

    roce = get_value(lookup, "Overview", "ROCE", "Value")
    if roce is None:
        roce = get_value(lookup, "Ratios", "ROCE %", "Mar 2026")
    roe = get_value(lookup, "Overview", "ROE", "Value")
    if roe is None:
        roe = get_value(lookup, "Ratios", "ROE %", "Mar 2026")
    cap_prod_roce = round(float(roce), 2) if roce is not None else None
    cap_prod_roe = round(float(roe), 2) if roe is not None else None

    balance_sheet_safety_status = None
    if debt_to_reserves is not None:
        if debt_to_reserves < 0.5:
            balance_sheet_safety_status = "SAFE_FORTRESS"
        elif debt_to_reserves < 1.0:
            balance_sheet_safety_status = "WATCHED"
        else:
            balance_sheet_safety_status = "HIGH_LEVERAGE"

    # Pillar D
    cfo = get_value(lookup, "Cash_Flow", "Cash from Operating Activity+", "Mar 2026")
    fcf = get_value(lookup, "Cash_Flow", "Free Cash Flow", "Mar 2026")
    operating_profit = get_value(lookup, "Profit_Loss", "Operating Profit", "Mar 2026")

    cash_conversion_eff = None
    fcf_to_cfo_eff = None
    if cfo is not None and operating_profit is not None and operating_profit != 0:
        cash_conversion_eff = (float(cfo) / float(operating_profit)) * 100
    if fcf is not None and cfo is not None and cfo != 0:
        fcf_to_cfo_eff = (float(fcf) / float(cfo)) * 100

    # Pillar E
    def safe_delta(metric_name: str) -> Optional[float]:
        a = get_value(lookup, "Shareholding_Q", metric_name, "Jun 2026")
        b = get_value(lookup, "Shareholding_Q", metric_name, "Mar 2026")
        if a is None or b is None:
            return None
        try:
            return round(float(a) - float(b), 2)
        except Exception:
            return None

    promoter_delta = safe_delta("Promoters+")
    fii_delta = safe_delta("FIIs+")
    dii_delta = safe_delta("DIIs+")

    institutional_flow_pattern = None
    if promoter_delta is not None and fii_delta is not None and dii_delta is not None:
        if promoter_delta > 0 and dii_delta > 0 and fii_delta < 0:
            institutional_flow_pattern = "DOMESTIC_ABSORBING_FOREIGN_OUTFLOW"
        elif fii_delta > 0 and dii_delta < 0:
            institutional_flow_pattern = "FOREIGN_INFLOW"
        else:
            institutional_flow_pattern = "MIXED"

    # Qualitative
    pros = lookup.get("_qualitative", {}).get("pros", [])
    cons = lookup.get("_qualitative", {}).get("cons", [])

    result = {
        "symbol": symbol.upper(),
        "data_as_of": datetime.utcnow().date().isoformat(),
        "raw_baseline_inputs": {
            "current_price": float(current_price) if current_price is not None else None,
            "high_low": float(week_52_high) if week_52_high is not None else None,
            "market_cap": float(market_cap) if market_cap is not None else None,
            "book_value": float(book_value) if book_value is not None else None,
            "face_value": float(face_value) if face_value is not None else None,
            "total_borrowings_mar2026": float(borrowings) if borrowings is not None else None,
            "total_reserves_mar2026": float(reserves) if reserves is not None else None,
            "operating_cash_flow_mar2026": float(cfo) if cfo is not None else None,
            "free_cash_flow_mar2026": float(fcf) if fcf is not None else None,
            "operating_profit_mar2026": float(operating_profit) if operating_profit is not None else None,
        },
        "calculated_fundamental_derivatives": {
            "growth_velocity_cagr": {
                "sales_7year_trajectory_pct": sales_pct,
                "net_profit_7year_trajectory_pct": net_pct,
                "growth_alignment_status": growth_alignment_status,
            },
            "valuation_realism": {
                "static_pe_ratio": float(pe) if pe is not None else None,
                "peg_ratio_proxy": round(peg_proxy, 2) if peg_proxy is not None else None,
                "dividend_yield_pct": float(dividend_yield) if dividend_yield is not None else None,
                "pricing_premium_bracket": pricing_premium_bracket,
            },
            "solvency_and_leverage": {
                "debt_to_reserves_leverage": round(debt_to_reserves, 2)
                if isinstance(debt_to_reserves, (int, float))
                else debt_to_reserves,
                "capital_productivity_roce_pct": cap_prod_roce,
                "capital_productivity_roe_pct": cap_prod_roe,
                "balance_sheet_safety_status": balance_sheet_safety_status,
            },
            "liquidity_earnings_quality": {
                "cash_conversion_efficiency_pct": round(cash_conversion_eff, 2)
                if cash_conversion_eff is not None
                else None,
                "fcf_to_cfo_efficiency_pct": round(fcf_to_cfo_eff, 2)
                if fcf_to_cfo_eff is not None
                else None,
                "accounting_cash_realization_status": (
                    "HEALTHY_LIQUIDITY_REALIZATION"
                    if cash_conversion_eff and cash_conversion_eff > 50
                    else "POOR_REALIZATION"
                    if cash_conversion_eff is not None
                    else None
                ),
            },
            "smart_money_momentum_qoq": {
                "promoter_stake_delta_pct": promoter_delta,
                "fii_stake_delta_pct": fii_delta,
                "dii_stake_delta_pct": dii_delta,
                "institutional_flow_pattern": institutional_flow_pattern,
            },
        },
        "extracted_qualitative_context": {
            "verified_pros": pros,
            "verified_cons": cons,
        },
        "field_definitions": {
            "raw_baseline_inputs": {
                "current_price": "Closing market price of a single share on the date of extraction.",
                "high_low": "Highest and lowest transaction prices recorded over the trailing 52-week period.",
                "market_cap": "Total market capitalization; aggregate market value of all outstanding shares.",
                "book_value": "Net asset value per share; total assets minus liabilities divided by shares outstanding.",
                "face_value": "Nominal or par value of a share as stated in the corporate charter.",
                "total_borrowings_mar2026": "Total long-term and short-term debt obligations on the balance sheet as of Mar 2026.",
                "total_reserves_mar2026": "Accumulated retained earnings and surplus reserves as of Mar 2026.",
                "operating_cash_flow_mar2026": "Net cash generated from core business operations as of Mar 2026.",
                "free_cash_flow_mar2026": "Cash remaining after CapEx deductions from operating cash flow as of Mar 2026.",
                "operating_profit_mar2026": "Core operational earnings before interest, taxes, or depreciation as of Mar 2026.",
            },
            "growth_velocity_cagr": {
                "sales_7year_trajectory_pct": "Compound Annual Growth Rate (CAGR) of revenue over a 7-year timeline.",
                "net_profit_7year_trajectory_pct": "CAGR of net bottom-line profit over a 7-year timeline.",
                "growth_alignment_status": "Flag identifying margin trends (SALES_OUTPACING_PROFIT, PROFIT_OUTPACING_SALES, or IN_LINE).",
            },
            "valuation_realism": {
                "static_pe_ratio": "Price-to-Earnings multiple; market premium relative to current earnings per share.",
                "peg_ratio_proxy": "P/E divided by 7-Year Net Profit CAGR; normalizes valuation against growth.",
                "dividend_yield_pct": "Annual dividend income per share divided by current stock price.",
                "pricing_premium_bracket": "Categorization of growth pricing boundaries (EXTREME/HIGH/MODERATE_GROWTH_PREMIUM or FAIR_VALUE).",
            },
            "solvency_and_leverage": {
                "debt_to_reserves_leverage": "Absolute borrowings divided by reserves; evaluates financial resilience and default risk.",
                "capital_productivity_roce_pct": "Return on Capital Employed; efficiency of all deployed capital in generating profit.",
                "capital_productivity_roe_pct": "Return on Equity; compounding efficiency of shareholders equity funds.",
                "balance_sheet_safety_status": "Safety bracket based on leverage (SAFE_FORTRESS, WATCHED, or HIGH_LEVERAGE).",
            },
            "liquidity_earnings_quality": {
                "cash_conversion_efficiency_pct": "Operating cash flow divided by operating profit; verifies paper profits materialize as cash.",
                "fcf_to_cfo_efficiency_pct": "Share of operating cash flows converted into liquid free cash flow.",
                "accounting_cash_realization_status": "Earnings quality classification (HEALTHY_LIQUIDITY_REALIZATION or POOR_REALIZATION).",
            },
            "smart_money_momentum_qoq": {
                "promoter_stake_delta_pct": "Quarter-over-quarter ownership percentage shift by founders/insiders.",
                "fii_stake_delta_pct": "QoQ ownership percentage shift by Foreign Institutional Investors.",
                "dii_stake_delta_pct": "QoQ ownership percentage shift by Domestic Institutional Investors.",
                "institutional_flow_pattern": "Flag identifying supply absorption behavior in the public float.",
            },
            "extracted_qualitative_context": {
                "verified_pros": "Array of corporate strengths algorithmically verified by Screener parameters.",
                "verified_cons": "Array of balance sheet vulnerabilities algorithmically highlighted by Screener parameters.",
            },
        },
    }

    return result


def save_result(symbol: str, result: Dict[str, Any], fundamental_dir: Path) -> Path:
    fundamental_dir = Path(fundamental_dir)
    fundamental_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = normalise_symbol(symbol)
    out_path = fundamental_dir / f"{safe_symbol}_derived.json"
    import json as _json

    out_path.write_text(_json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute derived fundamentals from JSONL")
    parser.add_argument("symbol", nargs="?", default="RELIANCE")
    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    root = get_companies_root(Path(__file__).resolve().parents[2] / "companies")
    fundamental_dir, _ts = get_category_directory(symbol, "fundamental", root=root, run_timestamp=None)
    safe_symbol = normalise_symbol(symbol)

    jsonl1 = Path(fundamental_dir) / f"{safe_symbol.lower()}.jsonl"
    jsonl2 = Path(fundamental_dir) / f"{safe_symbol}.jsonl"
    jpath = jsonl1 if jsonl1.exists() else jsonl2

    items = load_jsonl(jpath)
    lookup = build_lookup(items)
    result = compute_derivatives(lookup, symbol)
    outp = save_result(symbol, result, fundamental_dir)
    print(str(outp))
