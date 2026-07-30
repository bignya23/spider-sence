#!/usr/bin/env python3
"""Cleaner/Clerk: compute derived fundamentals from JSONL and save JSON payload.

Usage: python3 screener_enricher.py RELIANCE
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


def parse_number(s: Any) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        if math.isfinite(s):
            return float(s)
        return None
    text = str(s).strip()
    if text == '' or text.upper() in {'NA', 'N/A', '-'}:
        return None
    # percent
    if text.endswith('%'):
        try:
            return float(text.rstrip('%').replace(',', ''))
        except Exception:
            return None
    # parentheses negative
    neg = False
    if text.startswith('(') and text.endswith(')'):
        neg = True
        text = text[1:-1]
    # extract first numeric-like token
    m = re.search(r"-?[0-9,.]+(?:\.[0-9]+)?", text)
    if not m:
        return None
    tok = m.group(0).replace(',', '')
    try:
        val = float(tok)
        return -val if neg else val
    except Exception:
        return None


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    items = []
    with path.open('r', encoding='utf8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                logging.warning('Skipping invalid JSON line: %s', line[:200])
    return items


def build_lookup(items: list[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    # structure: lookup[sheet][metric][period] = raw_value
    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    overview: Dict[str, Any] = {}
    for it in items:
        meta = it.get('metadata', {}) or {}
        sheet = meta.get('sheet') or (meta.get('sheet', '') if meta else '')
        text = it.get('text', '')
        metric = meta.get('metric')
        period = meta.get('period')

        # fallback: try to parse pattern like "Metric | A | Mar 2026 = 12345"
        if not metric or not period:
            # try to heuristically split
            parts = text.split('|')
            if len(parts) >= 3:
                metric = metric or parts[1].strip()
                # last part may contain "= value" or "period = value"
                last = parts[-1].strip()
                if '=' in last:
                    left, right = last.split('=', 1)
                    # if left looks like period
                    if re.search(r'[A-Za-z]{3}\s?\d{4}|TTM|Mar|Jun|Dec', left):
                        period = period or left.strip()
                else:
                    # maybe middle part is period
                    period = period or parts[2].strip()

        # extract numeric from text
        val = None
        # try metadata-stored value (some generators may include it)
        if 'value' in meta:
            val = meta.get('value')
        if val is None:
            # try parse after '='
            m = re.search(r"=\s*([^;]+)$", text)
            if m:
                val = m.group(1).strip()
            else:
                # try "Key: Value" patterns
                m2 = re.search(r"([^:;]+):\s*([^;]+)", text)
                if m2:
                    # use second group
                    val = m2.group(2).strip()

        num = parse_number(val) if val is not None else None

        if sheet and metric:
            lookup.setdefault(sheet, {}).setdefault(metric, {})[period or ''] = num if num is not None else val
        else:
            # save into overview-like
            if sheet == 'Overview' or 'Overview' in (sheet or ''):
                # try to find "Field: Value" pairs
                pairs = re.findall(r"([^:;]+):\s*([^;\n]+)", text)
                for k, v in pairs:
                    overview[k.strip()] = parse_number(v.strip()) if parse_number(v.strip()) is not None else v.strip()
    # attach overview
    lookup['_overview'] = overview
    return lookup


def _normalize_period(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    s = str(p).strip()
    s = s.replace('-', ' ').replace('/', ' ')
    # common TTM
    if s.upper() in {'TTM', 'TTM.'}:
        return 'TTM'
    # month-year patterns like 'Mar 2026' or 'Mar2026'
    m = re.search(r"([A-Za-z]{3,9})\s*(\d{4})", s)
    if m:
        mon = m.group(1)[:3].title()
        yr = m.group(2)
        return f"{mon} {yr}"
    # year-only
    m2 = re.search(r"(\d{4})", s)
    if m2:
        return f"{m2.group(1)}"
    return s


def get_value(lookup: Dict[str, Any], sheet: str, metric: str, period: Optional[str]) -> Optional[float]:
    s = lookup.get(sheet, {})
    if not s:
        return None
    # exact metric match
    m = s.get(metric)
    if not m:
        # try fuzzy match using difflib
        keys = list(s.keys())
        if keys:
            matches = difflib.get_close_matches(metric, keys, n=1, cutoff=0.6)
            if matches:
                m = s.get(matches[0])
        # substring fallback
        if not m:
            for k in keys:
                if metric.lower() in k.lower():
                    m = s.get(k)
                    break
    if not m:
        return None
    if period is None:
        # pick most recent non-null value
        for p, v in reversed(list(m.items())):
            if v is not None:
                return v if isinstance(v, (int, float)) else parse_number(v)
        return None

    # normalize requested period and try exact
    norm = _normalize_period(period)
    # try exact
    val = m.get(period) if period in m else None
    if val is None and norm is not None:
        # try normalized keys
        for p_key in m.keys():
            if _normalize_period(p_key) == norm:
                val = m.get(p_key)
                break
    if val is None:
        # fuzzy period by substring
        for p_key, v in m.items():
            if p_key and norm and norm.lower() in str(p_key).lower():
                return v if isinstance(v, (int, float)) else parse_number(v)
        # last resort: use close matches on period strings
        period_keys = [str(k) for k in m.keys()]
        matches = difflib.get_close_matches(str(period), period_keys, n=1, cutoff=0.6)
        if matches:
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
    sales_ttm = get_value(lookup, 'Profit_Loss', 'Sales+', 'TTM')
    sales_2019 = get_value(lookup, 'Profit_Loss', 'Sales+', 'Mar 2019')
    net_ttm = get_value(lookup, 'Profit_Loss', 'Net Profit+', 'TTM')
    net_2019 = get_value(lookup, 'Profit_Loss', 'Net Profit+', 'Mar 2019')

    def cagr(start, end, years=7.5):
        try:
            if start is None or end is None:
                return None
            if start <= 0 or end <= 0:
                return None
            return (end / start) ** (1.0 / years) - 1.0
        except Exception:
            return None

    sales_cagr = cagr(sales_2019, sales_ttm)
    net_cagr = cagr(net_2019, net_ttm)

    sales_pct = round(sales_cagr * 100, 2) if sales_cagr is not None else None
    net_pct = round(net_cagr * 100, 2) if net_cagr is not None else None
    growth_alignment_status = None
    if sales_pct is not None and net_pct is not None:
        if sales_pct > net_pct * 1.1:
            growth_alignment_status = 'SALES_OUTPACING_PROFIT'
        elif net_pct > sales_pct * 1.1:
            growth_alignment_status = 'PROFIT_OUTPACING_SALES'
        else:
            growth_alignment_status = 'IN_LINE'

    # Pillar B
    # stock PE from overview
    overview = lookup.get('_overview', {})
    pe = None
    # try direct overview fields
    for key in ('Stock P/E', 'P/E', 'Stock P/E (TTM)', 'Stock P/E (Value)'):
        if key in overview:
            pe = parse_number(overview[key]) or overview[key]
            break
    # fallback to sheet search
    if pe is None:
        pe = get_value(lookup, 'Overview', 'Stock P/E', 'Value')

    dividend_yield = None
    for key in ('Dividend yield', 'Dividend Yield', 'Dividend yield %', 'Dividend'): 
        if key in overview:
            dividend_yield = parse_number(overview[key]) or overview[key]
            break

    peg_proxy = None
    if pe is not None and net_pct is not None and net_pct != 0:
        try:
            peg_proxy = float(pe) / float(net_pct)
        except Exception:
            peg_proxy = None

    pricing_premium_bracket = None
    if peg_proxy is not None:
        if peg_proxy > 20:
            pricing_premium_bracket = 'EXTREME_GROWTH_PREMIUM'
        elif peg_proxy > 10:
            pricing_premium_bracket = 'HIGH_GROWTH_PREMIUM'
        elif peg_proxy > 5:
            pricing_premium_bracket = 'MODERATE_GROWTH_PREMIUM'
        else:
            pricing_premium_bracket = 'FAIR_VALUE'

    # Pillar C
    borrowings = get_value(lookup, 'Balance_Sheet', 'Borrowings+', 'Mar 2026')
    reserves = get_value(lookup, 'Balance_Sheet', 'Reserves', 'Mar 2026')
    if borrowings is None:
        debt_to_reserves = 0.0
    else:
        debt_to_reserves = None
        if reserves not in (None, 0):
            try:
                debt_to_reserves = float(borrowings) / float(reserves)
            except Exception:
                debt_to_reserves = None

    roce = get_value(lookup, 'Ratios', 'ROCE', 'Mar 2026') or get_value(lookup, 'Ratios', 'ROCE %', 'Mar 2026')
    roe = get_value(lookup, 'Ratios', 'ROE', 'Mar 2026') or get_value(lookup, 'Ratios', 'ROE %', 'Mar 2026')
    cap_prod_roce = round(float(roce), 2) if roce is not None else None
    cap_prod_roe = round(float(roe), 2) if roe is not None else None

    balance_sheet_safety_status = None
    if debt_to_reserves is not None:
        if debt_to_reserves < 0.5:
            balance_sheet_safety_status = 'SAFE_FORTRESS'
        elif debt_to_reserves < 1.0:
            balance_sheet_safety_status = 'WATCHED'
        else:
            balance_sheet_safety_status = 'HIGH_LEVERAGE'

    # Pillar D
    cfo = get_value(lookup, 'Cash_Flow', 'Cash from Operating Activity+', 'Mar 2026')
    fcf = get_value(lookup, 'Cash_Flow', 'Free Cash Flow', 'Mar 2026')
    operating_profit = get_value(lookup, 'Profit_Loss', 'Operating Profit', 'Mar 2026')

    cash_conversion_eff = None
    fcf_to_cfo = None
    if cfo is not None and operating_profit is not None and operating_profit != 0:
        cash_conversion_eff = (float(cfo) / float(operating_profit)) * 100
    if fcf is not None and cfo is not None and cfo != 0:
        fcf_to_cfo = (float(fcf) / float(cfo)) * 100

    # Pillar E
    def safe_delta(metric_name: str) -> Optional[float]:
        a = get_value(lookup, 'Shareholding_Q', metric_name, 'Jun 2026')
        b = get_value(lookup, 'Shareholding_Q', metric_name, 'Mar 2026')
        if a is None or b is None:
            return None
        try:
            return float(a) - float(b)
        except Exception:
            return None

    promoter_delta = safe_delta('Promoters+')
    fii_delta = safe_delta('FIIs+')
    dii_delta = safe_delta('DIIs+')

    institutional_flow_pattern = None
    if promoter_delta is not None and fii_delta is not None and dii_delta is not None:
        if promoter_delta > 0 and dii_delta > 0 and fii_delta < 0:
            institutional_flow_pattern = 'DOMESTIC_ABSORBING_FOREIGN_OUTFLOW'
        elif fii_delta > 0 and dii_delta < 0:
            institutional_flow_pattern = 'FOREIGN_INFLOW'
        else:
            institutional_flow_pattern = 'MIXED'

    # Qualitative
    pros = []
    cons = []
    # try to extract pros/cons from Pros_Cons sheet
    pros_list = lookup.get('Pros_Cons', {})
    # pros_list likely has metric names as keys; iterate values
    for k, periods in pros_list.items():
        for p, v in periods.items():
            # v might be string
            if isinstance(v, str):
                if 'Pros' in k or 'Pros' in str(v):
                    pros.append(v)
                if 'Cons' in k or 'Cons' in str(v):
                    cons.append(v)

    # fallback: check overview pros/cons in lookup text
    # assemble result
    result = {
        'symbol': symbol.upper(),
        'data_as_of': datetime.utcnow().date().isoformat(),
        'calculated_fundamental_derivatives': {
            'growth_velocity_cagr': {
                'sales_7year_trajectory_pct': sales_pct,
                'net_profit_7year_trajectory_pct': net_pct,
                'growth_alignment_status': growth_alignment_status,
            },
            'valuation_realism': {
                'static_pe_ratio': float(pe) if pe is not None else None,
                'peg_ratio_proxy': round(peg_proxy, 2) if peg_proxy is not None else None,
                'dividend_yield_pct': float(dividend_yield) if dividend_yield is not None else None,
                'pricing_premium_bracket': pricing_premium_bracket,
            },
            'solvency_and_leverage': {
                'debt_to_reserves_leverage': round(debt_to_reserves, 2) if isinstance(debt_to_reserves, (int, float)) else debt_to_reserves,
                'capital_productivity_roce_pct': cap_prod_roce,
                'capital_productivity_roe_pct': cap_prod_roe,
                'balance_sheet_safety_status': balance_sheet_safety_status,
            },
            'liquidity_earnings_quality': {
                'cash_conversion_efficiency_pct': round(cash_conversion_eff, 2) if cash_conversion_eff is not None else None,
                'fcf_to_cfo_efficiency_pct': round(fcf_to_cfo, 2) if fcf_to_cfo is not None else None,
                'accounting_cash_realization_status': 'HEALTHY_LIQUIDITY_REALIZATION' if cash_conversion_eff and cash_conversion_eff > 50 else 'POOR_REALIZATION' if cash_conversion_eff is not None else None,
            },
            'smart_money_momentum_qoq': {
                'promoter_stake_delta_pct': promoter_delta,
                'fii_stake_delta_pct': fii_delta,
                'dii_stake_delta_pct': dii_delta,
                'institutional_flow_pattern': institutional_flow_pattern,
            },
        },
        'extracted_qualitative_context': {
            'verified_pros': pros,
            'verified_cons': cons,
        },
    }

    return result


def save_result(symbol: str, result: Dict[str, Any], base: Path) -> Path:
    folder = Path(base) / symbol.strip().upper()
    folder.mkdir(parents=True, exist_ok=True)
    outp = folder / f"{symbol.strip().upper()}_derived.json"
    with outp.open('w', encoding='utf8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return outp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('symbol')
    parser.add_argument('--exports-dir', default=str(Path(__file__).resolve().parent / 'exports'))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    symbol = args.symbol.strip().upper()
    exports = Path(args.exports_dir) / symbol
    # JSONL file may be lowercase name
    jsonl1 = exports / f"{symbol.lower()}.jsonl"
    jsonl2 = exports / f"{symbol}.jsonl"
    if jsonl1.exists():
        jpath = jsonl1
    elif jsonl2.exists():
        jpath = jsonl2
    else:
        logging.error('JSONL file not found for %s in %s', symbol, exports)
        raise SystemExit(2)

    items = load_jsonl(jpath)
    lookup = build_lookup(items)
    result = compute_derivatives(lookup, symbol)
    outp = save_result(symbol, result, Path(args.exports_dir))
    print('Saved derived JSON:', outp)


if __name__ == '__main__':
    main()
