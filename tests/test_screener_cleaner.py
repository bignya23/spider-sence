import json
from pathlib import Path
import pytest

from agents.fundamental_agent.screener_enricher import load_jsonl, build_lookup, compute_derivatives


def test_compute_reliance_derivatives():
    exports = Path('agents/fundamental_agent/exports/RELIANCE')
    jsonl = exports / 'reliance.jsonl'
    assert jsonl.exists(), f"Expected JSONL at {jsonl}"

    items = load_jsonl(jsonl)
    assert len(items) > 0

    lookup = build_lookup(items)
    result = compute_derivatives(lookup, 'RELIANCE')
    # basic structure checks
    assert result['symbol'] == 'RELIANCE'
    assert 'calculated_fundamental_derivatives' in result
    deriv = result['calculated_fundamental_derivatives']
    # ensure keys exist
    assert 'growth_velocity_cagr' in deriv
    assert 'valuation_realism' in deriv
    assert 'solvency_and_leverage' in deriv
    assert 'liquidity_earnings_quality' in deriv
    assert 'smart_money_momentum_qoq' in deriv

    # types
    gv = deriv['growth_velocity_cagr']
    assert (gv['sales_7year_trajectory_pct'] is None) or isinstance(gv['sales_7year_trajectory_pct'], float)
    assert (gv['net_profit_7year_trajectory_pct'] is None) or isinstance(gv['net_profit_7year_trajectory_pct'], float)

    vr = deriv['valuation_realism']
    assert 'static_pe_ratio' in vr


if __name__ == '__main__':
    pytest.main([str(Path(__file__))])
