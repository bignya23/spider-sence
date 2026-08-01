"""signal_engine.py

Synthesises raw indicator values into actionable trading signals.

Outputs:
  - Trend direction per timeframe (bullish / bearish / ranging)
  - Multi-timeframe confluence score (0–10)
  - ATR-based entry / stop / target levels
  - Risk/Reward ratio
  - Position size (1% risk model)
  - Human-readable signal summary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SignalConfig:
    account_size: float = 100_000.0   # ₹ account size for position sizing
    risk_pct: float = 0.01             # 1% risk per trade
    atr_stop_mult: float = 1.5         # stop = entry ± (1.5 × ATR)
    atr_target_mult: float = 3.0       # target = entry ± (3.0 × ATR)  → RR ≥ 2


# ── helpers ───────────────────────────────────────────────────────────────────

def _v(d: Dict, *keys) -> float | None:
    """Safely navigate nested dict and return a float or None."""
    obj = d
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    if obj is None:
        return None
    try:
        return float(obj)
    except (TypeError, ValueError):
        return None


def _trend_direction(ind: Dict[str, Any]) -> str:
    """Classify a timeframe as bullish / bearish / ranging from its indicators."""
    signals: list[int] = []   # +1 bullish, -1 bearish, 0 neutral

    close = _v(ind, "ema20")   # proxy for close ≈ last EMA20 when price ≈ EMA

    # EMA alignment
    ema20 = _v(ind, "ema20")
    ema50 = _v(ind, "ema50")
    ema200 = _v(ind, "ema200")
    if ema20 and ema50 and ema200:
        if ema20 > ema50 > ema200:
            signals.append(1)
        elif ema20 < ema50 < ema200:
            signals.append(-1)
        else:
            signals.append(0)

    # RSI
    rsi = _v(ind, "rsi")
    if rsi is not None:
        if rsi > 55:
            signals.append(1)
        elif rsi < 45:
            signals.append(-1)
        else:
            signals.append(0)

    # MACD histogram
    hist = _v(ind, "macd", "histogram")
    if hist is not None:
        signals.append(1 if hist > 0 else -1)

    # ADX direction (DI+ vs DI-)
    di_plus = _v(ind, "adx", "di_plus")
    di_minus = _v(ind, "adx", "di_minus")
    adx_val = _v(ind, "adx", "adx")
    if di_plus and di_minus and adx_val and adx_val > 20:
        signals.append(1 if di_plus > di_minus else -1)

    # Supertrend
    st_dir = ind.get("supertrend", {}).get("direction")
    if st_dir == "bullish":
        signals.append(1)
    elif st_dir == "bearish":
        signals.append(-1)

    # Stochastic
    k = _v(ind, "stochastic", "k")
    d = _v(ind, "stochastic", "d")
    if k and d:
        if k > d and k < 80:
            signals.append(1)
        elif k < d and k > 20:
            signals.append(-1)

    # MFI
    mfi = _v(ind, "mfi")
    if mfi is not None:
        if mfi > 60:
            signals.append(1)
        elif mfi < 40:
            signals.append(-1)
        else:
            signals.append(0)

    if not signals:
        return "ranging"

    score = sum(signals) / len(signals)
    if score >= 0.3:
        return "bullish"
    elif score <= -0.3:
        return "bearish"
    return "ranging"


def _confluence_score(timeframes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score multi-timeframe alignment on a 0–10 scale.

    10 = all timeframes, all indicators agree on direction.
    """
    directions: list[str] = []
    tf_directions: dict[str, str] = {}

    for tf_name, ind in timeframes.items():
        d = _trend_direction(ind)
        directions.append(d)
        tf_directions[tf_name] = d

    bulls = directions.count("bullish")
    bears = directions.count("bearish")
    total = len(directions)

    if bulls >= bears:
        dominant = "bullish"
        agreement = bulls / total if total else 0
    else:
        dominant = "bearish"
        agreement = bears / total if total else 0

    # Scale 0–10: full agreement across all TFs = 10
    score = round(agreement * 10, 1)

    return {
        "score": score,
        "out_of": 10,
        "dominant_bias": dominant,
        "agreement_pct": round(agreement * 100),
        "per_timeframe": tf_directions,
        "interpretation": (
            "Strong signal" if score >= 7
            else "Moderate signal" if score >= 5
            else "Weak / mixed signal"
        ),
    }


def _trade_levels(
    primary_indicators: Dict[str, Any],
    dominant_bias: str,
    config: SignalConfig,
) -> Dict[str, Any]:
    """Compute ATR-based entry, stop-loss, and take-profit levels."""
    atr = _v(primary_indicators, "atr")
    vwap = _v(primary_indicators, "vwap")
    ema20 = _v(primary_indicators, "ema20")

    # Use VWAP as entry proxy if available, else EMA20
    entry = vwap or ema20
    if entry is None or atr is None:
        return {"entry": None, "stop": None, "target": None, "rr": None}

    stop_distance = config.atr_stop_mult * atr
    target_distance = config.atr_target_mult * atr

    if dominant_bias == "bullish":
        stop = round(entry - stop_distance, 2)
        target = round(entry + target_distance, 2)
    else:
        stop = round(entry + stop_distance, 2)
        target = round(entry - target_distance, 2)

    rr = round(target_distance / stop_distance, 2) if stop_distance else None

    # Position size: risk 1% of account / stop distance per share
    risk_amount = config.account_size * config.risk_pct
    qty = int(risk_amount / stop_distance) if stop_distance else 0

    return {
        "entry": round(entry, 2),
        "stop_loss": stop,
        "take_profit": target,
        "risk_reward": rr,
        "atr": round(atr, 2),
        "atr_pct": _v(primary_indicators, "atr_pct"),
        "position_size_shares": qty,
        "risk_amount": round(risk_amount, 2),
    }


def _key_levels(primary_indicators: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidate all key price levels into one clean dict."""
    pivots = primary_indicators.get("pivot_points", {}) or {}
    levels_52w = primary_indicators.get("52w", {}) or {}
    boll = primary_indicators.get("bollinger", {}) or {}
    don55 = primary_indicators.get("donchian_55", {}) or {}
    ichi = primary_indicators.get("ichimoku", {}) or {}
    vwap = _v(primary_indicators, "vwap")

    return {
        "vwap": vwap,
        "pivot": pivots.get("pivot"),
        "resistance": {
            "r1": pivots.get("r1"),
            "r2": pivots.get("r2"),
            "r3": pivots.get("r3"),
            "bollinger_upper": boll.get("upper"),
            "donchian_55_upper": don55.get("upper"),
            "52w_high": levels_52w.get("high"),
            "ichimoku_cloud_top": max(
                ichi.get("senkou_a") or 0, ichi.get("senkou_b") or 0
            ) or None,
        },
        "support": {
            "s1": pivots.get("s1"),
            "s2": pivots.get("s2"),
            "s3": pivots.get("s3"),
            "bollinger_lower": boll.get("lower"),
            "donchian_55_lower": don55.get("lower"),
            "52w_low": levels_52w.get("low"),
            "ichimoku_cloud_bottom": min(
                ichi.get("senkou_a") or float("inf"),
                ichi.get("senkou_b") or float("inf"),
            ) if (ichi.get("senkou_a") and ichi.get("senkou_b")) else None,
        },
        "cpr": {
            "top": pivots.get("cpr_top"),
            "bottom": pivots.get("cpr_bottom"),
        },
        "52w_dist_from_high_pct": levels_52w.get("dist_from_high_pct"),
        "52w_dist_from_low_pct": levels_52w.get("dist_from_low_pct"),
    }


def _human_summary(
    confluence: Dict[str, Any],
    levels: Dict[str, Any],
    trade: Dict[str, Any],
    candlestick: Dict[str, Any | None],
) -> str:
    """Build a one-paragraph plain-English trading summary."""
    bias = confluence["dominant_bias"].upper()
    score = confluence["score"]
    interp = confluence["interpretation"]
    tf_str = ", ".join(f"{tf}: {d}" for tf, d in confluence["per_timeframe"].items())

    parts = [
        f"Bias: {bias} ({interp}, {score}/10).",
        f"Timeframe alignment: [{tf_str}].",
    ]

    # Candlestick signals
    cs_signals = [v for v in candlestick.values() if v]
    if cs_signals:
        parts.append(f"Recent candlestick patterns: {', '.join(cs_signals)}.")

    # Levels
    pivot = levels.get("pivot")
    if pivot:
        parts.append(f"Pivot: {pivot}.")
    vwap = levels.get("vwap")
    if vwap:
        parts.append(f"VWAP: {vwap:.2f}.")

    # Trade levels
    if trade.get("entry"):
        direction = "Buy" if confluence["dominant_bias"] == "bullish" else "Sell"
        parts.append(
            f"{direction} near {trade['entry']:.2f}, "
            f"SL: {trade['stop_loss']:.2f}, "
            f"Target: {trade['take_profit']:.2f} "
            f"(RR: {trade['risk_reward']}×)."
        )
        if trade.get("position_size_shares"):
            parts.append(f"Position size: {trade['position_size_shares']} shares "
                         f"(risking ₹{trade['risk_amount']:.0f}).")

    dist_high = levels.get("52w_dist_from_high_pct")
    if dist_high is not None:
        parts.append(f"Stock is {abs(dist_high):.1f}% {'below' if dist_high < 0 else 'above'} its 52-week high.")

    return " ".join(parts)


# ── public API ────────────────────────────────────────────────────────────────

def generate_signals(
    timeframes: Dict[str, Dict[str, Any]],
    patterns: Dict[str, Any],
    config: SignalConfig | None = None,
) -> Dict[str, Any]:
    """Generate the full signal report for a symbol.

    Args:
        timeframes: The ``timeframes`` dict from :func:`TechnicalAgent.analyse`.
        patterns: The ``patterns`` dict from :func:`TechnicalAgent.analyse`.
        config: Optional :class:`SignalConfig` for position sizing parameters.

    Returns:
        A dict with keys: ``confluence``, ``levels``, ``trade``, ``summary``.
    """
    cfg = config or SignalConfig()

    confluence = _confluence_score(timeframes)

    # Use daily as primary frame; fallback to first available
    primary = timeframes.get("daily") or next(iter(timeframes.values()), {})

    trade = _trade_levels(primary, confluence["dominant_bias"], cfg)
    levels = _key_levels(primary)

    candlestick = patterns.get("candlestick", {}) or {}
    summary = _human_summary(confluence, levels, trade, candlestick)

    return {
        "confluence": confluence,
        "trade": trade,
        "levels": levels,
        "candlestick_signals": {k: v for k, v in candlestick.items() if v},
        "summary": summary,
    }
