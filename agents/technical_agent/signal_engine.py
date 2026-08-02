"""signal_engine.py

Synthesises raw indicator values into actionable trading signals.

Philosophy:
  - A trade is ONLY generated when it passes a quality gate
  - Entry is at a REAL structural level (not just VWAP)
  - RR is maximised dynamically using the next real S/R level
  - Each of the 4 trading styles is evaluated independently
  - Position sizing is left to the user; outputs per-share risk/reward ₹ amounts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .trading_styles import TRADING_STYLES, TradingStyleProfile


@dataclass
class SignalConfig:
    min_rr: float = 1.5          # absolute minimum RR to generate any setup
    atr_stop_mult: float = 1.0   # fallback stop = entry ± (1.0 × ATR) if no level found


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
    signals: list[int] = []   # +1 bullish, -1 bearish

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
    st_dir = ind.get("supertrend", {}).get("direction") if isinstance(ind.get("supertrend"), dict) else None
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

    score = round(agreement * 10, 1)

    return {
        "score": score,
        "out_of": 10,
        "dominant_bias": dominant,
        "agreement_pct": round(agreement * 100),
        "per_timeframe": tf_directions,
        "interpretation": (
            "Strong signal" if score >= 7
            else "Moderate signal" if score >= 4
            else "Weak / mixed signal"
        ),
    }


# ── quality gate ──────────────────────────────────────────────────────────────

def _passes_quality_gate(
    bias_indicators: Dict[str, Any],
    style: TradingStyleProfile,
    confluence_score: float,
    bias: str,
) -> Tuple[bool, str]:
    """Return (passed, reason_string) for a setup quality check."""
    # 1. Confluence
    if confluence_score < style.min_confluence:
        return False, f"Confluence {confluence_score}/10 < minimum {style.min_confluence} for {style.display_name}"

    # 2. Trend strength via ADX on bias TF
    adx = _v(bias_indicators, "adx", "adx")
    if adx is None or adx < style.min_adx:
        adx_str = f"{adx:.1f}" if adx else "N/A"
        return False, f"ADX {adx_str} < minimum {style.min_adx} on bias TF (no clear trend)"

    # 3. Direction must be clear (not ranging) on bias TF
    if bias == "ranging":
        return False, "Bias timeframe is ranging — no directional trade"

    return True, "OK"


# ── level-based entry logic ───────────────────────────────────────────────────

def _find_entry_and_levels(
    entry_tf_ind: Dict[str, Any],
    structure_tf_ind: Dict[str, Any],
    bias: str,
    config: SignalConfig,
) -> Dict[str, Any] | None:
    """
    Find a realistic entry level using structural S/R.
    Returns entry, stop, target, rr, per_share_risk, per_share_reward, or None.
    """
    atr = _v(entry_tf_ind, "atr")
    current_price = _v(entry_tf_ind, "vwap") or _v(entry_tf_ind, "ema20")
    if not atr or not current_price:
        return None

    # Collect candidate support levels (for long entries)
    pivots = structure_tf_ind.get("pivot_points", {}) or {}
    boll = entry_tf_ind.get("bollinger", {}) or {}
    ichi = entry_tf_ind.get("ichimoku", {}) or {}
    don20 = entry_tf_ind.get("donchian_20", {}) or {}
    vwap = _v(entry_tf_ind, "vwap")

    support_levels: List[Tuple[float, str]] = []
    resistance_levels: List[Tuple[float, str]] = []

    def _add(val, label):
        if val is not None:
            v = float(val)
            if v > 0:
                if v < current_price:
                    support_levels.append((v, label))
                elif v > current_price:
                    resistance_levels.append((v, label))

    _add(pivots.get("s1"), "Pivot S1")
    _add(pivots.get("s2"), "Pivot S2")
    _add(pivots.get("pivot"), "Pivot PP")
    _add(boll.get("lower"), "Bollinger Lower")
    _add(boll.get("middle"), "Bollinger Mid")
    _add(don20.get("lower"), "Donchian Lower (20)")
    ichi_bot = min(
        _v(ichi, "senkou_a") or float("inf"),
        _v(ichi, "senkou_b") or float("inf"),
    )
    if ichi_bot != float("inf"):
        _add(ichi_bot, "Ichimoku Cloud Bottom")
    if vwap:
        _add(vwap, "VWAP")

    _add(pivots.get("r1"), "Pivot R1")
    _add(pivots.get("r2"), "Pivot R2")
    _add(pivots.get("r3"), "Pivot R3")
    _add(boll.get("upper"), "Bollinger Upper")
    _add(boll.get("middle"), "Bollinger Mid")
    _add(don20.get("upper"), "Donchian Upper (20)")
    ichi_top = max(
        _v(ichi, "senkou_a") or 0,
        _v(ichi, "senkou_b") or 0,
    )
    if ichi_top > 0:
        _add(ichi_top, "Ichimoku Cloud Top")

    if bias == "bullish":
        # Entry: nearest support level above 1 ATR below current price
        candidates = [(v, l) for v, l in support_levels if v >= current_price - 2 * atr]
        if not candidates:
            return None
        entry_val, entry_label = max(candidates, key=lambda x: x[0])

        # Stop: next support level below entry
        stop_candidates = [(v, l) for v, l in support_levels if v < entry_val - 0.2 * atr]
        if stop_candidates:
            stop_val, stop_label = max(stop_candidates, key=lambda x: x[0])
        else:
            stop_val = round(entry_val - config.atr_stop_mult * atr, 2)
            stop_label = f"ATR-based ({config.atr_stop_mult}×ATR)"

        stop_distance = entry_val - stop_val
        if stop_distance <= 0:
            return None

        # Target: furthest valid resistance level with RR ≥ min
        target_min = entry_val + stop_distance * config.min_rr
        target_candidates = [(v, l) for v, l in resistance_levels if v >= target_min]
        if target_candidates:
            target_val, target_label = max(target_candidates, key=lambda x: x[0])
        else:
            target_val = round(entry_val + stop_distance * config.min_rr, 2)
            target_label = f"Min RR target ({config.min_rr}×)"

    else:  # bearish
        # Entry: nearest resistance below 1 ATR above current price
        candidates = [(v, l) for v, l in resistance_levels if v <= current_price + 2 * atr]
        if not candidates:
            return None
        entry_val, entry_label = min(candidates, key=lambda x: x[0])

        # Stop: next resistance above entry
        stop_candidates = [(v, l) for v, l in resistance_levels if v > entry_val + 0.2 * atr]
        if stop_candidates:
            stop_val, stop_label = min(stop_candidates, key=lambda x: x[0])
        else:
            stop_val = round(entry_val + config.atr_stop_mult * atr, 2)
            stop_label = f"ATR-based ({config.atr_stop_mult}×ATR)"

        stop_distance = stop_val - entry_val
        if stop_distance <= 0:
            return None

        # Target: furthest valid support with RR ≥ min
        target_min = entry_val - stop_distance * config.min_rr
        target_candidates = [(v, l) for v, l in support_levels if v <= target_min]
        if target_candidates:
            target_val, target_label = min(target_candidates, key=lambda x: x[0])
        else:
            target_val = round(entry_val - stop_distance * config.min_rr, 2)
            target_label = f"Min RR target ({config.min_rr}×)"

    if bias == "bullish":
        reward_distance = target_val - entry_val
    else:
        reward_distance = entry_val - target_val

    rr = round(reward_distance / stop_distance, 2) if stop_distance > 0 else 0.0

    return {
        "entry": round(entry_val, 2),
        "entry_level": entry_label,
        "stop_loss": round(stop_val, 2),
        "stop_level": stop_label,
        "take_profit": round(target_val, 2),
        "target_level": target_label,
        "risk_reward": rr,
        "per_share_risk": round(stop_distance, 2),
        "per_share_reward": round(reward_distance, 2),
        "atr": round(atr, 2),
        "atr_pct": _v(entry_tf_ind, "atr_pct"),
    }


# ── key levels ────────────────────────────────────────────────────────────────

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


# ── human summary ─────────────────────────────────────────────────────────────

def _human_summary(
    confluence: Dict[str, Any],
    levels: Dict[str, Any],
    style_setups: Dict[str, Any],
    candlestick: Dict[str, Any | None],
) -> str:
    """Build a detailed plain-English trading analysis summary."""
    bias = confluence["dominant_bias"].upper()
    score = confluence["score"]
    interp = confluence["interpretation"]
    tf_str = ", ".join(
        f"{tf.capitalize()}: {d.capitalize()}"
        for tf, d in confluence["per_timeframe"].items()
    )

    parts = [
        f"The dominant technical bias is {bias} with a multi-timeframe confluence score of {score}/10, indicating a {interp.lower()} environment.",
        f"Timeframe breakdown: [{tf_str}].",
    ]

    # Key Levels
    level_notes = []
    pivot = levels.get("pivot")
    if pivot:
        level_notes.append(f"Daily Pivot at {pivot}")
    vwap = levels.get("vwap")
    if vwap:
        level_notes.append(f"VWAP at {vwap:.2f}")
    if level_notes:
        parts.append(f"Key structural levels: {', '.join(level_notes)}.")

    # Style Setups
    valid_setups = {s: v for s, v in style_setups.items() if v.get("setup")}
    rejected_styles = {s: v for s, v in style_setups.items() if not v.get("setup")}

    if valid_setups:
        parts.append(
            f"{len(valid_setups)} trading style(s) produced a valid setup: "
            + ", ".join(s.capitalize() for s in valid_setups.keys()) + "."
        )
        for style_name, style_data in valid_setups.items():
            setup = style_data["setup"]
            d = "Long" if style_data["bias"] == "bullish" else "Short"
            parts.append(
                f"[{style_name.upper()}] {d} near {setup['entry']:.2f} "
                f"(at {setup['entry_level']}). "
                f"Stop at {setup['stop_loss']:.2f} ({setup['stop_level']}), "
                f"Target {setup['take_profit']:.2f} ({setup['target_level']}). "
                f"RR: {setup['risk_reward']}×. "
                f"Per-share risk: ₹{setup['per_share_risk']:.2f}, "
                f"Per-share reward: ₹{setup['per_share_reward']:.2f}."
            )
    else:
        parts.append(
            "No high-confidence setup found across any trading style. "
            "Market appears to be in a consolidation or transition phase."
        )

    if rejected_styles:
        reasons = "; ".join(
            f"{s.capitalize()}: {v.get('rejection_reason', 'filtered')}"
            for s, v in rejected_styles.items()
        )
        parts.append(f"Filtered styles — {reasons}.")

    # Candlestick signals
    cs_signals = [v for v in candlestick.values() if v]
    if cs_signals:
        parts.append(f"Recent candlestick patterns: {', '.join(cs_signals)}.")

    # 52-week context
    dist_high = levels.get("52w_dist_from_high_pct")
    if dist_high is not None:
        pos = "below" if dist_high < 0 else "above"
        zone = "lower" if dist_high < -20 else "upper"
        parts.append(
            f"The stock is {abs(dist_high):.1f}% {pos} its 52-week high, "
            f"placing it in the {zone} end of its annual price range."
        )

    return " ".join(parts)


# ── public API ────────────────────────────────────────────────────────────────

def generate_signals(
    timeframes: Dict[str, Dict[str, Any]],
    patterns: Dict[str, Any],
    config: SignalConfig | None = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive signal report for a symbol.

    Evaluates ALL 4 trading styles (scalp, intraday, swing, positional).
    Each style only produces a setup if it passes the quality gate.

    Args:
        timeframes: The ``timeframes`` dict from :func:`TechnicalAgent.analyse`.
        patterns: The ``patterns`` dict from :func:`TechnicalAgent.analyse`
        config: Optional :class:`SignalConfig`.

    Returns:
        A dict with keys: ``confluence``, ``style_setups``, ``levels``,
        ``candlestick_signals``, ``summary``.
    """
    cfg = config or SignalConfig()

    confluence = _confluence_score(timeframes)

    # Primary frame for key levels
    primary = timeframes.get("daily") or next(iter(timeframes.values()), {})
    levels = _key_levels(primary)

    style_setups: Dict[str, Any] = {}

    for style_name, style_profile in TRADING_STYLES.items():
        # Get the required timeframes for this style
        bias_tf_data = timeframes.get(style_profile.bias_tf, {})
        structure_tf_data = timeframes.get(style_profile.structure_tf, {})
        entry_tf_data = timeframes.get(style_profile.entry_tf, {})

        # If any required TF is missing, skip this style
        if not bias_tf_data or not entry_tf_data:
            style_setups[style_name] = {
                "setup": None,
                "rejection_reason": f"Required timeframe '{style_profile.bias_tf}' or '{style_profile.entry_tf}' not fetched",
                "bias": "N/A",
            }
            continue

        # Determine bias from the bias TF (not global consensus)
        bias = _trend_direction(bias_tf_data)

        # Quality gate
        passed, reason = _passes_quality_gate(
            bias_tf_data, style_profile, confluence["score"], bias
        )

        if not passed:
            style_setups[style_name] = {
                "setup": None,
                "rejection_reason": reason,
                "bias": bias,
            }
            continue

        # Find level-based entry
        setup = _find_entry_and_levels(
            entry_tf_ind=entry_tf_data,
            structure_tf_ind=structure_tf_data or entry_tf_data,
            bias=bias,
            config=cfg,
        )

        if setup is None or setup.get("risk_reward", 0) < style_profile.min_rr:
            rr = setup.get("risk_reward", 0) if setup else 0
            style_setups[style_name] = {
                "setup": None,
                "rejection_reason": f"No valid entry level found or RR {rr}× < minimum {style_profile.min_rr}×",
                "bias": bias,
            }
            continue

        setup["bias"] = bias
        style_setups[style_name] = {
            "setup": setup,
            "rejection_reason": None,
            "bias": bias,
            "style_description": style_profile.description,
        }

    candlestick = patterns.get("candlestick", {}) or {}
    summary = _human_summary(confluence, levels, style_setups, candlestick)

    return {
        "confluence": confluence,
        "style_setups": style_setups,
        "levels": levels,
        "candlestick_signals": {k: v for k, v in candlestick.items() if v},
        "summary": summary,
    }
