"""signal_engine.py

Synthesises raw indicator values into actionable trading signals using a
4-step hierarchical filter system.

Step 1: Trend & Regime Filter    — defines market environment (trending / ranging)
Step 2: Structural Level Check   — ensures price is near a meaningful S/R zone
Step 3: Momentum + Volume Trigger — hard volume confirmation gate (lie detector)
Step 4: Dynamic Risk Management  — 1.5x ATR stop + Supertrend trailing exit

Philosophy:
  - A trade is ONLY generated when ALL 4 gates pass sequentially
  - ADX < 20 activates Mean Reversion mode (Bollinger/VWAP bounce logic)
  - ADX > 25 activates Trend Following mode (breakout/continuation logic)
  - RVOL > 1.2x is a HARD gate — no signal fires without volume confirmation
  - Stop = 1.5x ATR; trailing stop via Supertrend line
  - Position sizing left to the user; outputs per-share risk/reward ₹ amounts
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from .trading_styles import TRADING_STYLES, TradingStyleProfile


@dataclass
class SignalConfig:
    min_rr: float = 1.5           # absolute minimum RR to generate any setup
    atr_stop_mult: float = 1.5    # stop = entry ± (1.5 × ATR) — upgraded from 1.0
    rvol_threshold: float = 1.2   # RVOL must be above this multiple (hard gate)


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


# ── Step 1: Trend & Regime Filter ────────────────────────────────────────────

def _regime_mode(adx_val: float | None) -> str:
    """Classify market regime from ADX value.

    Returns:
        'trending'   — ADX > 25: clear directional trend
        'transition' — ADX 20–25: emerging trend, reduced conviction
        'ranging'    — ADX < 20: choppy/sideways, use mean-reversion logic
    """
    if adx_val is None:
        return "ranging"
    if adx_val > 25:
        return "trending"
    if adx_val >= 20:
        return "transition"
    return "ranging"


def _trend_direction(ind: Dict[str, Any]) -> str:
    """Classify a timeframe as bullish / bearish / ranging from its indicators."""
    signals: list[int] = []   # +1 bullish, -1 bearish

    # EMA alignment
    ema20 = _v(ind, "ema20")
    ema50 = _v(ind, "ema50")
    ema200 = _v(ind, "ema200")
    if ema20 is not None and ema50 is not None and ema200 is not None:
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
    if di_plus is not None and di_minus is not None and adx_val is not None and adx_val > 20:
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
    if k is not None and d is not None:
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


def _step1_trend_filter(
    bias_tf_ind: Dict[str, Any],
    style: TradingStyleProfile,
) -> Tuple[bool, str, str, str]:
    """Step 1: Trend & Regime Filter.

    Establishes market direction and regime. Rejects setups that would trade
    against the primary trend or are incompatible with the current ADX regime.

    Returns:
        (passed, direction, regime, reason)
    """
    adx_val = _v(bias_tf_ind, "adx", "adx")
    regime = _regime_mode(adx_val)

    # Regime compatibility check — hard gate
    if regime not in style.valid_regimes:
        adx_str = f"{adx_val:.1f}" if adx_val is not None else "N/A"
        return (
            False, "ranging", regime,
            f"Global regime '{regime}' not valid for {style.display_name} "
            f"[valid: {', '.join(style.valid_regimes)}]"
        )

    direction = _trend_direction(bias_tf_ind)

    # In trending mode: price must be on the right side of EMA200 and VWAP
    if regime in ("trending", "transition"):
        ema200 = _v(bias_tf_ind, "ema200")
        vwap = _v(bias_tf_ind, "vwap") or _v(bias_tf_ind, "session_vwap")
        
        # Missing data should fail safe, don't fall back to ema200 itself
        price_ref = _v(bias_tf_ind, "close") or _v(bias_tf_ind, "ema20") or vwap
        
        if price_ref is None or ema200 is None:
            return (
                False, direction, regime,
                "Missing price or EMA200 data — cannot confirm trend"
            )

        if direction == "bullish" and price_ref < ema200:
                return (
                    False, direction, regime,
                    "Long trade blocked: price below 200 EMA (counter-trend)"
                )
        if direction == "bearish" and price_ref > ema200:
                return (
                    False, direction, regime,
                    "Short trade blocked: price above 200 EMA (counter-trend)"
                )

        if direction == "ranging":
            return (
                False, direction, regime,
                "Bias timeframe is ranging — no directional trend-following trade"
            )

        # ADX minimum check for trend styles
        if adx_val is not None and adx_val < style.min_adx:
            return (
                False, direction, regime,
                f"ADX {adx_val:.1f} < minimum {style.min_adx} for {style.display_name}"
            )

    return True, direction, regime, "OK"


# ── Step 2: Structural Level Check ──────────────────────────────────────────

def _step2_structural_check(
    entry_tf_ind: Dict[str, Any],
    bias_tf_ind: Dict[str, Any],
    direction: str,
    regime: str,
    config: SignalConfig,
) -> Tuple[bool, str]:
    """Step 2: Structural Level Check.

    Ensures price is interacting with a meaningful S/R zone rather than trading
    in empty air. For mean-reversion, checks band proximity. For trend-following,
    checks breakout from structure.

    Returns:
        (at_level, level_description)
    """
    atr = _v(entry_tf_ind, "atr")
    if not atr:
        return False, "ATR not available — cannot assess structural proximity"

    price = _v(entry_tf_ind, "close") or _v(entry_tf_ind, "vwap") or _v(entry_tf_ind, "ema20")
    if not price:
        return False, "No price reference available"

    proximity = atr * config.atr_stop_mult   # Within configured ATR mult of a level = "at the level"

    boll = entry_tf_ind.get("bollinger", {}) or {}
    pivots = entry_tf_ind.get("pivot_points", {}) or bias_tf_ind.get("pivot_points", {}) or {}
    ichi = entry_tf_ind.get("ichimoku", {}) or {}
    vwap_b = entry_tf_ind.get("vwap_bands", {}) or {}
    ib = entry_tf_ind.get("initial_balance", {}) or {}

    levels_hit: list[str] = []

    def _near(val, label):
        if val is not None and abs(price - float(val)) <= proximity:
            levels_hit.append(label)

    # ── Mean-reversion regime: look for band extremes ───────────────────────
    if regime == "ranging":
        if direction == "bullish":
            _near(boll.get("lower"), "Bollinger Lower Band")
            _near(vwap_b.get("lower_1sd"), "VWAP -1σ Band")
            _near(vwap_b.get("lower_2sd"), "VWAP -2σ Band (Extreme Oversold)")
            _near(pivots.get("s1"), "Pivot S1")
            _near(pivots.get("s2"), "Pivot S2")
        else:  # bearish
            _near(boll.get("upper"), "Bollinger Upper Band")
            _near(vwap_b.get("upper_1sd"), "VWAP +1σ Band")
            _near(vwap_b.get("upper_2sd"), "VWAP +2σ Band (Extreme Overbought)")
            _near(pivots.get("r1"), "Pivot R1")
            _near(pivots.get("r2"), "Pivot R2")
    # ── Trending regime: look for breakout / pullback to structure ───────────
    else:
        ichi_top = max(_v(ichi, "senkou_a") or 0, _v(ichi, "senkou_b") or 0)
        ichi_bot = min(
            _v(ichi, "senkou_a") or float("inf"),
            _v(ichi, "senkou_b") or float("inf"),
        )
        ib_high = ib.get("ib_high")
        ib_low = ib.get("ib_low")

        _near(pivots.get("pivot"), "Daily Pivot")
        _near(pivots.get("cpr_top"), "CPR Top")
        _near(pivots.get("cpr_bottom"), "CPR Bottom")
        _near(boll.get("middle"), "Bollinger Mid (20 SMA)")

        if direction == "bullish":
            _near(pivots.get("r1"), "Pivot R1 Breakout")
            if ichi_top > 0:
                _near(ichi_top, "Ichimoku Cloud Top")
            if ib_high:
                _near(ib_high, "Initial Balance High Breakout")
        else:
            _near(pivots.get("s1"), "Pivot S1 Breakdown")
            if ichi_bot != float("inf"):
                _near(ichi_bot, "Ichimoku Cloud Bottom")
            if ib_low:
                _near(ib_low, "Initial Balance Low Breakdown")

    if not levels_hit:
        return False, "Price not near any structural S/R level — no structural edge"

    return True, " | ".join(levels_hit)


# ── Step 3: Momentum + Volume Trigger ────────────────────────────────────────

def _step3_momentum_trigger(
    entry_tf_ind: Dict[str, Any],
    direction: str,
    regime: str,
    config: SignalConfig,
    context: Dict[str, Any] = None,
) -> Tuple[bool, str]:
    """Step 3: Momentum + Volume Confirmation Gate.

    RVOL is a HARD gate — no signal fires without it.
    Combines volume confirmation with momentum indicator alignment.

    Returns:
        (triggered, reason)
    """
    reasons: list[str] = []
    blocks: list[str] = []

    # ── Index Correlation Check ──────────────────────────────────────────────
    if context:
        idx_bias = context.get("bias", "neutral").lower()
        if direction == "bullish" and idx_bias == "bearish":
            blocks.append("Blocked by Index: Market is trending bearish")
        elif direction == "bearish" and idx_bias == "bullish":
            blocks.append("Blocked by Index: Market is trending bullish")
            
    if blocks:
        return False, f"Sector Correlation failed: {'; '.join(blocks)}"

    # ── Hard Gate 1: Relative Volume ─────────────────────────────────────────
    rvol = _v(entry_tf_ind, "rvol")
    if rvol is None:
        blocks.append("RVOL unavailable")
    elif rvol < config.rvol_threshold:
        blocks.append(
            f"RVOL {rvol:.2f}× < threshold {config.rvol_threshold}× "
            f"(volume does not confirm the move)"
        )
    else:
        reasons.append(f"RVOL {rvol:.2f}×")

    if blocks:
        return False, f"Volume gate failed: {'; '.join(blocks)}"

    # ── Hard Gate 2: OBV direction ───────────────────────────────────────────
    # OBV alone is a cumulative series; we check its recent direction via MACD
    # on price as a proxy (OBV trend = price trend if confirmed)
    hist = _v(entry_tf_ind, "macd", "histogram")
    macd_line = _v(entry_tf_ind, "macd", "macd")
    signal_line = _v(entry_tf_ind, "macd", "signal")
    obv = _v(entry_tf_ind, "obv")

    macd_ok = False
    if hist is not None and macd_line is not None and signal_line is not None:
        if direction == "bullish" and hist > 0 and macd_line > signal_line:
            macd_ok = True
            reasons.append("MACD bullish crossover")
        elif direction == "bearish" and hist < 0 and macd_line < signal_line:
            macd_ok = True
            reasons.append("MACD bearish crossover")

    # ── Momentum confirmation (RSI) ──────────────────────────────────────────
    rsi = _v(entry_tf_ind, "rsi")
    rsi_ok = False
    if rsi is not None:
        if regime == "ranging":
            # Mean reversion: RSI in extreme territory
            if direction == "bullish" and rsi < 40:
                rsi_ok = True
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif direction == "bearish" and rsi > 60:
                rsi_ok = True
                reasons.append(f"RSI overbought ({rsi:.1f})")
        else:
            # Trending: RSI crossing 50 in direction of trade
            if direction == "bullish" and rsi > 50:
                rsi_ok = True
                reasons.append(f"RSI above 50 ({rsi:.1f})")
            elif direction == "bearish" and rsi < 50:
                rsi_ok = True
                reasons.append(f"RSI below 50 ({rsi:.1f})")

    # ── Stochastic (secondary confirmation) ─────────────────────────────────
    k = _v(entry_tf_ind, "stochastic", "k")
    d = _v(entry_tf_ind, "stochastic", "d")
    stoch_ok = False
    if k is not None and d is not None:
        if direction == "bullish" and k > d and k < 80:
            stoch_ok = True
            reasons.append(f"Stoch K>{d:.0f} (bullish cross)")
        elif direction == "bearish" and k < d and k > 20:
            stoch_ok = True
            reasons.append(f"Stoch K<{d:.0f} (bearish cross)")

    # Require at least 2 of 3 momentum confirmations: MACD, RSI, Stochastic
    momentum_count = sum([macd_ok, rsi_ok, stoch_ok])
    if momentum_count < 2:
        return (
            False,
            f"Momentum gate failed: need 2/3 indicators aligned — got {momentum_count}/3"
        )

    return True, " + ".join(reasons)


# ── Step 4: Dynamic Risk Management ──────────────────────────────────────────

def _find_entry_and_levels(
    entry_tf_ind: Dict[str, Any],
    structure_tf_ind: Dict[str, Any],
    bias: str,
    regime: str,
    config: SignalConfig,
) -> Dict[str, Any] | None:
    """Step 4: Build the trade setup with ATR-based stop and Supertrend trail.

    Returns entry, stop (1.5×ATR), target, rr, trail_via, or None.
    """
    atr = _v(entry_tf_ind, "atr")
    current_price = _v(entry_tf_ind, "close") or _v(entry_tf_ind, "vwap") or _v(entry_tf_ind, "ema20")
    if not atr or not current_price:
        return None

    pivots = structure_tf_ind.get("pivot_points", {}) or {}
    boll = entry_tf_ind.get("bollinger", {}) or {}
    ichi = entry_tf_ind.get("ichimoku", {}) or {}
    don20 = entry_tf_ind.get("donchian_20", {}) or {}
    vwap_b = entry_tf_ind.get("vwap_bands", {}) or {}
    ib = entry_tf_ind.get("initial_balance", {}) or {}
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

    # Structural levels
    _add(pivots.get("s1"), "Pivot S1")
    _add(pivots.get("s2"), "Pivot S2")
    _add(pivots.get("pivot"), "Pivot PP")
    _add(pivots.get("r1"), "Pivot R1")
    _add(pivots.get("r2"), "Pivot R2")
    _add(pivots.get("r3"), "Pivot R3")
    _add(boll.get("lower"), "Bollinger Lower")
    _add(boll.get("middle"), "Bollinger Mid")
    _add(boll.get("upper"), "Bollinger Upper")
    _add(don20.get("lower"), "Donchian Lower (20)")
    _add(don20.get("upper"), "Donchian Upper (20)")
    _add(vwap_b.get("lower_1sd"), "VWAP -1σ")
    _add(vwap_b.get("upper_1sd"), "VWAP +1σ")
    _add(ib.get("ib_high"), "IB High")
    _add(ib.get("ib_low"), "IB Low")

    ichi_bot = min(
        _v(ichi, "senkou_a") or float("inf"),
        _v(ichi, "senkou_b") or float("inf"),
    )
    ichi_top = max(_v(ichi, "senkou_a") or 0, _v(ichi, "senkou_b") or 0)
    if ichi_bot != float("inf"):
        _add(ichi_bot, "Ichimoku Cloud Bottom")
    if ichi_top > 0:
        _add(ichi_top, "Ichimoku Cloud Top")
    if vwap:
        _add(vwap, "VWAP")

    # ── Mean-reversion entries: from band extremes targeting middle ──────────
    if regime == "ranging":
        if bias == "bullish":
            entry_val = current_price
            entry_label = "Current Price (MR Long Entry)"
            # Stop: 1.5× ATR below entry
            stop_val = round(entry_val - config.atr_stop_mult * atr, 2)
            stop_label = f"ATR Stop ({config.atr_stop_mult}×ATR)"
            # Target: Bollinger mid or VWAP (mean)
            target_candidates = [
                (v, l) for v, l in resistance_levels
                if "Mid" in l or "VWAP" in l
            ]
            if not target_candidates:
                target_candidates = [(v, l) for v, l in resistance_levels]
        else:  # bearish MR short
            entry_val = current_price
            entry_label = "Current Price (MR Short Entry)"
            stop_val = round(entry_val + config.atr_stop_mult * atr, 2)
            stop_label = f"ATR Stop ({config.atr_stop_mult}×ATR)"
            target_candidates = [
                (v, l) for v, l in support_levels
                if "Mid" in l or "VWAP" in l
            ]
            if not target_candidates:
                target_candidates = [(v, l) for v, l in support_levels]

        stop_distance = abs(entry_val - stop_val)
        if stop_distance <= 0:
            return None

        if bias == "bullish":
            if target_candidates:
                target_val, target_label = min(target_candidates, key=lambda x: abs(x[0] - entry_val))
            else:
                target_val = round(entry_val + stop_distance * config.min_rr, 2)
                target_label = f"Min RR target ({config.min_rr}×)"
            reward_distance = target_val - entry_val
        else:
            if target_candidates:
                target_val, target_label = min(target_candidates, key=lambda x: abs(x[0] - entry_val))
            else:
                target_val = round(entry_val - stop_distance * config.min_rr, 2)
                target_label = f"Min RR target ({config.min_rr}×)"
            reward_distance = entry_val - target_val

    # ── Trend-following entries: structural level + ATR stop ─────────────────
    else:
        if bias == "bullish":
            candidates = [(v, l) for v, l in support_levels if v >= current_price - 2 * atr]
            if not candidates:
                return None
            entry_val, entry_label = max(candidates, key=lambda x: x[0])

            stop_val = round(entry_val - config.atr_stop_mult * atr, 2)
            stop_label = f"ATR Stop ({config.atr_stop_mult}×ATR below {entry_label})"

            stop_distance = entry_val - stop_val
            if stop_distance <= 0:
                return None

            target_min = entry_val + stop_distance * config.min_rr
            target_candidates = [(v, l) for v, l in resistance_levels if v >= target_min]
            if target_candidates:
                target_val, target_label = max(target_candidates, key=lambda x: x[0])
            else:
                target_val = round(entry_val + stop_distance * config.min_rr, 2)
                target_label = f"Min RR target ({config.min_rr}×)"

            reward_distance = target_val - entry_val

        else:  # bearish
            candidates = [(v, l) for v, l in resistance_levels if v <= current_price + 2 * atr]
            if not candidates:
                return None
            entry_val, entry_label = min(candidates, key=lambda x: x[0])

            stop_val = round(entry_val + config.atr_stop_mult * atr, 2)
            stop_label = f"ATR Stop ({config.atr_stop_mult}×ATR above {entry_label})"

            stop_distance = stop_val - entry_val
            if stop_distance <= 0:
                return None

            target_min = entry_val - stop_distance * config.min_rr
            target_candidates = [(v, l) for v, l in support_levels if v <= target_min]
            if target_candidates:
                target_val, target_label = min(target_candidates, key=lambda x: x[0])
            else:
                target_val = round(entry_val - stop_distance * config.min_rr, 2)
                target_label = f"Min RR target ({config.min_rr}×)"

            reward_distance = entry_val - target_val

    rr = round(reward_distance / stop_distance, 2) if stop_distance > 0 else 0.0

    # Determine trailing stop mechanism
    st_dir = (entry_tf_ind.get("supertrend") or {}).get("direction")
    st_val = _v(entry_tf_ind, "supertrend", "value")
    if st_val and st_dir:
        trail_via = "supertrend"
        trail_level = round(st_val, 2)
        trail_note = f"Trail stop: Supertrend @ {trail_level} (flip = exit)"
    else:
        trail_via = "ema20"
        trail_level = round(_v(entry_tf_ind, "ema20") or stop_val, 2)
        trail_note = f"Trail stop: EMA20 @ {trail_level} (close below = exit)"

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
        "trail_via": trail_via,
        "trail_level": trail_level,
        "trail_note": trail_note,
    }


# ── Multi-timeframe confluence score ──────────────────────────────────────────

def _confluence_score(timeframes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Score multi-timeframe alignment on a 0–10 scale."""
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


# ── Key levels ─────────────────────────────────────────────────────────────────

def _key_levels(primary_indicators: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidate all key price levels into one clean dict."""
    pivots = primary_indicators.get("pivot_points", {}) or {}
    levels_52w = primary_indicators.get("52w", {}) or {}
    boll = primary_indicators.get("bollinger", {}) or {}
    don55 = primary_indicators.get("donchian_55", {}) or {}
    ichi = primary_indicators.get("ichimoku", {}) or {}
    vwap_b = primary_indicators.get("vwap_bands", {}) or {}
    ib = primary_indicators.get("initial_balance", {}) or {}
    vwap = _v(primary_indicators, "vwap")
    svwap = _v(primary_indicators, "session_vwap")

    return {
        "vwap": vwap,
        "session_vwap": svwap,
        "vwap_bands": {
            "upper_1sd": vwap_b.get("upper_1sd"),
            "lower_1sd": vwap_b.get("lower_1sd"),
            "upper_2sd": vwap_b.get("upper_2sd"),
            "lower_2sd": vwap_b.get("lower_2sd"),
        },
        "initial_balance": {
            "high": ib.get("ib_high"),
            "low": ib.get("ib_low"),
            "midpoint": ib.get("ib_midpoint"),
            "range": ib.get("ib_range"),
        },
        "pivot": pivots.get("pivot"),
        "cpr": {"top": pivots.get("cpr_top"), "bottom": pivots.get("cpr_bottom")},
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
        "52w_dist_from_high_pct": levels_52w.get("dist_from_high_pct"),
        "52w_dist_from_low_pct": levels_52w.get("dist_from_low_pct"),
    }


# ── Human-readable summary ────────────────────────────────────────────────────

def _human_summary(
    confluence: Dict[str, Any],
    levels: Dict[str, Any],
    style_setups: Dict[str, Any],
    candlestick: Dict[str, Any | None],
    regime: str,
) -> str:
    """Build a detailed plain-English trading analysis summary."""
    bias = confluence["dominant_bias"].upper()
    score = confluence["score"]
    interp = confluence["interpretation"]
    tf_str = ", ".join(
        f"{tf.capitalize()}: {d.capitalize()}"
        for tf, d in confluence["per_timeframe"].items()
    )

    regime_desc = {
        "trending": "trending (ADX > 25)",
        "transition": "transitioning (ADX 20-25)",
        "ranging": "ranging/choppy (ADX < 20)",
    }.get(regime, regime)

    parts = [
        f"Market regime: {regime_desc}.",
        f"Dominant technical bias is {bias} with multi-timeframe confluence {score}/10, "
        f"indicating a {interp.lower()} environment.",
        f"Timeframe breakdown: [{tf_str}].",
    ]

    # Key Levels
    level_notes = []
    pivot = levels.get("pivot")
    if pivot:
        level_notes.append(f"Daily Pivot at {pivot}")
    svwap = levels.get("session_vwap") or levels.get("vwap")
    if svwap:
        level_notes.append(f"Session VWAP at {svwap:.2f}")
    ib = levels.get("initial_balance", {})
    if ib.get("high") and ib.get("low"):
        level_notes.append(f"IB range {ib['low']:.2f}–{ib['high']:.2f}")
    if level_notes:
        parts.append(f"Key structural levels: {', '.join(level_notes)}.")

    # Style Setups
    valid_setups = {s: v for s, v in style_setups.items() if v.get("setup")}
    rejected_styles = {s: v for s, v in style_setups.items() if not v.get("setup")}

    if valid_setups:
        parts.append(
            f"{len(valid_setups)} trading style(s) produced a valid setup: "
            + ", ".join(s.replace("_", " ").capitalize() for s in valid_setups.keys()) + "."
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
                f"Per-share reward: ₹{setup['per_share_reward']:.2f}. "
                f"{setup.get('trail_note', '')}."
            )
    else:
        if regime == "ranging":
            parts.append(
                "Market is in a ranging regime. Mean Reversion setups require "
                "price to be at Bollinger Band extremes or VWAP bands with RSI confirmation."
            )
        else:
            parts.append(
                "No high-confidence setup found across any trading style. "
                "Market appears to be in a consolidation or transition phase."
            )

    if rejected_styles:
        reasons = "; ".join(
            f"{s.replace('_', ' ').capitalize()}: {v.get('rejection_reason', 'filtered')}"
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


# ── Public API ────────────────────────────────────────────────────────────────

def generate_signals(
    timeframes: Dict[str, Dict[str, Any]],
    patterns: Dict[str, Any],
    config: SignalConfig | None = None,
    context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive signal report using the 4-step hierarchical filter.

    Steps per trading style:
      1. Trend & Regime Filter    — regime mode + direction hard block
      2. Structural Level Check   — price must be near a meaningful S/R zone
      3. Momentum + Volume Gate   — RVOL > 1.2× (hard) + MACD/RSI confirmation
      4. Dynamic Risk Management  — 1.5×ATR stop + Supertrend trailing exit

    Args:
        timeframes: The ``timeframes`` dict from :class:`TechnicalAgent.analyse`.
        patterns: The ``patterns`` dict from :class:`TechnicalAgent.analyse`.
        config: Optional :class:`SignalConfig`.

    Returns:
        Dict with keys: ``confluence``, ``regime``, ``style_setups``, ``levels``,
        ``candlestick_signals``, ``summary``.
    """
    cfg = config or SignalConfig()

    confluence = _confluence_score(timeframes)

    # Determine global regime from the daily timeframe ADX (most authoritative)
    primary = timeframes.get("daily") or next(iter(timeframes.values()), {})
    adx_val = _v(primary, "adx", "adx")
    global_regime = _regime_mode(adx_val)
    levels = _key_levels(primary)

    style_setups: Dict[str, Any] = {}

    for style_name, style_profile in TRADING_STYLES.items():
        bias_tf_data = timeframes.get(style_profile.bias_tf, {})
        structure_tf_data = timeframes.get(style_profile.structure_tf, {})
        entry_tf_data = timeframes.get(style_profile.entry_tf, {})

        # Skip styles whose required timeframes were not fetched
        if not bias_tf_data or not entry_tf_data:
            style_setups[style_name] = {
                "setup": None,
                "rejection_reason": (
                    f"Required timeframe '{style_profile.bias_tf}' or "
                    f"'{style_profile.entry_tf}' not fetched"
                ),
                "bias": "N/A",
                "regime": global_regime,
                "steps_passed": 0,
            }
            continue

        # ── STEP 1: Trend & Regime Filter ────────────────────────────────────
        passed1, direction, regime, reason1 = _step1_trend_filter(bias_tf_data, style_profile)
        if not passed1:
            style_setups[style_name] = {
                "setup": None,
                "rejection_reason": f"[Step 1] {reason1}",
                "bias": direction,
                "regime": regime,
                "steps_passed": 0,
            }
            continue

        # ── STEP 2: Structural Level Check ───────────────────────────────────
        passed2, level_desc = _step2_structural_check(
            entry_tf_data, bias_tf_data, direction, regime, cfg
        )
        if not passed2:
            style_setups[style_name] = {
                "setup": None,
                "rejection_reason": f"[Step 2] {level_desc}",
                "bias": direction,
                "regime": regime,
                "steps_passed": 1,
            }
            continue

        # ── STEP 3: Momentum + Volume Trigger ────────────────────────────────
        passed3, trigger_desc = _step3_momentum_trigger(
            entry_tf_data, direction, regime, cfg, context
        )
        if not passed3:
            style_setups[style_name] = {
                "setup": None,
                "rejection_reason": f"[Step 3] {trigger_desc}",
                "bias": direction,
                "regime": regime,
                "steps_passed": 2,
                "structural_level": level_desc,
            }
            continue

        # ── STEP 4: Build Setup with Dynamic Risk Management ─────────────────
        setup = _find_entry_and_levels(
            entry_tf_ind=entry_tf_data,
            structure_tf_ind=structure_tf_data or entry_tf_data,
            bias=direction,
            regime=regime,
            config=cfg,
        )

        if setup is None or setup.get("risk_reward", 0) < style_profile.min_rr:
            rr = setup.get("risk_reward", 0) if setup else 0
            style_setups[style_name] = {
                "setup": None,
                "rejection_reason": (
                    f"[Step 4] No valid entry level found or "
                    f"RR {rr}× < minimum {style_profile.min_rr}×"
                ),
                "bias": direction,
                "regime": regime,
                "steps_passed": 3,
                "structural_level": level_desc,
                "trigger": trigger_desc,
            }
            continue

        setup["bias"] = direction
        style_setups[style_name] = {
            "setup": setup,
            "rejection_reason": None,
            "bias": direction,
            "regime": regime,
            "steps_passed": 4,
            "structural_level": level_desc,
            "trigger": trigger_desc,
            "style_description": style_profile.description,
        }

    candlestick = patterns.get("candlestick", {}) or {}
    summary = _human_summary(confluence, levels, style_setups, candlestick, global_regime)

    return {
        "confluence": confluence,
        "regime": global_regime,
        "adx": adx_val,
        "style_setups": style_setups,
        "levels": levels,
        "candlestick_signals": {k: v for k, v in candlestick.items() if v},
        "summary": summary,
    }

