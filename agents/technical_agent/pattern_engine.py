"""pattern_engine.py

Detects two classes of technical patterns:

  1. **Macro chart patterns** — structural patterns over many candles
     (breakout, double-bottom, H&S, wedges, triangles, etc.)

  2. **Candlestick micro-patterns** — 1-3 candle reversal / continuation signals
     (hammer, engulfing, doji, morning star, marubozu, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatternConfig:
    lookback: int = 120
    breakout_lookback: int = 30
    breakout_threshold: float = 0.005
    double_bottom_tolerance: float = 0.02
    double_bottom_separation: int = 10
    ascending_triangle_lookback: int = 60

    triple_lookback: int = 90
    triple_tolerance: float = 0.02
    triple_separation: int = 8

    head_shoulders_lookback: int = 90
    head_shoulders_shoulder_tolerance: float = 0.03
    head_shoulders_min_prominence: float = 0.02

    triangle_lookback: int = 60
    triangle_min_points: int = 20
    triangle_flat_tolerance: float = 0.02

    wedge_lookback: int = 60
    wedge_min_points: int = 20
    wedge_convergence_ratio: float = 0.5

    # candlestick micro-pattern thresholds
    doji_body_pct: float = 0.05
    hammer_body_pct: float = 0.35


def _recent(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    return df.tail(periods) if len(df) >= periods else df.copy()


def _find_local_extrema(values: np.ndarray, order: int = 3, kind: str = "max") -> np.ndarray:
    n = len(values)
    idxs = []
    for i in range(order, n - order):
        window = values[i - order : i + order + 1]
        if kind == "max" and values[i] == window.max() and (window == values[i]).sum() == 1:
            idxs.append(i)
        elif kind == "min" and values[i] == window.min() and (window == values[i]).sum() == 1:
            idxs.append(i)
    return np.array(idxs, dtype=int)


# ── macro chart patterns ──────────────────────────────────────────────────────

def detect_breakout(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    if len(df) < cfg.breakout_lookback + 1:
        return False
    segment = df.tail(cfg.breakout_lookback + 1)
    prior_high = segment["close"].iloc[:-1].max()
    last_close = segment["close"].iloc[-1]
    return bool(last_close > prior_high * (1 + cfg.breakout_threshold))


def detect_double_bottom(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    segment = _recent(df, cfg.lookback)
    if len(segment) < cfg.double_bottom_separation * 2:
        return False
    closes = segment["close"]
    trough_value = closes.min()
    tolerance = trough_value * cfg.double_bottom_tolerance
    trough_indices = closes[closes <= trough_value + tolerance].index
    if len(trough_indices) < 2:
        return False
    positions = sorted(segment.index.get_loc(idx) for idx in trough_indices)
    for i in range(len(positions) - 1):
        if positions[i + 1] - positions[i] >= cfg.double_bottom_separation:
            return True
    return False


def detect_cup_and_handle(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    segment = _recent(df, cfg.lookback)
    if len(segment) < 45:
        return False
    thirds = len(segment) // 3
    left = segment.iloc[:thirds]
    middle = segment.iloc[thirds : 2 * thirds]
    right = segment.iloc[2 * thirds :]
    left_high = left["close"].max()
    right_high = right["close"].max()
    cup_low = middle["close"].min()
    if cup_low >= left_high * 0.97:
        return False
    if abs(left_high - right_high) / left_high > 0.03:
        return False
    handle = right.tail(max(5, len(right) // 3))
    handle_pullback = (right_high - handle["close"].mean()) / right_high
    return bool(handle_pullback <= 0.05)


def detect_ascending_triangle(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    segment = _recent(df, cfg.ascending_triangle_lookback)
    if len(segment) < 20:
        return False
    highs = segment["high"].to_numpy()
    lows = segment["low"].to_numpy()
    x = np.arange(len(segment))
    top_threshold = np.percentile(highs, 90)
    top_zone = highs[highs >= top_threshold]
    if top_zone.size == 0:
        return False
    if top_zone.max() - top_zone.min() > top_zone.mean() * 0.02:
        return False
    slope = np.polyfit(x, lows, 1)[0]
    return bool(slope > 0)


def detect_descending_triangle(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    segment = _recent(df, cfg.triangle_lookback)
    if len(segment) < cfg.triangle_min_points:
        return False
    highs = segment["high"].to_numpy()
    lows = segment["low"].to_numpy()
    x = np.arange(len(segment))
    bottom_threshold = np.percentile(lows, 10)
    bottom_zone = lows[lows <= bottom_threshold]
    if bottom_zone.size == 0:
        return False
    if bottom_zone.max() - bottom_zone.min() > bottom_zone.mean() * cfg.triangle_flat_tolerance:
        return False
    slope = np.polyfit(x, highs, 1)[0]
    return bool(slope < 0)


def detect_symmetric_triangle(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    segment = _recent(df, cfg.triangle_lookback)
    if len(segment) < cfg.triangle_min_points:
        return False
    highs = segment["high"].to_numpy()
    lows = segment["low"].to_numpy()
    x = np.arange(len(segment))
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]
    range_start = highs[0] - lows[0]
    range_end = highs[-1] - lows[-1]
    if range_start <= 0:
        return False
    converging = range_end < range_start * 0.7
    return bool(high_slope < 0 and low_slope > 0 and converging)


def detect_rising_wedge(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    segment = _recent(df, cfg.wedge_lookback)
    if len(segment) < cfg.wedge_min_points:
        return False
    highs = segment["high"].to_numpy()
    lows = segment["low"].to_numpy()
    x = np.arange(len(segment))
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]
    if high_slope <= 0 or low_slope <= 0:
        return False
    range_start = highs[0] - lows[0]
    range_end = highs[-1] - lows[-1]
    if range_start <= 0:
        return False
    narrowing = range_end < range_start * cfg.wedge_convergence_ratio
    slower_top = high_slope < low_slope
    return bool(narrowing and slower_top)


def detect_falling_wedge(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    segment = _recent(df, cfg.wedge_lookback)
    if len(segment) < cfg.wedge_min_points:
        return False
    highs = segment["high"].to_numpy()
    lows = segment["low"].to_numpy()
    x = np.arange(len(segment))
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]
    if high_slope >= 0 or low_slope >= 0:
        return False
    range_start = highs[0] - lows[0]
    range_end = highs[-1] - lows[-1]
    if range_start <= 0:
        return False
    narrowing = range_end < range_start * cfg.wedge_convergence_ratio
    slower_bottom = abs(low_slope) < abs(high_slope)
    return bool(narrowing and slower_bottom)


def detect_triple_top(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    segment = _recent(df, cfg.triple_lookback)
    if len(segment) < cfg.triple_separation * 3:
        return False
    closes = segment["close"]
    peak_value = closes.max()
    tolerance = peak_value * cfg.triple_tolerance
    peak_indices = closes[closes >= peak_value - tolerance].index
    if len(peak_indices) < 3:
        return False
    positions = sorted(segment.index.get_loc(idx) for idx in peak_indices)
    spaced = [positions[0]]
    for pos in positions[1:]:
        if pos - spaced[-1] >= cfg.triple_separation:
            spaced.append(pos)
    return len(spaced) >= 3


def detect_triple_bottom(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    segment = _recent(df, cfg.triple_lookback)
    if len(segment) < cfg.triple_separation * 3:
        return False
    closes = segment["close"]
    trough_value = closes.min()
    tolerance = trough_value * cfg.triple_tolerance
    trough_indices = closes[closes <= trough_value + tolerance].index
    if len(trough_indices) < 3:
        return False
    positions = sorted(segment.index.get_loc(idx) for idx in trough_indices)
    spaced = [positions[0]]
    for pos in positions[1:]:
        if pos - spaced[-1] >= cfg.triple_separation:
            spaced.append(pos)
    return len(spaced) >= 3


def detect_head_and_shoulders(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    segment = _recent(df, cfg.head_shoulders_lookback)
    if len(segment) < 30:
        return False
    highs = segment["high"].to_numpy()
    peak_idx = _find_local_extrema(highs, order=3, kind="max")
    if len(peak_idx) < 3:
        return False
    top3 = sorted(peak_idx[np.argsort(highs[peak_idx])[-3:]])
    left_val, head_val, right_val = highs[top3[0]], highs[top3[1]], highs[top3[2]]
    if not (head_val > left_val and head_val > right_val):
        return False
    shoulder_diff = abs(left_val - right_val) / max(left_val, right_val)
    prominence = (head_val - max(left_val, right_val)) / head_val
    return bool(shoulder_diff <= cfg.head_shoulders_shoulder_tolerance
                and prominence >= cfg.head_shoulders_min_prominence)


def detect_inverse_head_and_shoulders(df: pd.DataFrame, cfg: PatternConfig) -> bool:
    segment = _recent(df, cfg.head_shoulders_lookback)
    if len(segment) < 30:
        return False
    lows = segment["low"].to_numpy()
    trough_idx = _find_local_extrema(lows, order=3, kind="min")
    if len(trough_idx) < 3:
        return False
    bottom3 = sorted(trough_idx[np.argsort(lows[trough_idx])[:3]])
    left_val, head_val, right_val = lows[bottom3[0]], lows[bottom3[1]], lows[bottom3[2]]
    if not (head_val < left_val and head_val < right_val):
        return False
    shoulder_diff = abs(left_val - right_val) / max(left_val, right_val)
    prominence = (min(left_val, right_val) - head_val) / head_val
    return bool(shoulder_diff <= cfg.head_shoulders_shoulder_tolerance
                and prominence >= cfg.head_shoulders_min_prominence)


def identify_levels(df: pd.DataFrame, lookback: int = 40) -> Dict[str, float]:
    segment = _recent(df, lookback)
    if segment.empty:
        return {"support": None, "resistance": None}
    return {
        "support": float(segment["low"].min()),
        "resistance": float(segment["high"].max()),
    }


# ── candlestick micro-patterns ────────────────────────────────────────────────

def _candle_parts(row: pd.Series) -> tuple[float, float, float, float]:
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    body = abs(c - o)
    rng = max(h - l, 1e-6)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    return body, upper_shadow, lower_shadow, rng


def detect_doji(df: pd.DataFrame, cfg: PatternConfig) -> str | None:
    if len(df) < 1:
        return None
    body, upper, lower, rng = _candle_parts(df.iloc[-1])
    if body / rng > cfg.doji_body_pct:
        return None
    if lower > rng * 0.6 and upper < rng * 0.1:
        return "dragonfly_doji"
    if upper > rng * 0.6 and lower < rng * 0.1:
        return "gravestone_doji"
    return "doji"


def detect_hammer(df: pd.DataFrame, cfg: PatternConfig) -> str | None:
    if len(df) < 1:
        return None
    body, upper, lower, rng = _candle_parts(df.iloc[-1])
    if body / rng > cfg.hammer_body_pct or lower < rng * 0.6 or upper > rng * 0.1:
        return None
    trend = df["close"].tail(10).diff().mean()
    return "hammer" if trend < 0 else "hanging_man"


def detect_shooting_star(df: pd.DataFrame, cfg: PatternConfig) -> str | None:
    if len(df) < 1:
        return None
    body, upper, lower, rng = _candle_parts(df.iloc[-1])
    if body / rng > cfg.hammer_body_pct or upper < rng * 0.6 or lower > rng * 0.1:
        return None
    trend = df["close"].tail(10).diff().mean()
    return "shooting_star" if trend > 0 else "inverted_hammer"


def detect_engulfing(df: pd.DataFrame, cfg: PatternConfig) -> str | None:
    if len(df) < 2:
        return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    prev_body = abs(prev["close"] - prev["open"])
    curr_body = abs(curr["close"] - curr["open"])
    if curr_body <= prev_body:
        return None
    prev_bull = prev["close"] > prev["open"]
    curr_bull = curr["close"] > curr["open"]
    if not prev_bull and curr_bull and curr["open"] < prev["close"] and curr["close"] > prev["open"]:
        return "bullish_engulfing"
    if prev_bull and not curr_bull and curr["open"] > prev["close"] and curr["close"] < prev["open"]:
        return "bearish_engulfing"
    return None


def detect_morning_evening_star(df: pd.DataFrame, cfg: PatternConfig) -> str | None:
    if len(df) < 3:
        return None
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    b1 = abs(c1["close"] - c1["open"])
    b2 = abs(c2["close"] - c2["open"])
    b3 = abs(c3["close"] - c3["open"])
    if b2 > b1 * 0.5 or b2 > b3 * 0.5:
        return None
    if c1["close"] < c1["open"] and c3["close"] > c3["open"]:
        if c3["close"] > (c1["open"] + c1["close"]) / 2:
            return "morning_star"
    if c1["close"] > c1["open"] and c3["close"] < c3["open"]:
        if c3["close"] < (c1["open"] + c1["close"]) / 2:
            return "evening_star"
    return None


def detect_marubozu(df: pd.DataFrame, cfg: PatternConfig) -> str | None:
    if len(df) < 1:
        return None
    body, upper, lower, rng = _candle_parts(df.iloc[-1])
    if body / rng < 0.90:
        return None
    row = df.iloc[-1]
    return "bullish_marubozu" if row["close"] > row["open"] else "bearish_marubozu"


def detect_three_soldiers_crows(df: pd.DataFrame, cfg: PatternConfig) -> str | None:
    if len(df) < 3:
        return None
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    all_bull = all(r["close"] > r["open"] for r in [c1, c2, c3])
    all_bear = all(r["close"] < r["open"] for r in [c1, c2, c3])
    if all_bull:
        if c2["close"] > c1["close"] and c3["close"] > c2["close"]:
            if c2["open"] > c1["open"] and c3["open"] > c2["open"]:
                return "three_white_soldiers"
    if all_bear:
        if c2["close"] < c1["close"] and c3["close"] < c2["close"]:
            if c2["open"] < c1["open"] and c3["open"] < c2["open"]:
                return "three_black_crows"
    return None


def detect_candlestick_patterns(df: pd.DataFrame, cfg: PatternConfig | None = None) -> Dict[str, str | None]:
    """Run all micro-pattern detectors on the last 1-3 candles."""
    cfg = cfg or PatternConfig()
    return {
        "doji": detect_doji(df, cfg),
        "hammer_hanging_man": detect_hammer(df, cfg),
        "shooting_star_inv_hammer": detect_shooting_star(df, cfg),
        "engulfing": detect_engulfing(df, cfg),
        "morning_evening_star": detect_morning_evening_star(df, cfg),
        "marubozu": detect_marubozu(df, cfg),
        "three_candle": detect_three_soldiers_crows(df, cfg),
    }


# ── main entry point ──────────────────────────────────────────────────────────

def detect_patterns(
    df: pd.DataFrame, config: PatternConfig | None = None
) -> Dict[str, bool | Dict]:
    cfg = config or PatternConfig()
    if df.empty:
        raise ValueError("pattern_engine.detect_patterns requires non-empty dataframe")

    return {
        "breakout": detect_breakout(df, cfg),
        "double_bottom": detect_double_bottom(df, cfg),
        "cup_and_handle": detect_cup_and_handle(df, cfg),
        "ascending_triangle": detect_ascending_triangle(df, cfg),
        "descending_triangle": detect_descending_triangle(df, cfg),
        "symmetric_triangle": detect_symmetric_triangle(df, cfg),
        "rising_wedge": detect_rising_wedge(df, cfg),
        "falling_wedge": detect_falling_wedge(df, cfg),
        "triple_top": detect_triple_top(df, cfg),
        "triple_bottom": detect_triple_bottom(df, cfg),
        "head_and_shoulders": detect_head_and_shoulders(df, cfg),
        "inverse_head_and_shoulders": detect_inverse_head_and_shoulders(df, cfg),
        "levels": identify_levels(df),
        "candlestick": detect_candlestick_patterns(df, cfg),
    }