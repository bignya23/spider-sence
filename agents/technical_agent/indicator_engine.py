"""indicator_engine.py

Calculates a comprehensive suite of technical indicators for professional
trading and investing.  Two API surfaces are provided:

  * :func:`calculate_indicators`        — last scalar per indicator (for JSON)
  * :func:`calculate_indicator_series`  — full Pandas Series (for charts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IndicatorConfig:
    # EMA / MACD
    ema_fast: int = 12
    ema_slow: int = 26
    ema_signal: int = 9
    ema_short: int = 20
    ema_medium: int = 50
    ema_long: int = 200
    # Momentum
    rsi_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth: int = 3
    williams_r_period: int = 14
    cci_period: int = 20
    # Volatility
    atr_period: int = 14
    bollinger_period: int = 20
    bollinger_stddev: float = 2.0
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    donchian_short: int = 20
    donchian_long: int = 55
    # Volume
    rvol_period: int = 20
    mfi_period: int = 14
    # Trend
    adx_period: int = 14
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52
    # Pivot Points
    pivot_type: str = "classic"  # classic | woodie | camarilla


# ── helpers ───────────────────────────────────────────────────────────────────

def _last_valid(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _bollinger_bands(series: pd.Series, period: int, stddev: float) -> pd.DataFrame:
    ma = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    return pd.DataFrame({"upper": ma + stddev * std, "middle": ma, "lower": ma - stddev * std})


def _stochastic(df: pd.DataFrame, k_period: int, d_period: int, smooth: int) -> pd.DataFrame:
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    k_raw = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = k_raw.rolling(smooth, min_periods=1).mean()
    d = k.rolling(d_period, min_periods=1).mean()
    return pd.DataFrame({"k": k, "d": d})


def _williams_r(df: pd.DataFrame, period: int) -> pd.Series:
    high_max = df["high"].rolling(period, min_periods=period).max()
    low_min = df["low"].rolling(period, min_periods=period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)


def _cci(df: pd.DataFrame, period: int) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    ma = typical.rolling(period, min_periods=period).mean()
    mad = typical.rolling(period, min_periods=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (typical - ma) / (0.015 * mad.replace(0, np.nan))


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).cumsum()


def _mfi(df: pd.DataFrame, period: int) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf = typical * df["volume"]
    pos_mf = raw_mf.where(typical > typical.shift(1), 0)
    neg_mf = raw_mf.where(typical < typical.shift(1), 0)
    pos_sum = pos_mf.rolling(period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(period, min_periods=period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP from the first bar in the dataframe."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical * df["volume"]).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP — resets at the start of each trading day.

    Requires a 'datetime' column (tz-naive, IST assumed).  If the dataframe
    spans multiple calendar days the VWAP resets at 09:15 of each day so
    intraday price action always has an accurate institutional reference.
    """
    if "datetime" not in df.columns or df.empty:
        return _vwap(df)  # graceful fallback

    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]

    # Determine the session date: bars between 09:15 and 15:30 belong to that
    # calendar date.  Bars before 09:15 are attributed to the previous session.
    dates = pd.to_datetime(df["datetime"])
    session_key = dates.dt.date.astype(str) + "_" + (
        # bars before 09:15 get pushed to the 'previous' key so they don't
        # reset the accumulator mid-session
        (dates.dt.hour * 60 + dates.dt.minute >= 9 * 60 + 15)
        .cumsum()
        .astype(str)
    )

    cum_tp_vol = tp_vol.groupby(session_key).cumsum()
    cum_vol = df["volume"].groupby(session_key).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP Standard Deviation Bands (±1σ, ±2σ).

    Returns a DataFrame with columns: vwap, upper_1sd, lower_1sd,
    upper_2sd, lower_2sd.  These act as dynamic mean-reversion zones
    used by institutional prop desks for entry/exit.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan)
    cum_vol = vol.cumsum()
    cum_tp_vol = (typical * vol).cumsum()
    vwap = cum_tp_vol / cum_vol

    # Variance: Σ(weight × (price − vwap)²) / Σ(weight)
    cum_var = ((typical - vwap) ** 2 * vol).cumsum() / cum_vol
    sd = np.sqrt(cum_var.clip(lower=0))

    return pd.DataFrame({
        "vwap": vwap,
        "upper_1sd": vwap + 1 * sd,
        "lower_1sd": vwap - 1 * sd,
        "upper_2sd": vwap + 2 * sd,
        "lower_2sd": vwap - 2 * sd,
    })


def calc_initial_balance(
    df: pd.DataFrame,
    ib_minutes: int = 60,
) -> Dict[str, float | None]:
    """Calculate the Initial Balance (first N minutes of the trading session).

    The Initial Balance High/Low defines the "auction range" set in the
    opening period.  A breakout above IB_High = bullish impulse; below
    IB_Low = bearish.  Requires a 'datetime' column.

    Args:
        df: OHLCV DataFrame with a 'datetime' column (tz-naive, IST).
        ib_minutes: Duration of the initial balance period in minutes (default: 60).

    Returns:
        Dict with keys: ib_high, ib_low, ib_midpoint, ib_range.
    """
    if "datetime" not in df.columns or df.empty:
        return {"ib_high": None, "ib_low": None, "ib_midpoint": None, "ib_range": None}

    dates = pd.to_datetime(df["datetime"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
        
    # Find the most recent trading day in the data
    last_date = dates.dt.date.max()

    # Market open: 09:15 IST
    session_open = pd.Timestamp(f"{last_date} 09:15:00")
    ib_end = session_open + pd.Timedelta(minutes=ib_minutes)

    ib_bars = df[(dates >= session_open) & (dates < ib_end)]
    if ib_bars.empty:
        return {"ib_high": None, "ib_low": None, "ib_midpoint": None, "ib_range": None}

    ib_high = float(ib_bars["high"].max())
    ib_low = float(ib_bars["low"].min())
    ib_mid = round((ib_high + ib_low) / 2, 2)
    ib_range = round(ib_high - ib_low, 2)

    return {
        "ib_high": round(ib_high, 2),
        "ib_low": round(ib_low, 2),
        "ib_midpoint": ib_mid,
        "ib_range": ib_range,
    }


def _adx(df: pd.DataFrame, period: int) -> pd.DataFrame:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    dm_plus = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    alpha = 1 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    di_plus = 100 * dm_plus.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_s.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return pd.DataFrame({"adx": adx, "di_plus": di_plus, "di_minus": di_minus})


def _supertrend(df: pd.DataFrame, period: int, multiplier: float) -> pd.DataFrame:
    atr = _atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)  # 1=bullish, -1=bearish

    for i in range(1, len(df)):
        prev_upper = upper_band.iloc[i - 1]
        prev_lower = lower_band.iloc[i - 1]
        close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]

        # Adjust bands
        cur_upper = upper_band.iloc[i]
        cur_lower = lower_band.iloc[i]
        if cur_upper > prev_upper or prev_close > prev_upper:
            upper_band.iloc[i] = cur_upper
        else:
            upper_band.iloc[i] = prev_upper

        if cur_lower < prev_lower or prev_close < prev_lower:
            lower_band.iloc[i] = cur_lower
        else:
            lower_band.iloc[i] = prev_lower

        prev_st = supertrend.iloc[i - 1] if i > 1 else upper_band.iloc[i]
        if pd.isna(prev_st):
            prev_st = upper_band.iloc[i]

        if prev_st == prev_upper:
            if close <= upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
        else:
            if close >= lower_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1

    return pd.DataFrame({"supertrend": supertrend, "direction": direction})


def _ichimoku(df: pd.DataFrame, tenkan: int, kijun: int, senkou_b: int) -> pd.DataFrame:
    def mid(high, low, n):
        return (high.rolling(n).max() + low.rolling(n).min()) / 2

    tenkan_sen = mid(df["high"], df["low"], tenkan)
    kijun_sen = mid(df["high"], df["low"], kijun)
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_line = mid(df["high"], df["low"], senkou_b).shift(kijun)
    chikou_span = df["close"].shift(-kijun)
    return pd.DataFrame({
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b_line,
        "chikou_span": chikou_span,
    })


def _donchian(df: pd.DataFrame, period: int) -> pd.DataFrame:
    upper = df["high"].rolling(period, min_periods=period).max()
    lower = df["low"].rolling(period, min_periods=period).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2})


def _pivot_points(df: pd.DataFrame) -> Dict[str, float | None]:
    """Classic daily pivot points from the last complete candle."""
    if len(df) < 2:
        return {}
    prev = df.iloc[-2]
    h, l, c = float(prev["high"]), float(prev["low"]), float(prev["close"])
    pivot = (h + l + c) / 3
    r1 = 2 * pivot - l
    r2 = pivot + (h - l)
    r3 = h + 2 * (pivot - l)
    s1 = 2 * pivot - h
    s2 = pivot - (h - l)
    s3 = l - 2 * (h - pivot)
    bc = (h + l) / 2        # Bottom CPR
    tc = (pivot - bc) + pivot   # Top CPR
    return {
        "pivot": round(pivot, 2),
        "r1": round(r1, 2), "r2": round(r2, 2), "r3": round(r3, 2),
        "s1": round(s1, 2), "s2": round(s2, 2), "s3": round(s3, 2),
        "cpr_top": round(max(bc, tc), 2),
        "cpr_bottom": round(min(bc, tc), 2),
    }


def _52w_levels(df: pd.DataFrame) -> Dict[str, float | None]:
    """52-week high/low and distance from current price."""
    if df.empty:
        return {}
    tail_252 = df.tail(252)
    high_52w = float(tail_252["high"].max())
    low_52w = float(tail_252["low"].min())
    close = float(df["close"].iloc[-1])
    dist_from_high = round((close - high_52w) / high_52w * 100, 2) if high_52w else None
    dist_from_low = round((close - low_52w) / low_52w * 100, 2) if low_52w else None
    return {
        "high": round(high_52w, 2),
        "low": round(low_52w, 2),
        "dist_from_high_pct": dist_from_high,
        "dist_from_low_pct": dist_from_low,
    }


# ── public API — full series (for charts) ────────────────────────────────────

def calculate_indicator_series(
    df: pd.DataFrame, config: IndicatorConfig | None = None
) -> Dict[str, "pd.Series | Dict[str, pd.Series]"]:
    """Return full time-series for every indicator as Pandas Series.

    Composite indicators return a nested dict of named sub-series.
    """
    if df.empty:
        raise ValueError("calculate_indicator_series requires non-empty dataframe")

    cfg = config or IndicatorConfig()
    closes = df["close"].astype(float)
    volumes = df["volume"].astype(float)

    ema20 = _ema(closes, cfg.ema_short)
    ema50 = _ema(closes, cfg.ema_medium)
    ema200 = _ema(closes, cfg.ema_long)

    ema_fast = _ema(closes, cfg.ema_fast)
    ema_slow = _ema(closes, cfg.ema_slow)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=cfg.ema_signal, adjust=False).mean()

    boll = _bollinger_bands(closes, cfg.bollinger_period, cfg.bollinger_stddev)
    stoch = _stochastic(df, cfg.stoch_k_period, cfg.stoch_d_period, cfg.stoch_smooth)
    adx_df = _adx(df, cfg.adx_period)
    st_df = _supertrend(df, cfg.supertrend_period, cfg.supertrend_multiplier)
    ichi = _ichimoku(df, cfg.ichimoku_tenkan, cfg.ichimoku_kijun, cfg.ichimoku_senkou_b)
    don20 = _donchian(df, cfg.donchian_short)
    don55 = _donchian(df, cfg.donchian_long)
    volume_ma = volumes.rolling(window=cfg.rvol_period, min_periods=1).mean()

    vwap_b = _vwap_bands(df)
    sv = _session_vwap(df)

    return {
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": _rsi(closes, cfg.rsi_period),
        "macd": {
            "macd": macd_line,
            "signal": macd_signal,
            "histogram": macd_line - macd_signal,
        },
        "atr": _atr(df, cfg.atr_period),
        "bollinger": {
            "upper": boll["upper"],
            "middle": boll["middle"],
            "lower": boll["lower"],
        },
        "stochastic": {"k": stoch["k"], "d": stoch["d"]},
        "williams_r": _williams_r(df, cfg.williams_r_period),
        "cci": _cci(df, cfg.cci_period),
        "obv": _obv(df),
        "mfi": _mfi(df, cfg.mfi_period),
        "vwap": _vwap(df),
        "session_vwap": sv,
        "vwap_bands": {
            "upper_1sd": vwap_b["upper_1sd"],
            "lower_1sd": vwap_b["lower_1sd"],
            "upper_2sd": vwap_b["upper_2sd"],
            "lower_2sd": vwap_b["lower_2sd"],
        },
        "rvol": volumes / volume_ma,
        "adx": {"adx": adx_df["adx"], "di_plus": adx_df["di_plus"], "di_minus": adx_df["di_minus"]},
        "supertrend": {"supertrend": st_df["supertrend"], "direction": st_df["direction"]},
        "ichimoku": {
            "tenkan_sen": ichi["tenkan_sen"],
            "kijun_sen": ichi["kijun_sen"],
            "senkou_a": ichi["senkou_a"],
            "senkou_b": ichi["senkou_b"],
            "chikou_span": ichi["chikou_span"],
        },
        "donchian_20": {"upper": don20["upper"], "lower": don20["lower"], "mid": don20["mid"]},
        "donchian_55": {"upper": don55["upper"], "lower": don55["lower"], "mid": don55["mid"]},
    }


# ── public API — scalar summary (for JSON output) ────────────────────────────

def calculate_indicators(
    df: pd.DataFrame, config: IndicatorConfig | None = None
) -> Dict[str, float | Dict[str, float | None] | None]:
    """Return the last valid scalar value for every indicator.

    Used by :class:`TechnicalAgent` to produce the JSON analysis output.
    """
    if df.empty:
        raise ValueError("indicator_engine.calculate_indicators requires non-empty dataframe")

    cfg = config or IndicatorConfig()
    closes = df["close"].astype(float)
    volumes = df["volume"].astype(float)

    # ── trend ──────────────────────────────────────────────────────────────
    ema20 = _ema(closes, cfg.ema_short)
    ema50 = _ema(closes, cfg.ema_medium)
    ema200 = _ema(closes, cfg.ema_long)

    ema_fast = _ema(closes, cfg.ema_fast)
    ema_slow = _ema(closes, cfg.ema_slow)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=cfg.ema_signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    adx_df = _adx(df, cfg.adx_period)
    st_df = _supertrend(df, cfg.supertrend_period, cfg.supertrend_multiplier)

    ichi = _ichimoku(df, cfg.ichimoku_tenkan, cfg.ichimoku_kijun, cfg.ichimoku_senkou_b)
    pivots = _pivot_points(df)
    levels_52w = _52w_levels(df)

    # ── momentum ───────────────────────────────────────────────────────────
    rsi = _rsi(closes, cfg.rsi_period)
    stoch = _stochastic(df, cfg.stoch_k_period, cfg.stoch_d_period, cfg.stoch_smooth)
    wr = _williams_r(df, cfg.williams_r_period)
    cci = _cci(df, cfg.cci_period)

    # ── volatility ─────────────────────────────────────────────────────────
    atr = _atr(df, cfg.atr_period)
    atr_pct = atr / closes * 100  # normalized ATR
    boll = _bollinger_bands(closes, cfg.bollinger_period, cfg.bollinger_stddev)
    don20 = _donchian(df, cfg.donchian_short)
    don55 = _donchian(df, cfg.donchian_long)

    # ── volume ─────────────────────────────────────────────────────────────
    volume_ma = volumes.rolling(window=cfg.rvol_period, min_periods=1).mean()
    rvol = volumes / volume_ma
    obv = _obv(df)
    mfi = _mfi(df, cfg.mfi_period)
    vwap = _vwap(df)

    upper = _last_valid(boll["upper"])
    lower = _last_valid(boll["lower"])

    # Supertrend direction as string
    st_dir_val = _last_valid(st_df["direction"])
    st_direction = "bullish" if st_dir_val and st_dir_val > 0 else "bearish" if st_dir_val and st_dir_val < 0 else None

    # ADX trend strength label
    adx_val = _last_valid(adx_df["adx"])
    adx_trend = "strong" if adx_val and adx_val > 25 else "weak" if adx_val and adx_val < 20 else "moderate"

    return {
        "close": _last_valid(df["close"]),
        "open": _last_valid(df["open"]),
        "high": _last_valid(df["high"]),
        "low": _last_valid(df["low"]),
        # ── trend
        "ema20": _last_valid(ema20),
        "ema50": _last_valid(ema50),
        "ema200": _last_valid(ema200),
        "adx": {
            "adx": _last_valid(adx_df["adx"]),
            "di_plus": _last_valid(adx_df["di_plus"]),
            "di_minus": _last_valid(adx_df["di_minus"]),
            "trend_strength": adx_trend,
        },
        "supertrend": {
            "value": _last_valid(st_df["supertrend"]),
            "direction": st_direction,
        },
        "ichimoku": {
            "tenkan_sen": _last_valid(ichi["tenkan_sen"]),
            "kijun_sen": _last_valid(ichi["kijun_sen"]),
            "senkou_a": _last_valid(ichi["senkou_a"]),
            "senkou_b": _last_valid(ichi["senkou_b"]),
        },
        # ── momentum
        "rsi": _last_valid(rsi),
        "stochastic": {
            "k": _last_valid(stoch["k"]),
            "d": _last_valid(stoch["d"]),
        },
        "williams_r": _last_valid(wr),
        "cci": _last_valid(cci),
        "macd": {
            "macd": _last_valid(macd_line),
            "signal": _last_valid(macd_signal),
            "histogram": _last_valid(macd_hist),
        },
        # ── volatility
        "atr": _last_valid(atr),
        "atr_pct": round(_last_valid(atr_pct), 4) if _last_valid(atr_pct) else None,
        "bollinger": {
            "upper": upper,
            "middle": _last_valid(boll["middle"]),
            "lower": lower,
            "width": round(float(upper - lower), 4) if upper is not None and lower is not None else None,
        },
        "donchian_20": {
            "upper": _last_valid(don20["upper"]),
            "lower": _last_valid(don20["lower"]),
        },
        "donchian_55": {
            "upper": _last_valid(don55["upper"]),
            "lower": _last_valid(don55["lower"]),
        },
        # ── volume
        "rvol": _last_valid(rvol),
        "obv": _last_valid(obv),
        "mfi": _last_valid(mfi),
        "vwap": _last_valid(vwap),
        # ── session vwap & bands
        "session_vwap": _last_valid(_session_vwap(df)),
        "vwap_bands": {
            "upper_1sd": _last_valid(_vwap_bands(df)["upper_1sd"]),
            "lower_1sd": _last_valid(_vwap_bands(df)["lower_1sd"]),
            "upper_2sd": _last_valid(_vwap_bands(df)["upper_2sd"]),
            "lower_2sd": _last_valid(_vwap_bands(df)["lower_2sd"]),
        },
        # ── initial balance (only meaningful for intraday TFs)
        "initial_balance": calc_initial_balance(df),
        # ── levels
        "pivot_points": pivots,
        "52w": levels_52w,
    }
