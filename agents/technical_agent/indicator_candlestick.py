"""indicator_candlestick.py

Generates individual PNG charts for each technical indicator, stored inside
the ``technical/graphs/`` subfolder of each run directory so they sit alongside
the candlestick PNG and interactive HTML produced by the rest of the pipeline.

Each indicator gets its own dedicated, high-quality chart:

  * EMA        – candlestick with EMA 20 / 50 / 200 overlaid
  * RSI        – line chart with overbought (70) / oversold (30) bands
  * MACD       – 3-panel: MACD line, signal line, and histogram
  * Bollinger  – candlestick with upper / middle / lower bands + shaded region
  * ATR        – standalone line chart
  * RVOL       – bar chart coloured by relative strength
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from orchestrator.output_paths import (
    ensure_category_directory,
    get_companies_root,
    normalise_symbol,
)
from .indicator_engine import IndicatorConfig, calculate_indicator_series
from .chart_generator import extract_candles

# ── palette ──────────────────────────────────────────────────────────────────
_BG        = "#ffffff"
_GRID      = "#e2e8f0"
_TEXT      = "#0f1117"
_UP        = "#16a34a"
_DOWN      = "#dc2626"
_EMA20     = "#d97706"
_EMA50     = "#0284c7"
_EMA200    = "#db2777"
_RSI_LINE  = "#7c3aed"
_RSI_OB    = "#dc262644"
_RSI_OS    = "#16a34a44"
_MACD_LINE = "#0284c7"
_SIG_LINE  = "#ea580c"
_HIST_POS  = "#16a34a66"
_HIST_NEG  = "#dc262666"
_BOLL_MID  = "#64748b"
_BOLL_FILL = "#0284c722"
_ATR_LINE  = "#ea580c"
_RVOL_MED  = "#0284c7"
_RVOL_HIGH = "#ea580c"

_DPI = 160
_W   = 16
_H   = 7

_DEFAULT_COMPANIES_ROOT = Path(__file__).resolve().parents[2] / "companies"


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_dataframe(candles: list[Any]) -> pd.DataFrame:
    """Convert raw candle rows (list-of-lists or list-of-dicts) to a DataFrame."""
    if not candles:
        return pd.DataFrame()

    first = candles[0]
    if isinstance(first, (list, tuple)):
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    else:
        df = pd.DataFrame(candles)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    return df.sort_values("time").reset_index(drop=True)


def _apply_style(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply consistent styling to an axes."""
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.grid(True, color=_GRID, linestyle="--", linewidth=0.6, alpha=0.9)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.title.set_color(_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)


def _draw_candlesticks(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Render OHLC candlesticks on *ax* using the DataFrame's integer x positions."""
    width = 0.6
    for i, row in df.iterrows():
        color = _UP if row["close"] >= row["open"] else _DOWN
        ax.vlines(i, row["low"], row["high"], color=color, linewidth=0.8, zorder=2)
        ax.add_patch(plt.Rectangle(
            (i - width / 2, min(row["open"], row["close"])),
            width,
            max(abs(row["close"] - row["open"]), 1e-6),
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
            alpha=0.95,
            zorder=3,
        ))


def _x_date_ticks(ax: plt.Axes, df: pd.DataFrame, n: int = 12) -> None:
    """Set evenly-spaced date tick labels on the x-axis."""
    step = max(len(df) // n, 1)
    positions = list(range(0, len(df), step))
    if positions and positions[-1] != len(df) - 1:
        positions.append(len(df) - 1)
    labels = [df.loc[p, "time"].strftime("%b %d") for p in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7, color=_TEXT)


def _save(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ── per-indicator chart functions ─────────────────────────────────────────────

def _chart_ema(df: pd.DataFrame, series: dict, symbol: str, output_path: Path) -> Path:
    """Candlestick chart with EMA 20 / 50 / 200 overlaid."""
    fig, ax = plt.subplots(figsize=(_W, _H))
    _apply_style(ax, fig)

    _draw_candlesticks(ax, df)

    x = np.arange(len(df))
    for name, color, lw in [
        ("ema20",  _EMA20,  1.4),
        ("ema50",  _EMA50,  1.6),
        ("ema200", _EMA200, 1.8),
    ]:
        s = series[name].values
        ax.plot(x, s, color=color, linewidth=lw, label=name.upper(), zorder=4)

    ax.set_xlim(-1, len(df) + 2)
    ax.set_title(f"{symbol}  —  EMA 20 / 50 / 200", color=_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel("Price", color=_TEXT)
    ax.legend(facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    _x_date_ticks(ax, df)
    fig.tight_layout()
    return _save(fig, output_path)


def _chart_rsi(df: pd.DataFrame, series: dict, symbol: str, output_path: Path) -> Path:
    """RSI line chart with overbought / oversold horizontal bands."""
    fig, ax = plt.subplots(figsize=(_W, _H * 0.65))
    _apply_style(ax, fig)

    x = np.arange(len(df))
    rsi_vals = series["rsi"].values

    ax.axhspan(70, 100, alpha=0.15, color="#ef4444", zorder=1)
    ax.axhspan(0,  30,  alpha=0.15, color="#22c55e", zorder=1)
    ax.axhline(70, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(30, color="#22c55e", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(50, color=_GRID,     linestyle=":",  linewidth=0.6, alpha=0.8)

    ax.plot(x, rsi_vals, color=_RSI_LINE, linewidth=1.5, label="RSI")

    ax.set_ylim(0, 100)
    ax.set_xlim(-1, len(df) + 2)
    ax.set_title(f"{symbol}  —  RSI (14)", color=_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel("RSI", color=_TEXT)
    ax.text(len(df) + 1, 73, "OB", color="#ef4444", fontsize=7, va="center")
    ax.text(len(df) + 1, 27, "OS", color="#22c55e", fontsize=7, va="center")
    _x_date_ticks(ax, df)
    fig.tight_layout()
    return _save(fig, output_path)


def _chart_macd(df: pd.DataFrame, series: dict, symbol: str, output_path: Path) -> Path:
    """3-panel MACD chart: MACD + signal, histogram, price reference."""
    macd_dict = series["macd"]
    macd_vals  = macd_dict["macd"].values
    sig_vals   = macd_dict["signal"].values
    hist_vals  = macd_dict["histogram"].values
    x = np.arange(len(df))

    fig = plt.figure(figsize=(_W, _H), constrained_layout=True)
    fig.patch.set_facecolor(_BG)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1.5, 0.8], figure=fig)

    # --- panel 0: MACD line + signal ---
    ax0 = fig.add_subplot(gs[0])
    _apply_style(ax0, fig)
    ax0.plot(x, macd_vals, color=_MACD_LINE, linewidth=1.4, label="MACD")
    ax0.plot(x, sig_vals,  color=_SIG_LINE,  linewidth=1.2, label="Signal", linestyle="--")
    ax0.axhline(0, color=_GRID, linewidth=0.6)
    ax0.set_xlim(-1, len(df) + 2)
    ax0.set_title(f"{symbol}  —  MACD (12, 26, 9)", color=_TEXT, fontsize=13, fontweight="bold")
    ax0.legend(facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    ax0.set_xticklabels([])

    # --- panel 1: histogram ---
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    _apply_style(ax1, fig)
    colors = [_HIST_POS if v >= 0 else _HIST_NEG for v in hist_vals]
    ax1.bar(x, hist_vals, color=colors, width=0.7)
    ax1.axhline(0, color=_GRID, linewidth=0.6)
    ax1.set_ylabel("Hist", color=_TEXT, fontsize=8)
    ax1.set_xticklabels([])

    # --- panel 2: price (sparkline) ---
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    _apply_style(ax2, fig)
    ax2.plot(x, df["close"].values, color=_TEXT, linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Close", color=_TEXT, fontsize=8)
    _x_date_ticks(ax2, df)

    return _save(fig, output_path)



def _chart_bollinger(df: pd.DataFrame, series: dict, symbol: str, output_path: Path) -> Path:
    """Candlestick chart with Bollinger Bands overlaid and band fill."""
    boll = series["bollinger"]
    upper  = boll["upper"].values
    middle = boll["middle"].values
    lower  = boll["lower"].values
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(_W, _H))
    _apply_style(ax, fig)

    _draw_candlesticks(ax, df)

    ax.fill_between(x, lower, upper, alpha=0.12, color="#38bdf8", zorder=1)
    ax.plot(x, upper,  color="#38bdf8", linewidth=1.2, linestyle="--", label="Upper",  zorder=4)
    ax.plot(x, middle, color=_BOLL_MID, linewidth=1.0, linestyle="-",  label="Middle", zorder=4)
    ax.plot(x, lower,  color="#38bdf8", linewidth=1.2, linestyle="--", label="Lower",  zorder=4)

    ax.set_xlim(-1, len(df) + 2)
    ax.set_title(f"{symbol}  —  Bollinger Bands (20, 2σ)", color=_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel("Price", color=_TEXT)
    ax.legend(facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    _x_date_ticks(ax, df)
    fig.tight_layout()
    return _save(fig, output_path)


def _chart_atr(df: pd.DataFrame, series: dict, symbol: str, output_path: Path) -> Path:
    """ATR line chart."""
    fig, ax = plt.subplots(figsize=(_W, _H * 0.65))
    _apply_style(ax, fig)

    x = np.arange(len(df))
    ax.plot(x, series["atr"].values, color=_ATR_LINE, linewidth=1.5, label="ATR (14)")
    ax.fill_between(x, series["atr"].values, alpha=0.15, color=_ATR_LINE)

    ax.set_xlim(-1, len(df) + 2)
    ax.set_title(f"{symbol}  —  ATR (14)", color=_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel("ATR", color=_TEXT)
    ax.legend(facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    _x_date_ticks(ax, df)
    fig.tight_layout()
    return _save(fig, output_path)


def _chart_rvol(df: pd.DataFrame, series: dict, symbol: str, output_path: Path) -> Path:
    """Relative volume bar chart — bars coloured orange when RVOL > 2×."""
    rvol_vals = series["rvol"].values
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(_W, _H * 0.65))
    _apply_style(ax, fig)

    colors = [_RVOL_HIGH if v >= 2.0 else _RVOL_MED for v in rvol_vals]
    ax.bar(x, rvol_vals, color=colors, width=0.7, alpha=0.85)
    ax.axhline(1.0, color=_TEXT,     linestyle="--", linewidth=0.8, alpha=0.6, label="1× (average)")
    ax.axhline(2.0, color=_RVOL_HIGH, linestyle=":",  linewidth=0.8, alpha=0.8, label="2× (high)")

    ax.set_xlim(-1, len(df) + 2)
    ax.set_title(f"{symbol}  —  Relative Volume (20-bar avg)", color=_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel("RVOL", color=_TEXT)
    ax.legend(facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    _x_date_ticks(ax, df)
    fig.tight_layout()
    return _save(fig, output_path)


# ── output path helper ────────────────────────────────────────────────────────

def output_indicator_graph_dir(
    symbol: str,
    run_timestamp: str | None,
    *,
    root: Path | None = None,
) -> Path:
    """Return ``technical/graphs/`` directory for *symbol* and *run_timestamp*.

    The directory is created (including parents) if it does not exist.
    Output sits at::

        companies/<SYMBOL>/<run_timestamp>/technical/graphs/
    """
    companies_root = get_companies_root(root or _DEFAULT_COMPANIES_ROOT)
    technical_dir, _ = ensure_category_directory(
        symbol,
        "technical",
        root=companies_root,
        run_timestamp=run_timestamp,
    )
    graphs_dir = technical_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    return graphs_dir


def output_indicator_graph_path(
    symbol: str,
    run_timestamp: str | None,
    *,
    root: Path | None = None,
    indicator_name: str,
) -> Path:
    """Return the full path for a single indicator PNG file."""
    safe_symbol = normalise_symbol(symbol)
    base_dir = output_indicator_graph_dir(symbol, run_timestamp, root=root)
    return base_dir / f"{safe_symbol}_{indicator_name}.png"


def _chart_trade_setup_for_style(
    df: pd.DataFrame,
    series: dict,
    symbol: str,
    output_path: Path,
    style_name: str,
    setup: dict,
) -> Path:
    """Candlestick chart with entry/stop/target zones for a specific trading style."""
    fig, ax = plt.subplots(figsize=(_W, _H))
    _apply_style(ax, fig)
    _draw_candlesticks(ax, df)

    entry = setup["entry"]
    sl = setup["stop_loss"]
    tp = setup["take_profit"]
    bias = setup.get("bias", "bullish")
    rr = setup["risk_reward"]
    per_share_risk = setup["per_share_risk"]
    per_share_reward = setup["per_share_reward"]

    x = np.arange(len(df))

    # Shaded zones (green = reward, red = risk)
    if bias == "bullish":
        ax.fill_between(x, entry, tp, color=_UP, alpha=0.12)
        ax.fill_between(x, entry, sl, color=_DOWN, alpha=0.12)
    else:
        ax.fill_between(x, entry, sl, color=_DOWN, alpha=0.12)
        ax.fill_between(x, entry, tp, color=_UP, alpha=0.12)

    # Horizontal level lines
    ax.axhline(tp, color=_UP,   linestyle="--", linewidth=1.3, alpha=0.9, label=f"Target  {tp}  ({setup.get('target_level', '')}")
    ax.axhline(entry, color="#334155", linestyle="-",  linewidth=1.6, alpha=0.9, label=f"Entry   {entry}  ({setup.get('entry_level', '')}")
    ax.axhline(sl, color=_DOWN, linestyle="--", linewidth=1.3, alpha=0.9, label=f"Stop    {sl}  ({setup.get('stop_level', '')}")

    # Right-side text annotations
    bbox = dict(boxstyle="round,pad=0.3", fc=_BG, ec=_GRID, alpha=0.92)
    xlast = len(df) + 0.5
    ax.text(xlast, tp,    f" TP  {tp}\n+₹{per_share_reward:.2f}/share",  color=_UP,     va="center", fontsize=8.5, bbox=bbox)
    ax.text(xlast, entry, f" ENTRY  {entry}",                              color="#334155",va="center", fontsize=8.5, bbox=bbox)
    ax.text(xlast, sl,    f" SL  {sl}\n-₹{per_share_risk:.2f}/share",    color=_DOWN,   va="center", fontsize=8.5, bbox=bbox)

    direction = "LONG ▲" if bias == "bullish" else "SHORT ▼"
    ax.set_xlim(-1, len(df) + 12)
    ax.set_title(
        f"{symbol}  —  {style_name.upper()} Setup  |  {direction}  |  RR: {rr}×",
        color=_TEXT, fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("Price", color=_TEXT)
    ax.legend(facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT, fontsize=7.5, loc="upper left")
    _x_date_ticks(ax, df)
    fig.tight_layout()
    return _save(fig, output_path)

def _chart_supertrend(df: pd.DataFrame, series: dict, symbol: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(_W, _H))
    _apply_style(ax, fig)
    _draw_candlesticks(ax, df)
    x = np.arange(len(df))
    st_dict = series.get("supertrend", {})
    st = st_dict.get("supertrend")
    if st is not None:
        ax.plot(x, st, color=_TEXT, linewidth=2, label="Supertrend", alpha=0.7)
    ax.set_xlim(-1, len(df) + 2)
    ax.set_title(f"{symbol}  —  Supertrend", color=_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel("Price", color=_TEXT)
    ax.legend(facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    _x_date_ticks(ax, df)
    fig.tight_layout()
    return _save(fig, output_path)

def _chart_ichimoku(df: pd.DataFrame, series: dict, symbol: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(_W, _H))
    _apply_style(ax, fig)
    _draw_candlesticks(ax, df)
    x = np.arange(len(df))
    
    ichi = series.get("ichimoku", {})
    tenkan = ichi.get("tenkan_sen")
    kijun = ichi.get("kijun_sen")
    senkou_a = ichi.get("senkou_a")
    senkou_b = ichi.get("senkou_b")
    
    if tenkan is not None: ax.plot(x, tenkan, color=_EMA20, linewidth=1, label="Tenkan")
    if kijun is not None: ax.plot(x, kijun, color=_EMA50, linewidth=1, label="Kijun")
    if senkou_a is not None and senkou_b is not None:
        ax.plot(x, senkou_a, color=_UP, linewidth=0.5, alpha=0.5)
        ax.plot(x, senkou_b, color=_DOWN, linewidth=0.5, alpha=0.5)
        ax.fill_between(x, senkou_a, senkou_b, where=(senkou_a >= senkou_b), color=_UP, alpha=0.15, interpolate=True)
        ax.fill_between(x, senkou_a, senkou_b, where=(senkou_a < senkou_b), color=_DOWN, alpha=0.15, interpolate=True)

    ax.set_xlim(-1, len(df) + 2)
    ax.set_title(f"{symbol}  —  Ichimoku Cloud", color=_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel("Price", color=_TEXT)
    ax.legend(facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    _x_date_ticks(ax, df)
    fig.tight_layout()
    return _save(fig, output_path)

def _chart_donchian(df: pd.DataFrame, series: dict, symbol: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(_W, _H))
    _apply_style(ax, fig)
    _draw_candlesticks(ax, df)
    x = np.arange(len(df))
    don20 = series.get("donchian_20", {})
    up = don20.get("upper")
    low = don20.get("lower")
    if up is not None and low is not None:
        ax.plot(x, up, color=_BOLL_MID, linestyle="--", linewidth=1.2, label="DC Upper (20)")
        ax.plot(x, low, color=_BOLL_MID, linestyle="--", linewidth=1.2, label="DC Lower (20)")
        ax.fill_between(x, low, up, color=_BOLL_FILL)
    ax.set_xlim(-1, len(df) + 2)
    ax.set_title(f"{symbol}  —  Donchian Channels (20)", color=_TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel("Price", color=_TEXT)
    ax.legend(facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    _x_date_ticks(ax, df)
    fig.tight_layout()
    return _save(fig, output_path)

# ── main entry-point ──────────────────────────────────────────────────────────

_CHART_FNS = {
    "ema":        _chart_ema,
    "supertrend": _chart_supertrend,
    "ichimoku":   _chart_ichimoku,
    "donchian":   _chart_donchian,
    "rsi":        _chart_rsi,
    "macd":       _chart_macd,
    "bollinger":  _chart_bollinger,
    "atr":        _chart_atr,
    "rvol":       _chart_rvol,
}


def generate_indicator_graphs(
    history_payload: dict[str, Any],
    symbol: str,
    *,
    run_timestamp: str | None = None,
    root: Path | None = None,
    config: IndicatorConfig | None = None,
    indicator_names: Iterable[str] | None = None,
    setups: dict[str, Any] | None = None,
) -> list[Path]:
    """Generate per-indicator PNG charts and a separate chart per valid trading style.

    Indicator charts are saved to::

        companies/<SYMBOL>/<run_timestamp>/technical/graphs/

    Per-style trade setup charts are saved to::

        companies/<SYMBOL>/<run_timestamp>/technical/setups/<style_name>/
    """
    raw_candles = extract_candles(history_payload)
    if not raw_candles:
        raise ValueError(f"generate_indicator_graphs: no candles found in payload for {symbol!r}")

    df = _to_dataframe(raw_candles)
    if df.empty:
        raise ValueError(f"generate_indicator_graphs: empty DataFrame after parsing candles for {symbol!r}")

    series = calculate_indicator_series(df, config=config)
    safe_symbol = normalise_symbol(symbol)

    names = list(indicator_names) if indicator_names is not None else list(_CHART_FNS.keys())
    out_paths: list[Path] = []

    # 1. Standard indicator charts  → technical/graphs/
    for name in names:
        chart_fn = _CHART_FNS.get(name)
        if chart_fn is None:
            continue
        out_path = output_indicator_graph_path(symbol, run_timestamp, root=root, indicator_name=name)
        out_paths.append(chart_fn(df, series, symbol, out_path))

    # 2. Per-style trade setup charts  → technical/setups/<style_name>/
    if setups:
        graphs_dir = output_indicator_graph_dir(symbol, run_timestamp, root=root)
        technical_dir = graphs_dir.parent  # one level up from graphs/
        for style_name, style_data in setups.items():
            setup = style_data.get("setup") if isinstance(style_data, dict) else None
            if not setup or not setup.get("entry"):
                continue  # skip styles that were rejected
            style_dir = technical_dir / "setups" / style_name
            style_dir.mkdir(parents=True, exist_ok=True)
            out_path = style_dir / f"{safe_symbol}_{style_name}_setup.png"
            out_paths.append(
                _chart_trade_setup_for_style(df, series, symbol, out_path, style_name, setup)
            )

    return out_paths
