from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, StrMethodFormatter
import pandas as pd

from orchestrator.output_paths import ensure_category_directory, get_companies_root, normalise_symbol

DEFAULT_COMPANIES_ROOT = Path(__file__).resolve().parents[2] / "companies"


@dataclass(frozen=True)
class CandleChartStyle:
    background: str = "#ffffff"
    grid: str = "#e5e7eb"
    text: str = "#111827"
    up: str = "#16a34a"
    down: str = "#ef4444"
    volume_up: str = "#16a34a99"
    volume_down: str = "#ef444499"


def _resolve_technical_dir(
    symbol: str,
    *,
    root: Path | None = None,
    run_timestamp: str | None = None,
) -> tuple[Path, str]:
    companies_root = get_companies_root(root or DEFAULT_COMPANIES_ROOT)
    technical_dir, timestamp = ensure_category_directory(
        symbol,
        "technical",
        root=companies_root,
        run_timestamp=run_timestamp,
    )
    return technical_dir, timestamp


def extract_candles(history_payload: Any) -> list[list[float]]:
    """Extract FYERS candle rows from a raw MCP/FYERS history payload.

    Accepts dicts, JSON strings, or objects with a ``data`` attribute.
    FYERS sometimes uses ``candles`` and the docs mention ``Candels``; both
    spellings are supported.
    """
    payload = history_payload
    if hasattr(payload, "data"):
        payload = getattr(payload, "data")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return []
    if not isinstance(payload, dict):
        return []

    candles = payload.get("candles")
    if candles is None:
        candles = payload.get("Candels")
    if not isinstance(candles, list):
        return []

    normalized: list[list[float]] = []
    for row in candles:
        if isinstance(row, (list, tuple)) and len(row) >= 6:
            normalized.append([row[0], row[1], row[2], row[3], row[4], row[5]])
    return normalized


def _drop_current_day_partial_candle(candles: Sequence[Sequence[float]]) -> list[list[float]]:
    """Drop the last candle if it belongs to the current UTC date.

    This avoids plotting an in-progress daily candle when the API returns the
    current trading day.
    """
    if len(candles) < 2:
        return [list(row) for row in candles]

    latest_ts = int(candles[-1][0])
    latest_dt = datetime.fromtimestamp(latest_ts, tz=timezone.utc).date()
    today_utc = datetime.now(timezone.utc).date()
    if latest_dt == today_utc:
        return [list(row) for row in candles[:-1]]
    return [list(row) for row in candles]


def _to_dataframe(candles: Sequence[Sequence[float]]) -> pd.DataFrame:
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    df = df.sort_values("datetime").reset_index(drop=True)
    df["x"] = range(len(df))
    return df


def _draw_candles(ax, df: pd.DataFrame, style: CandleChartStyle) -> None:
    width = 0.65

    for _, row in df.iterrows():
        x = row["x"]
        is_up = row["close"] >= row["open"]
        color = style.up if is_up else style.down
        lower = min(row["open"], row["close"])
        height = max(abs(row["close"] - row["open"]), 0.01)

        ax.vlines(x, row["low"], row["high"], color=color, linewidth=1.2, alpha=0.95, zorder=2)
        ax.add_patch(
            Rectangle(
                (x - width / 2, lower),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.95,
                zorder=3,
            )
        )


def _draw_volume(ax, df: pd.DataFrame, style: CandleChartStyle) -> None:
    colors = [style.volume_up if close >= open_ else style.volume_down for open_, close in zip(df["open"], df["close"])]
    ax.bar(df["x"], df["volume"], width=0.65, color=colors, align="center", zorder=1)


def generate_candlestick_png(
    candles: Sequence[Sequence[float]],
    symbol: str,
    output_path: str | Path,
    title: str | None = None,
    style: CandleChartStyle | None = None,
    drop_current_day_partial_candle: bool = True,
) -> Path:
    """Render OHLCV candles to a dark TradingView-style PNG."""
    if not candles:
        raise ValueError("No candle data provided")

    style = style or CandleChartStyle()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_candles = _drop_current_day_partial_candle(candles) if drop_current_day_partial_candle else [list(row) for row in candles]
    if not source_candles:
        raise ValueError("No completed candle data available after trimming partial candle")

    df = _to_dataframe(source_candles)
    plt.style.use("default")
    fig = plt.figure(figsize=(16, 9), dpi=170)
    fig.patch.set_facecolor(style.background)
    gs = fig.add_gridspec(5, 1, height_ratios=[4, 0.05, 1, 0.15, 0.05], hspace=0.02)

    ax_price = fig.add_subplot(gs[0, 0])
    ax_volume = fig.add_subplot(gs[2, 0], sharex=ax_price)

    for ax in (ax_price, ax_volume):
        ax.set_facecolor(style.background)
        ax.grid(True, color=style.grid, linestyle="--", linewidth=0.6, alpha=1.0)
        ax.tick_params(colors=style.text, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(style.grid)

    ax_price.yaxis.tick_right()
    ax_price.yaxis.set_label_position("right")
    ax_price.spines["left"].set_visible(False)
    ax_price.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax_price.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_price.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))

    ax_volume.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_volume.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_volume.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    _draw_candles(ax_price, df, style)

    ax_price.set_ylabel("Price", color=style.text)
    price_legend = [
        Patch(facecolor=style.up, edgecolor=style.up, label="Bullish candle"),
        Patch(facecolor=style.down, edgecolor=style.down, label="Bearish candle"),
    ]
    ax_price.legend(handles=price_legend, loc="upper left", frameon=False, fontsize=9)

    _draw_volume(ax_volume, df, style)
    ax_volume.set_ylabel("Vol", color=style.text)
    volume_legend = [
        Patch(facecolor=style.volume_up, edgecolor=style.up, label="Up volume"),
        Patch(facecolor=style.volume_down, edgecolor=style.down, label="Down volume"),
    ]
    ax_volume.legend(handles=volume_legend, loc="upper left", frameon=False, fontsize=9)

    step = max(len(df) // 14, 1)
    tick_positions = list(range(0, len(df), step))
    if tick_positions[-1] != len(df) - 1:
        tick_positions.append(len(df) - 1)
    tick_labels = [df.loc[pos, "datetime"].strftime("%b %d") for pos in tick_positions]
    ax_volume.set_xticks(tick_positions)
    ax_volume.set_xticklabels(tick_labels, rotation=0)
    plt.setp(ax_price.get_xticklabels(), visible=False)
    right_padding = 3
    ax_price.set_xlim(-1, len(df) - 1 + right_padding)
    ax_volume.set_xlim(-1, len(df) - 1 + right_padding)
    ax_price.tick_params(axis="y", which="major", pad=5)
    ax_price.tick_params(axis="y", which="minor", length=2)
    ax_volume.tick_params(axis="y", which="major", pad=5)
    ax_volume.tick_params(axis="y", which="minor", length=2)

    last = df.iloc[-1]
    title_text = title or f"{symbol} Candlestick Chart"
    subtitle = (
        f"{last['datetime'].strftime('%Y-%m-%d')}  "
        f"O {last['open']:.2f}  H {last['high']:.2f}  L {last['low']:.2f}  "
        f"C {last['close']:.2f}  V {int(last['volume'])}"
    )
    fig.suptitle(f"{title_text}\n{subtitle}", color=style.text, fontsize=16, fontweight="bold", y=0.98)

    fig.autofmt_xdate()
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def output_default_path(
    symbol: str,
    *,
    run_timestamp: str | None = None,
    root: Path | None = None,
) -> Path:
    technical_dir, _ = _resolve_technical_dir(symbol, root=root, run_timestamp=run_timestamp)
    return technical_dir / f"{normalise_symbol(symbol)}_candles.png"
