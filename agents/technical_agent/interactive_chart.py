from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from orchestrator.output_paths import ensure_category_directory, get_companies_root, normalise_symbol

DEFAULT_COMPANIES_ROOT = Path(__file__).resolve().parents[2] / "companies"


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


@dataclass(frozen=True)
class InteractiveChartStyle:
    background: str = "#ffffff"
    grid: str = "#e5e7eb"
    text: str = "#111827"
    up: str = "#16a34a"
    down: str = "#ef4444"
    volume_up: str = "#16a34a99"
    volume_down: str = "#ef444499"


def extract_candles(history_payload: Any) -> list[list[float]]:
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

    candles = payload.get("candles") or payload.get("Candels")
    if not isinstance(candles, list):
        return []

    normalized: list[list[float]] = []
    for row in candles:
        if isinstance(row, (list, tuple)) and len(row) >= 6:
            normalized.append([row[0], row[1], row[2], row[3], row[4], row[5]])
    return normalized


def _to_dataframe(candles: Sequence[Sequence[float]]) -> pd.DataFrame:
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    df = df.sort_values("datetime").reset_index(drop=True)
    df["x"] = range(len(df))
    df["date_label"] = df["datetime"].dt.strftime("%b %d")
    return df


def _drop_current_day_partial_candle(candles: Sequence[Sequence[float]]) -> list[list[float]]:
    if len(candles) < 2:
        return [list(row) for row in candles]

    latest_date = pd.to_datetime(int(candles[-1][0]), unit="s", utc=True).date()
    today_utc = pd.Timestamp.utcnow().date()
    if latest_date == today_utc:
        return [list(row) for row in candles[:-1]]
    return [list(row) for row in candles]


def generate_interactive_html(
    candles: Sequence[Sequence[float]],
    symbol: str,
    output_path: str | Path,
    title: str | None = None,
    style: InteractiveChartStyle | None = None,
    drop_current_day_partial_candle: bool = True,
) -> Path:
    if not candles:
        raise ValueError("No candle data provided")

    style = style or InteractiveChartStyle()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_candles = _drop_current_day_partial_candle(candles) if drop_current_day_partial_candle else [list(row) for row in candles]
    if not source_candles:
        raise ValueError("No completed candle data available after trimming partial candle")

    df = _to_dataframe(source_candles)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["x"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=style.up,
            increasing_fillcolor=style.up,
            decreasing_line_color=style.down,
            decreasing_fillcolor=style.down,
            name="Price",
            customdata=df[["datetime", "date_label"]],
            hovertemplate=(
                "Date: %{customdata[1]}<br>"
                "Open: %{open:.2f}<br>"
                "High: %{high:.2f}<br>"
                "Low: %{low:.2f}<br>"
                "Close: %{close:.2f}<br>"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    volume_colors = [
        "rgba(22, 163, 74, 0.45)" if close >= open_ else "rgba(239, 68, 68, 0.45)"
        for open_, close in zip(df["open"], df["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["x"],
            y=df["volume"],
            marker_color=volume_colors,
            name="Volume",
            hovertemplate="Date: %{customdata}<br>Volume: %{y:,}<extra></extra>",
            customdata=df["date_label"],
        ),
        row=2,
        col=1,
    )

    tick_step = max(len(df) // 12, 1)
    tick_positions = list(range(0, len(df), tick_step))
    if tick_positions[-1] != len(df) - 1:
        tick_positions.append(len(df) - 1)
    tick_labels = [df.loc[pos, "date_label"] for pos in tick_positions]

    last = df.iloc[-1]
    title_text = title or f"{symbol} Interactive Candlestick Chart"
    subtitle = (
        f"{last['datetime'].strftime('%Y-%m-%d')}  "
        f"O {last['open']:.2f}  H {last['high']:.2f}  L {last['low']:.2f}  "
        f"C {last['close']:.2f}  V {int(last['volume'])}"
    )

    fig.update_layout(
        title={"text": f"{title_text}<br><sup>{subtitle}</sup>", "x": 0.5, "xanchor": "center"},
        paper_bgcolor=style.background,
        plot_bgcolor=style.background,
        font={"color": style.text, "family": "Arial"},
        height=900,
        width=1600,
        margin={"l": 60, "r": 80, "t": 90, "b": 50},
        showlegend=False,
        dragmode="pan",
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        xaxis_fixedrange=False,
        yaxis_fixedrange=False,
    )

    fig.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=tick_positions,
        ticktext=tick_labels,
        showgrid=True,
        gridcolor=style.grid,
        zeroline=False,
        rangeslider_visible=True,
    )
    fig.update_yaxes(
        side="right",
        showgrid=True,
        gridcolor=style.grid,
        zeroline=False,
        tickformat=",.2f",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=style.grid,
        zeroline=False,
        tickformat=",",
        row=2,
        col=1,
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True, auto_open=False)
    return output_path


def output_default_html_path(
    symbol: str,
    *,
    run_timestamp: str | None = None,
    root: Path | None = None,
) -> Path:
    technical_dir, _ = _resolve_technical_dir(symbol, root=root, run_timestamp=run_timestamp)
    return technical_dir / f"{normalise_symbol(symbol)}_interactive.html"
