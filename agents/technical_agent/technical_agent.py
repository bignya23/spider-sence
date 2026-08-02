from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
from fastmcp import Client

import pandas as pd
from fastmcp.exceptions import ToolError

from .indicator_engine import IndicatorConfig, calculate_indicators
from .pattern_engine import PatternConfig, detect_patterns
from .indicator_candlestick import generate_indicator_graphs
from .signal_engine import SignalConfig, generate_signals
from orchestrator.output_paths import (
    ensure_category_directory,
    format_run_timestamp,
    get_companies_root,
    normalise_symbol,
)


class MarketClient:
    def __init__(self, url: str = "http://127.0.0.1:8000/mcp/market"):
        self.client = Client(url)

    async def call(self, tool: str, **kwargs) -> Any:
        async with self.client:
            result = await self.client.call_tool(tool, arguments=kwargs)
            return result.data

    async def list_tools(self):
        async with self.client:
            return await self.client.list_tools()


def _history_to_dataframe(payload: Mapping[str, Any]) -> pd.DataFrame:
    candles = payload.get("candles") or payload.get("Candels")
    if not isinstance(candles, Iterable):
        raise ValueError("history payload missing 'candles' list")

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    if df.empty:
        return df

    df = df.astype(
        {
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        }
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _quote_direction(quote: Mapping[str, Any]) -> str:
    ltp = quote.get("ltp")
    prev = quote.get("prev_close") or quote.get("prev_close_price")
    if ltp is None or prev is None:
        return "neutral"
    try:
        change = float(ltp) - float(prev)
        pct = change / float(prev)
    except (TypeError, ValueError, ZeroDivisionError):
        return "neutral"
    if pct > 0.003:
        return "bullish"
    if pct < -0.003:
        return "bearish"
    return "neutral"


@dataclass(frozen=True)
class TimeframeSpec:
    resolution: str
    windows: tuple[int, ...]


DEFAULT_TIMEFRAMES: Dict[str, TimeframeSpec] = {
    "daily":  TimeframeSpec("D",  (365, 180, 90, 45)),
    "hourly": TimeframeSpec("60", (90, 60, 30, 15)),
    "m15":    TimeframeSpec("15", (30, 15, 7)),
    "m5":     TimeframeSpec("5",  (15, 7, 3)),
}


class TechnicalAgent:
    """Multi-timeframe technical analysis orchestrator (no LLM math)."""

    def __init__(
        self,
        *,
        market_url: str = "http://127.0.0.1:8000/mcp/market",
        timeframes: Dict[str, TimeframeSpec] | None = None,
        indicator_config: IndicatorConfig | None = None,
        pattern_config: PatternConfig | None = None,
        signal_config: SignalConfig | None = None,
        generate_graphs: bool = True,
        output_root: Path | None = None,
    ) -> None:
        self.market = MarketClient(market_url)
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES
        self.indicator_config = indicator_config or IndicatorConfig()
        self.pattern_config = pattern_config or PatternConfig()
        self.signal_config = signal_config or SignalConfig()
        self.generate_graphs = generate_graphs
        self.output_root = output_root
        self.graphs_root = output_root

    async def analyse(
        self,
        symbol: str,
        *,
        index_symbol: str = "NSE:NIFTY50-INDEX",
        sector_symbol: str | None = None,
        run_timestamp: str | None = None,
    ) -> Dict[str, Any]:
        symbol = symbol.strip().upper()
        run_ts = run_timestamp or format_run_timestamp()

        history_payloads = await self._fetch_histories(symbol)
        frames = {
            name: _history_to_dataframe(payload)
            for name, payload in history_payloads.items()
        }

        indicators = {
            name: calculate_indicators(df, self.indicator_config)
            for name, df in frames.items()
            if not df.empty
        }

        primary_frame = frames.get("daily")

        if primary_frame is None or primary_frame.empty:
            primary_frame = next(
                (df for df in frames.values() if not df.empty),
                pd.DataFrame(),
            )

        patterns = (
            detect_patterns(primary_frame, self.pattern_config)
            if not primary_frame.empty
            else {}
        )

        context = await self._build_context(
            index_symbol=index_symbol,
            sector_symbol=sector_symbol,
        )

        result: Dict[str, Any] = {
            "symbol": symbol,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_timestamp": run_ts,
            "timeframes": indicators,
            "patterns": patterns,
            "context": context,
        }

        # ── signal synthesis ──────────────────────────────────────────────
        if indicators:
            result["signals"] = generate_signals(
                indicators, patterns, self.signal_config
            )

        # ── save JSON to disk ────────────────────────────────────────────────
        json_path = self._save_json(result, symbol, run_ts)
        result["output_json"] = str(json_path)

        # ── generate indicator charts automatically ──────────────────────────
        if self.generate_graphs:
            graph_paths = await self._generate_graphs(
                symbol,
                history_payloads=history_payloads,
                signals=result.get("signals", {}),
                run_ts=run_ts,
            )
            result["graphs"] = [str(p) for p in graph_paths]

        return result

    def _save_json(self, data: Dict[str, Any], symbol: str, run_ts: str) -> Path:
        """Persist the analysis result as a JSON file in the technical directory.

        Saved to::

            companies/<SYMBOL>/<run_ts>/technical/<SYMBOL>_technical.json
        """
        companies_root = get_companies_root(self.output_root)
        technical_dir, _ = ensure_category_directory(
            symbol,
            "technical",
            root=companies_root,
            run_timestamp=run_ts,
        )
        safe_symbol = normalise_symbol(symbol)
        output_path = technical_dir / f"{safe_symbol}_technical.json"
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    async def _generate_graphs(
        self,
        symbol: str,
        *,
        history_payloads: Dict[str, Dict[str, Any]],
        signals: Dict[str, Any],
        run_ts: str,
    ) -> list[Path]:
        """Generate per-indicator charts using the already-fetched daily history.

        Falls back to re-fetching a 1-year daily window if the daily payload
        is missing or empty.
        """
        daily_payload = history_payloads.get("daily")

        if not daily_payload:
            # Re-fetch a 1-year daily window as fallback
            now = datetime.now(timezone.utc)
            try:
                daily_payload = await self.market.call(
                    "get_history",
                    symbol=symbol,
                    resolution="D",
                    range_from=(now.replace(year=now.year - 1)).strftime("%Y-%m-%d"),
                    range_to=now.strftime("%Y-%m-%d"),
                )
            except Exception:
                return []

        try:
            return generate_indicator_graphs(
                daily_payload,
                symbol,
                run_timestamp=run_ts,
                root=self.graphs_root,
                config=self.indicator_config,
                setups=signals.get("style_setups", {}),
            )
        except Exception:
            return []

    async def _fetch_histories(
        self,
        symbol: str,
    ) -> Dict[str, Dict[str, Any]]:
        now = datetime.now(timezone.utc)

        async def fetch_with_windows(
            name: str,
            spec: TimeframeSpec,
        ) -> Dict[str, Any]:
            last_error: Exception | None = None

            for days in spec.windows:
                start = now - timedelta(days=days)

                try:
                    return await self.market.call(
                        "get_history",
                        symbol=symbol,
                        resolution=spec.resolution,
                        range_from=start.strftime("%Y-%m-%d"),
                        range_to=now.strftime("%Y-%m-%d"),
                    )

                except (ToolError, RuntimeError, ValueError) as exc:
                    last_error = exc
                    continue

            raise RuntimeError(
                f"Failed to fetch history for timeframe '{name}' "
                f"({spec.resolution}) after trying windows "
                f"{spec.windows}: {last_error}"
            )

        tasks = [
            (
                name,
                asyncio.create_task(fetch_with_windows(name, spec)),
            )
            for name, spec in self.timeframes.items()
        ]

        results: Dict[str, Dict[str, Any]] = {}

        for name, task in tasks:
            results[name] = await task

        return results

    async def _build_context(
        self,
        *,
        index_symbol: str,
        sector_symbol: str | None,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        index_quote = await self.market.call(
            "get_quote",
            symbol=index_symbol,
        )

        summary["index"] = {
            "symbol": index_symbol,
            "direction": _quote_direction(index_quote),
            "ltp": index_quote.get("ltp"),
            "prev_close": index_quote.get("prev_close")
            or index_quote.get("prev_close_price"),
        }

        if sector_symbol:
            sector_quote = await self.market.call(
                "get_quote",
                symbol=sector_symbol,
            )

            summary["sector"] = {
                "symbol": sector_symbol,
                "direction": _quote_direction(sector_quote),
                "ltp": sector_quote.get("ltp"),
                "prev_close": sector_quote.get("prev_close")
                or sector_quote.get("prev_close_price"),
            }

        return summary


async def run_basic_analysis(
    symbol: str,
    *,
    market_url: str = "http://127.0.0.1:8000/mcp/market",
    index_symbol: str = "NSE:NIFTY50-INDEX",
    sector_symbol: str | None = None,
    generate_graphs: bool = True,
) -> Dict[str, Any]:
    agent = TechnicalAgent(market_url=market_url, generate_graphs=generate_graphs)

    return await agent.analyse(
        symbol,
        index_symbol=index_symbol,
        sector_symbol=sector_symbol,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Technical agent CLI")
    parser.add_argument("symbol", help="Trading symbol (e.g. NSE:SBIN-EQ)")
    parser.add_argument("--config", help="Path to technical config JSON", default=None)
    parser.add_argument(
        "--market-url",
        default="http://127.0.0.1:8000/mcp/market",
        help="MCP market endpoint base URL",
    )
    parser.add_argument("--index-symbol", default="NSE:NIFTY50-INDEX")
    parser.add_argument("--sector-symbol", default=None)
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="Skip indicator graph generation (graphs are generated by default)",
    )
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help="Reuse a specific run timestamp folder for output",
    )

    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    indicator_cfg = None
    pattern_cfg = None
    if "indicator_config" in cfg:
        indicator_cfg = IndicatorConfig(**cfg["indicator_config"])
    if "pattern_config" in cfg:
        pattern_cfg = PatternConfig(**cfg["pattern_config"])

    market_url = cfg.get("market_url") or args.market_url
    index_symbol = cfg.get("index_symbol") or args.index_symbol
    sector_symbol = cfg.get("sector_symbol") if cfg.get("sector_symbol") is not None else args.sector_symbol

    async def _main() -> None:
        agent = TechnicalAgent(
            market_url=market_url,
            indicator_config=indicator_cfg,
            pattern_config=pattern_cfg,
            generate_graphs=not args.no_graphs,
        )

        result = await agent.analyse(
            args.symbol,
            index_symbol=index_symbol,
            sector_symbol=sector_symbol,
            run_timestamp=args.run_timestamp,
        )

        # Print JSON result (excluding internal path keys for cleaner output)
        printable = {k: v for k, v in result.items() if k not in ("graphs", "output_json")}
        print(json.dumps(printable, indent=2, ensure_ascii=False))

        # ── signal summary ────────────────────────────────────────────────────
        signals = result.get("signals", {})
        if signals:
            conf = signals.get("confluence", {})
            style_setups = signals.get("style_setups", {})
            print("\n" + "═" * 68)
            print(f"  📈 TECHNICAL ANALYSIS  ({args.symbol})")
            print("═" * 68)
            print(f"  {signals.get('summary', '')}")
            print(f"\n  Confluence: {conf.get('score')}/10  │  Dominant Bias: {conf.get('dominant_bias', '').upper()}  │  {conf.get('interpretation', '')}")

            if style_setups:
                print("\n  ── TRADING STYLE SETUPS ──")
                for style_name, style_data in style_setups.items():
                    setup = style_data.get("setup")
                    bias = style_data.get("bias", "N/A").upper()
                    if setup:
                        direction = "LONG ▲" if style_data["bias"] == "bullish" else "SHORT ▼"
                        print(f"\n  ✅ [{style_name.upper()}]  Bias: {bias}  |  {direction}")
                        print(f"     Entry: {setup['entry']}  ({setup['entry_level']})")
                        print(f"     Stop:  {setup['stop_loss']}  ({setup['stop_level']})")
                        print(f"     Target:{setup['take_profit']}  ({setup['target_level']})")
                        print(f"     RR: {setup['risk_reward']}×  │  Risk/share: ₹{setup['per_share_risk']}  │  Reward/share: ₹{setup['per_share_reward']}")
                    else:
                        reason = style_data.get("rejection_reason", "Filtered")
                        print(f"  ✗  [{style_name.upper()}]  Bias: {bias}  → {reason}")
            print("\n" + "═" * 68)

        # ── saved file locations ──────────────────────────────────────────────
        from pathlib import Path as _Path
        json_out = result.get("output_json")
        if json_out:
            print(f"\n💾 JSON saved → {json_out}")

        graph_paths = result.get("graphs", [])
        if graph_paths:
            graphs_dir = _Path(graph_paths[0]).parent
            print(f"📊 Graphs saved → {graphs_dir}")
            for p in graph_paths:
                print(f"   {_Path(p).name}")
            print(f"   ({len(graph_paths)} files written)")

    asyncio.run(_main())