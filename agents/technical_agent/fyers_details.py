from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import Client

from .chart_generator import extract_candles, generate_candlestick_png, output_default_path
from .interactive_chart import generate_interactive_html, output_default_html_path


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


def save_history_chart(
    history: dict[str, Any],
    symbol: str,
    *,
    run_timestamp: str | None = None,
    root: Path | None = None,
) -> Path:
    candles = extract_candles(history)
    output_path = output_default_path(symbol, run_timestamp=run_timestamp, root=root)
    return generate_candlestick_png(candles, symbol=symbol, output_path=output_path)


def save_interactive_history_chart(
    history: dict[str, Any],
    symbol: str,
    *,
    run_timestamp: str | None = None,
    root: Path | None = None,
) -> Path:
    candles = extract_candles(history)
    output_path = output_default_html_path(symbol, run_timestamp=run_timestamp, root=root)
    return generate_interactive_html(candles, symbol=symbol, output_path=output_path)


# -------------------------------
# Example Usage
# -------------------------------

if __name__ == "__main__":
    import asyncio

    async def main():
        market = MarketClient()

        print("\nAvailable Tools:")
        tools = await market.list_tools()

        for tool in tools:
            print(f"- {tool.name}")

        print("\nStatus:")
        print(await market.call("status"))

        print("\nQuote:")
        print(
            await market.call(
                "get_quote",
                symbol="NSE:SBIN-EQ",
            )
        )

        print(
            await market.call(
                "get_profile",
            )
        )

        print("\nHistory:")
        history = await market.call(
            "get_history",
            symbol="NSE:SBIN-EQ",
            range_from="2026-01-01",
            range_to="2026-07-31",
            resolution="D",
        )
        print(history)
        chart_path = save_history_chart(history, "NSE:SBIN-EQ")
        run_timestamp = chart_path.parent.parent.name
        run_root = chart_path.parent.parent
        print(f"\nChart saved to: {chart_path}")
        interactive_chart_path = save_interactive_history_chart(
            history,
            "NSE:SBIN-EQ",
            run_timestamp=run_timestamp,
        )
        print(f"Interactive chart saved to: {interactive_chart_path}")
        print(f"Run folder: {run_root}")
        print(f"Run timestamp: {run_timestamp}")

    asyncio.run(main())


    
