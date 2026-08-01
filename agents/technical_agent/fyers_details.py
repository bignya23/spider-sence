from __future__ import annotations

from typing import Any

from fastmcp import Client


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

        print("\nHistory:")
        print(
            await market.call(
                "get_history",
                symbol="NSE:SBIN-EQ",
                range_from="2025-01-01",
                range_to="2025-07-31",
                resolution="D",
            )
        )

    asyncio.run(main())


    