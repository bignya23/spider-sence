from __future__ import annotations

import os
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
import uvicorn

from mcp_servers.market_mcp.server import mcp as market_mcp
from mcp_servers.screener_mcp.server import mcp as screener_mcp
from mcp_servers.news_mcp.server import mcp as news_mcp

market_app = market_mcp.http_app(path="/")
screener_app = screener_mcp.http_app(path="/")
news_app = news_mcp.http_app(path="/")


@asynccontextmanager
async def lifespan(app: FastAPI):
	async with AsyncExitStack() as stack:
		await stack.enter_async_context(market_app.lifespan(app))
		await stack.enter_async_context(screener_app.lifespan(app))
		await stack.enter_async_context(news_app.lifespan(app))
		yield


app = FastAPI(title="spider-sence", lifespan=lifespan)
app.mount("/mcp/market", market_app)
app.mount("/mcp/screener", screener_app)
app.mount("/mcp/news", news_app)


def main() -> None:
	host = os.getenv("SPIDER_SENCE_HOST", "127.0.0.1")
	port_env = os.getenv("SPIDER_SENCE_PORT", "8000")
	try:
		port = int(port_env)
	except ValueError:
		port = 8000

	uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
	main()
