from fastapi import FastAPI
from mcp_servers.market_mcp.server import market_mcp
from mcp_servers.news_mcp.server import news_mcp
from mcp_servers.execution_mcp.server import execution_mcp

app = FastAPI()

app.mount("/mcp/market", market_mcp.streamable_http_app())
app.mount("/mcp/news", news_mcp.streamable_http_app())
app.mount("/mcp/execution", execution_mcp.streamable_http_app())