from __future__ import annotations

import logging
import os
from typing import Any, Dict

from agents.fundamental_agent import report_generator, screener_enricher
from agents.fundamental_agent import pipeline

try:
    from fastmcp import FastMCP
except Exception as exc:  # pragma: no cover - hard import
    raise ImportError("fastmcp is required for this MCP server; install fastmcp and try again") from exc

logger = logging.getLogger("fundamental_agent.mcp")


# Instantiate the FastMCP server
mcp = FastMCP("fundamental_agent")


@mcp.tool("generate_report")
def tool_generate_report(symbol: str) -> str:
    """Generate a fundamental report for `symbol` and return the report path."""
    return str(report_generator.run_report_generation(symbol))


@mcp.tool("compute_derivatives")
def tool_compute_derivatives(symbol: str) -> Dict[str, Any]:
    from pathlib import Path

    exports_dir = Path(__file__).resolve().parent / "exports"
    symbol = symbol.strip().upper()
    jsonl1 = exports_dir / symbol / f"{symbol.lower()}.jsonl"
    jsonl2 = exports_dir / symbol / f"{symbol}.jsonl"
    jpath = jsonl1 if jsonl1.exists() else (jsonl2 if jsonl2.exists() else None)
    if jpath is None:
        raise FileNotFoundError(f"JSONL for {symbol} not found in {exports_dir}")
    items = screener_enricher.load_jsonl(jpath)
    lookup = screener_enricher.build_lookup(items)
    return screener_enricher.compute_derivatives(lookup, symbol)


@mcp.tool("run_pipeline")
def tool_run_pipeline(symbol: str, generate_report: bool = False) -> str:
    """Run the full pipeline (fetch + enrich) for `symbol` and optionally generate report.

    Returns the path to the derived JSON file or the generated report path when `generate_report` is True.
    """
    outp = pipeline.run_pipeline(symbol, generate_report=generate_report)
    # pipeline.run_pipeline returns a Path to the derived JSON; return string
    return str(outp)


if __name__ == "__main__":
    # Run the FastMCP server; prefer HTTP transport so tools are reachable via HTTP.
    port = int(os.getenv("FUNDAMENTAL_AGENT_PORT", "8000"))
    path = os.getenv("FUNDAMENTAL_AGENT_PATH", "/mcp")
    host = os.getenv("FUNDAMENTAL_AGENT_HOST", "127.0.0.1")

    try:
        mcp.run(transport="http", host=host, port=port, path=path)
    except TypeError:
        # Older fastmcp versions may not accept host/path; try common fallback
        try:
            mcp.run(transport="http", port=port, path=path)
        except Exception:
            # Last resort: run with defaults (may use stdio)
            mcp.run()



