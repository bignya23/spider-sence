from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from agents.fundamental_agent import pipeline, report_generator, screener_enricher
from orchestrator.output_paths import get_category_directory, get_companies_root, normalise_symbol

COMPANIES_ROOT = Path(__file__).resolve().parents[2] / "companies"

try:
    from fastmcp import FastMCP
except Exception as exc:  # pragma: no cover - hard import
    raise ImportError("fastmcp is required for this MCP server; install fastmcp and try again") from exc


logger = logging.getLogger("screener_mcp")

mcp = FastMCP("screener")


@mcp.tool("generate_report")
def tool_generate_report(symbol: str, run_timestamp: str | None = None) -> str:
    """Generate a fundamental report for `symbol` and return the report path."""
    root = get_companies_root(COMPANIES_ROOT)
    return str(
        report_generator.run_report_generation(
            symbol,
            exports_dir=root,
            run_timestamp=run_timestamp,
        )
    )


@mcp.tool("compute_derivatives")
def tool_compute_derivatives(symbol: str, run_timestamp: str | None = None) -> Dict[str, Any]:
    symbol = symbol.strip().upper()
    root = get_companies_root(COMPANIES_ROOT)
    try:
        fundamental_dir, _ = get_category_directory(
            symbol,
            "fundamental",
            root=root,
            run_timestamp=run_timestamp,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc))

    safe_symbol = normalise_symbol(symbol)
    jsonl1 = fundamental_dir / f"{safe_symbol.lower()}.jsonl"
    jsonl2 = fundamental_dir / f"{safe_symbol}.jsonl"
    jpath = jsonl1 if jsonl1.exists() else (jsonl2 if jsonl2.exists() else None)
    if jpath is None:
        raise FileNotFoundError(f"JSONL for {symbol} not found in {fundamental_dir}")
    items = screener_enricher.load_jsonl(jpath)
    lookup = screener_enricher.build_lookup(items)
    return screener_enricher.compute_derivatives(lookup, symbol)


@mcp.tool("run_pipeline")
def tool_run_pipeline(symbol: str, generate_report: bool = False, run_timestamp: str | None = None) -> Dict[str, Any]:
    """Run the full pipeline (fetch + enrich) and optionally generate a report."""
    root = get_companies_root(COMPANIES_ROOT)
    return pipeline.run_pipeline(
        symbol,
        exports_dir=root,
        generate_report=generate_report,
        run_timestamp=run_timestamp,
    )


def main() -> None:
    port_env = os.getenv("SCREENER_AGENT_PORT") or os.getenv("FUNDAMENTAL_AGENT_PORT")
    path = os.getenv("SCREENER_AGENT_PATH") or os.getenv("FUNDAMENTAL_AGENT_PATH") or "/mcp/screener"
    host = os.getenv("SCREENER_AGENT_HOST") or os.getenv("FUNDAMENTAL_AGENT_HOST") or "127.0.0.1"

    try:
        port = int(port_env) if port_env else 8000
    except ValueError:
        port = 8000

    try:
        mcp.run(transport="http", host=host, port=port, path=path)
    except TypeError:
        try:
            mcp.run(transport="http", port=port, path=path)
        except Exception:
            mcp.run()


if __name__ == "__main__":
    main()
