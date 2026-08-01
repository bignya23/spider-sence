#!/usr/bin/env python3
"""Example client for the Fundamental Agent MCP server.

Usage:
  python agents/fundamental_agent/client_example.py generate_report RELIANCE
  python agents/fundamental_agent/client_example.py compute_derivatives RELIANCE

Set `FUNDAMENTAL_AGENT_URL` to change the server URL (default http://localhost:8000).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import anyio
from fastmcp import Client


# Default FastMCP HTTP endpoint (streamable-http path)
DEFAULT_MCP_URL = (
    os.getenv("SCREENER_AGENT_URL")
    or os.getenv("FUNDAMENTAL_AGENT_URL")
    or "http://localhost:8000/mcp/screener"
)


async def invoke_tool_via_fastmcp(tool: str, args: Dict[str, Any], base_url: str = DEFAULT_MCP_URL) -> Any:
    """Invoke a tool using fastmcp.Client over HTTP transport.

    Example transport URL: "http://localhost:8000/mcp/screener"
    """
    async with Client(base_url) as client:
        # call_tool returns a CallToolResult (or raises ToolError)
        res = await client.call_tool(tool, args or {})
        # Try to return structured data if available
        try:
            # dataclass-like, prefer `data` or `content`
            if hasattr(res, "data") and res.data is not None:
                return res.data
            if hasattr(res, "content") and res.content is not None:
                return res.content
        except Exception:
            pass
        return res


def invoke_tool(tool: str, args: Dict[str, Any], base_url: str = DEFAULT_MCP_URL) -> Any:
    return anyio.run(invoke_tool_via_fastmcp, tool, args, base_url)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Fundamental Agent client example")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("generate_report")
    p1.add_argument("symbol")

    p2 = sub.add_parser("compute_derivatives")
    p2.add_argument("symbol")

    p4 = sub.add_parser("run_pipeline")
    p4.add_argument("symbol")
    p4.add_argument("--report", action="store_true", help="Generate LLM report after pipeline")

    p3 = sub.add_parser("invoke")
    p3.add_argument("tool")
    p3.add_argument("--args", default="{}", help="JSON string of args")

    args = parser.parse_args(argv)

    try:
        if args.cmd == "generate_report":
            out = invoke_tool("generate_report", {"symbol": args.symbol})
        elif args.cmd == "compute_derivatives":
            out = invoke_tool("compute_derivatives", {"symbol": args.symbol})
        elif args.cmd == "run_pipeline":
            out = invoke_tool(
                "run_pipeline",
                {"symbol": args.symbol, "generate_report": bool(args.report)},
            )
        else:
            # generic invoke
            try:
                parsed = json.loads(args.args)
            except Exception:
                print("Invalid JSON for --args", file=sys.stderr)
                return 2
            out = invoke_tool(args.tool, parsed)
    except Exception as exc:
        print("Invocation failed:", exc, file=sys.stderr)
        return 1

    # Print result
    try:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    except Exception:
        print(repr(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
