"""Run the fundamental agent MCP server locally.

Usage:
  python -m agents.fundamental_agent.run_mcp_server
"""
from __future__ import annotations

import os
import sys


def main() -> None:
  # Import the mcp server module and call its run() entrypoint.
  # This keeps the run behavior consistent with FastMCP's expectations.
  try:
    from mcp_servers.screener_mcp import server as mcp_server
  except Exception as exc:
    print("Failed to import screener MCP server:", exc, file=sys.stderr)
    raise

  # If FastMCP exposes a run method, call it. Otherwise, import side-effects should start the server.
  mcp = getattr(mcp_server, "mcp", None)
  if mcp is None:
    raise RuntimeError("mcp object not exposed by mcp_servers.screener_mcp.server")

  # Allow optional env-driven port if FastMCP supports it via run kwargs
  port_env = os.getenv("SCREENER_AGENT_PORT") or os.getenv("FUNDAMENTAL_AGENT_PORT")
  path = os.getenv("SCREENER_AGENT_PATH") or os.getenv("FUNDAMENTAL_AGENT_PATH") or "/mcp/screener"
  host = os.getenv("SCREENER_AGENT_HOST") or os.getenv("FUNDAMENTAL_AGENT_HOST") or "127.0.0.1"

  try:
    port_num = int(port_env) if port_env else 8000
  except ValueError:
    port_num = 8000

  # Start FastMCP with HTTP transport so clients can reach it at /mcp
  try:
    mcp.run(transport="http", host=host, port=port_num, path=path)
    return
  except TypeError:
    # Older fastmcp might use different parameter names; try common fallback
    try:
      mcp.run(transport="http", port=port_num, path=path)
      return
    except Exception:
      # fallback to plain run() if HTTP args are unsupported
      pass

  mcp.run()


if __name__ == "__main__":
  main()
