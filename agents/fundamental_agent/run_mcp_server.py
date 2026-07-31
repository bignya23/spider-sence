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
    from agents.fundamental_agent import mcp_server
  except Exception as exc:
    print("Failed to import mcp_server:", exc, file=sys.stderr)
    raise

  # If FastMCP exposes a run method, call it. Otherwise, import side-effects should start the server.
  mcp = getattr(mcp_server, "mcp", None)
  if mcp is None:
    raise RuntimeError("mcp object not found in agents.fundamental_agent.mcp_server")

  # Allow optional env-driven port if FastMCP supports it via run kwargs
  port = os.getenv("FUNDAMENTAL_AGENT_PORT")
  path = os.getenv("FUNDAMENTAL_AGENT_PATH", "/mcp")
  host = os.getenv("FUNDAMENTAL_AGENT_HOST", "127.0.0.1")

  try:
    port_num = int(port) if port else 8000
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
