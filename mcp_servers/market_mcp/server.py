from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastmcp import FastMCP
from fyers_apiv3 import fyersModel

from .config import load_config

logger = logging.getLogger("market_mcp")

mcp = FastMCP("market")


class FyersClient:
    def __init__(self):
        cfg = load_config()
        if not cfg.client_id or not cfg.access_token:
            raise RuntimeError("Missing FYERS_CLIENT_ID or FYERS_ACCESS_TOKEN")
        self.client = fyersModel.FyersModel(
            client_id=cfg.client_id,
            token=cfg.access_token,
            is_async=False,
            log_path="",
        )

    def quote(self, symbol: str):
        return self.client.quotes({"symbols": symbol})

    @staticmethod
    def _normalize_range(value: str) -> str:
        """FYERS history expects epoch seconds; accept ISO dates for convenience."""
        text = str(value).strip()
        if text.isdigit():
            return text
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
                return str(int(dt.timestamp()))
            except ValueError:
                continue
        return text

    def history(self, symbol: str, range_from: str, range_to: str, resolution: str):
        return self.client.history({
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "0",
            "range_from": self._normalize_range(range_from),
            "range_to": self._normalize_range(range_to),
            "cont_flag": "1",
        })

    def depth(self, symbol: str):
        return self.client.depth({"symbol": symbol})

    def profile(self):
        return self.client.get_profile()

    def funds(self):
        return self.client.funds()

    def positions(self):
        return self.client.positions()


class MarketService:
    def __init__(self):
        self.client = FyersClient()

    @staticmethod
    def _check(raw):
        if isinstance(raw, dict) and raw.get("s") == "error":
            raise RuntimeError(raw.get("message", "Broker error"))
        return raw

    def get_quote(self, symbol: str):
        raw = self._check(self.client.quote(symbol))
        if "d" not in raw:
            return raw
        row = raw["d"][0]
        v = row.get("v", {})
        return {
            "symbol": row.get("n"),
            "exchange": row.get("e"),
            "ltp": v.get("lp"),
            "open": v.get("open_price"),
            "high": v.get("high_price"),
            "low": v.get("low_price"),
            "prev_close": v.get("prev_close_price"),
            "volume": v.get("volume"),
            "raw": raw,
        }

    def get_history(self, symbol, range_from, range_to, resolution):
        return self._check(
            self.client.history(symbol, range_from, range_to, resolution)
        )

    def get_depth(self, symbol):
        return self._check(self.client.depth(symbol))

    def get_profile(self):
        return self._check(self.client.profile())

    def get_funds(self):
        return self._check(self.client.funds())

    def get_positions(self):
        return self._check(self.client.positions())


service = MarketService()


@mcp.tool
def get_quote(symbol: str) -> dict[str, Any]:
    return service.get_quote(symbol)


@mcp.tool
def get_history(
    symbol: str,
    range_from: str,
    range_to: str,
    resolution: str = "D",
) -> dict[str, Any]:
    return service.get_history(symbol, range_from, range_to, resolution)


@mcp.tool
def get_depth(symbol: str) -> dict[str, Any]:
    return service.get_depth(symbol)


@mcp.tool
def get_profile() -> dict[str, Any]:
    return service.get_profile()


@mcp.tool
def get_funds() -> dict[str, Any]:
    return service.get_funds()


@mcp.tool
def get_positions() -> dict[str, Any]:
    return service.get_positions()


@mcp.tool
def status():
    return {"status": "ok", "service": "market"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8000,
        path="/mcp/market",
    )