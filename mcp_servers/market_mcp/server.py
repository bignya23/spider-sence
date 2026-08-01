"""Market MCP server for FYERS API.

Tools: get_quote, get_history, get_depth, get_profile, get_funds, get_positions,
       get_option_chain, get_market_status.
"""

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
        return self.client.depth({"symbol": symbol, "ohlcv_flag": "1"})

    def profile(self):
        return self.client.get_profile()

    def funds(self):
        return self.client.funds()

    def positions(self):
        return self.client.positions()

    def option_chain(self, symbol: str, strike_count: int = 10, timestamp: str = ""):
        payload: dict[str, Any] = {"symbol": symbol, "strikecount": strike_count}
        if timestamp:
            payload["timestamp"] = timestamp
        return self.client.optionchain(payload)

    def market_status(self):
        return self.client.market_status()


class MarketService:
    def __init__(self):
        self.client = FyersClient()

    @staticmethod
    def _check(raw):
        if isinstance(raw, dict) and raw.get("s") == "error":
            raise RuntimeError(raw.get("message", "Broker error"))
        return raw

    def get_quote(self, symbol: str) -> dict[str, Any]:
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
            "52w_high": v.get("52w_high"),
            "52w_low": v.get("52w_low"),
            "avg_price": v.get("avg_price"),
            "raw": raw,
        }

    def get_history(self, symbol, range_from, range_to, resolution):
        return self._check(
            self.client.history(symbol, range_from, range_to, resolution)
        )

    def get_depth(self, symbol: str) -> dict[str, Any]:
        """Level 2 market depth with bid/ask pressure score."""
        raw = self._check(self.client.depth({"symbol": symbol, "ohlcv_flag": "1"}))
        if "d" not in raw:
            return raw
        d = raw["d"].get(symbol, raw["d"])
        if not isinstance(d, dict):
            return raw
        bids = d.get("bids", []) or []
        asks = d.get("ask", []) or []
        total_bid_qty = sum(b.get("quantity", 0) for b in bids)
        total_ask_qty = sum(a.get("quantity", 0) for a in asks)
        total = total_bid_qty + total_ask_qty
        pressure_score = round(total_bid_qty / total, 3) if total else 0.5
        return {
            "symbol": symbol,
            "bids": bids[:5],
            "asks": asks[:5],
            "total_bid_qty": total_bid_qty,
            "total_ask_qty": total_ask_qty,
            "pressure_score": pressure_score,   # >0.5 = buyer-dominated
            "bias": "buy" if pressure_score > 0.55 else "sell" if pressure_score < 0.45 else "neutral",
            "ohlcv": {
                "open": d.get("o"),
                "high": d.get("h"),
                "low": d.get("l"),
                "close": d.get("c"),
                "volume": d.get("v"),
            },
        }

    def get_profile(self):
        return self._check(self.client.profile())

    def get_funds(self):
        return self._check(self.client.funds())

    def get_positions(self):
        return self._check(self.client.positions())

    def get_option_chain(self, symbol: str, strike_count: int = 10, timestamp: str = "") -> dict[str, Any]:
        """Fetch option chain and compute PCR, Max Pain, and OI summary."""
        raw = self._check(self.client.option_chain(symbol, strike_count, timestamp))
        if "data" not in raw:
            return raw

        data = raw["data"]
        strikes = data.get("optionsChain", [])

        total_ce_oi = 0.0
        total_pe_oi = 0.0
        pain_by_strike: dict[float, float] = {}

        for row in strikes:
            strike = float(row.get("strikePrice", 0))
            ce_oi = float(row.get("ce", {}).get("openInterest", 0) or 0)
            pe_oi = float(row.get("pe", {}).get("openInterest", 0) or 0)
            total_ce_oi += ce_oi
            total_pe_oi += pe_oi
            pain_by_strike[strike] = ce_oi + pe_oi

        pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi else None
        max_pain_strike = min(pain_by_strike, key=pain_by_strike.get) if pain_by_strike else None

        top_ce = sorted(strikes, key=lambda r: float(r.get("ce", {}).get("openInterest", 0) or 0), reverse=True)[:3]
        top_pe = sorted(strikes, key=lambda r: float(r.get("pe", {}).get("openInterest", 0) or 0), reverse=True)[:3]

        return {
            "symbol": symbol,
            "expiry": data.get("expiryData", [{}])[0].get("date") if data.get("expiryData") else None,
            "pcr": pcr,
            "pcr_bias": "bullish" if pcr and pcr > 1.2 else "bearish" if pcr and pcr < 0.7 else "neutral",
            "max_pain": max_pain_strike,
            "total_ce_oi": total_ce_oi,
            "total_pe_oi": total_pe_oi,
            "ce_resistance_strikes": [r.get("strikePrice") for r in top_ce],
            "pe_support_strikes": [r.get("strikePrice") for r in top_pe],
            "raw_chain": strikes,
        }

    def get_market_status(self) -> dict[str, Any]:
        raw = self._check(self.client.market_status())
        if "marketStatus" not in raw:
            return raw
        segments = raw["marketStatus"]
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "segments": [
                {
                    "exchange": s.get("exchange"),
                    "segment": s.get("segment"),
                    "status": s.get("status"),
                    "market_type": s.get("market_type"),
                }
                for s in (segments if isinstance(segments, list) else [])
            ],
        }


service = MarketService()


@mcp.tool
def get_quote(symbol: str) -> dict[str, Any]:
    """Get real-time quote for a symbol. Includes 52W high/low and avg price."""
    return service.get_quote(symbol)


@mcp.tool
def get_history(
    symbol: str,
    range_from: str,
    range_to: str,
    resolution: str = "D",
) -> dict[str, Any]:
    """Get OHLCV history. resolution: D=daily, 60=hourly, 15=15min, 5=5min, 1=1min."""
    return service.get_history(symbol, range_from, range_to, resolution)


@mcp.tool
def get_depth(symbol: str) -> dict[str, Any]:
    """Get Level 2 market depth with bid/ask pressure score and directional bias."""
    return service.get_depth(symbol)


@mcp.tool
def get_profile() -> dict[str, Any]:
    """Get Fyers account profile."""
    return service.get_profile()


@mcp.tool
def get_funds() -> dict[str, Any]:
    """Get available funds and margins."""
    return service.get_funds()


@mcp.tool
def get_positions() -> dict[str, Any]:
    """Get current open positions."""
    return service.get_positions()


@mcp.tool
def get_option_chain(
    symbol: str,
    strike_count: int = 10,
    timestamp: str = "",
) -> dict[str, Any]:
    """Fetch option chain with PCR, Max Pain, and top OI strikes (support/resistance).

    Args:
        symbol: Underlying symbol, e.g. 'NSE:SBIN-EQ'
        strike_count: Number of strikes on each side of ATM (default 10)
        timestamp: Expiry timestamp string (empty = nearest expiry)
    """
    return service.get_option_chain(symbol, strike_count, timestamp)


@mcp.tool
def get_market_status() -> dict[str, Any]:
    """Get current open/close status for all exchange segments (NSE, BSE, MCX etc.)."""
    return service.get_market_status()


@mcp.tool
def status():
    """Health check."""
    return {"status": "ok", "service": "market"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8000,
        path="/mcp/market",
    )