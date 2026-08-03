"""live_scanner.py

Phase 3: WebSocket Live Scanner

Connects to Fyers Data WebSocket, maintains live tick data, builds
15-minute candles, and pushes them to the BatchIndicatorEngine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from fyers_apiv3.FyersWebsocket import data_ws
from fyers_apiv3.fyersModel import FyersModel
from mcp_servers.market_mcp.config import load_config

from .batch_engine import BatchIndicatorEngine
from .signal_dispatcher import SignalDispatcher
from .pre_gate import UniversePreGate

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("live_scanner")

class LiveScanner:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.engine = BatchIndicatorEngine()
        self.dispatcher = SignalDispatcher()
        self.pre_gate = UniversePreGate()
        
        self.cfg = load_config()
        if not self.cfg.client_id or not self.cfg.access_token:
            raise RuntimeError("Missing FYERS_CLIENT_ID or FYERS_ACCESS_TOKEN")
            
        # REST client for initial history fetch
        self.rest_client = FyersModel(
            client_id=self.cfg.client_id,
            token=self.cfg.access_token,
            is_async=False,
            log_path="",
        )
        
        # Websocket client
        self.ws = data_ws.FyersDataSocket(
            access_token=f"{self.cfg.client_id}:{self.cfg.access_token}",
            write_to_file=False,
            reconnect=True,
            on_connect=self.on_connect,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        
        # Aggregation state
        self.current_candles: Dict[str, Dict[str, Any]] = {}
        self.last_candle_ts: int = 0
        
    def _normalize_symbol(self, sym: str) -> str:
        # Fyers requires NSE:SYMBOL-EQ format
        return sym
        
    def pre_fetch_history(self) -> None:
        """Fetch past 7 days of 15-minute history to seed the engine."""
        logger.info(f"Pre-fetching history for {len(self.symbols)} symbols...")
        now = int(time.time())
        past = now - (7 * 24 * 60 * 60) # 7 days
        
        for i, sym in enumerate(self.symbols, 1):
            logger.info(f"[{i}/{len(self.symbols)}] Fetching history for {sym}")
            try:
                res = self.rest_client.history({
                    "symbol": sym,
                    "resolution": "5",
                    "date_format": "0",
                    "range_from": str(past),
                    "range_to": str(now),
                    "cont_flag": "1",
                })
                
                if res.get("s") == "ok" and "candles" in res:
                    import pandas as pd
                    candles = res["candles"]
                    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    # Fyers timestamp is usually in seconds or milliseconds, typically seconds for resolution data
                    if df["timestamp"].max() > 2e9:
                        df["timestamp"] = df["timestamp"] // 1000
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
                    self.engine.initialize_symbol(sym, df)
                else:
                    logger.warning(f"Failed to fetch {sym}: {res}")
            except Exception as e:
                logger.error(f"Error fetching {sym}: {e}")
                
            time.sleep(0.5) # respect rate limit

    def on_connect(self) -> None:
        logger.info("✅ WebSocket connected! Subscribing to symbols...")
        self.ws.subscribe(symbols=self.symbols, data_type="SymbolUpdate")
        # Keep full modes for OHLCV
        self.ws.keep_running()

    def on_message(self, message: dict) -> None:
        """Handle incoming tick data and build 15m candles."""
        logger.debug(f"Raw ws message: {message}")
        if not isinstance(message, dict) or "symbol" not in message:
            return
            
        sym = message["symbol"]
        ltp = message.get("ltp")
        vol_traded = message.get("vol_traded_today", 0)
        
        if not ltp:
            return
            
        logger.debug(f"Tick received for {sym}: {ltp} (Vol: {vol_traded})")

        now = int(time.time())
        # Determine current 5m boundary (e.g. 09:15, 09:20)
        boundary = now - (now % 300)
        
        # If boundary changed, flush old candles to engine and scan
        if boundary > self.last_candle_ts:
            if self.current_candles:
                logger.info(f"⏳ 5-minute boundary crossed ({boundary}). Flushing candles and running scan...")
                
                # IMPORTANT: Update state BEFORE flushing, so if flushing crashes, it doesn't infinite loop
                self.last_candle_ts = boundary
                old_candles = self.current_candles
                self.current_candles = {}
                
                self.flush_candles(old_candles)
                
                # Check for 09:30 AM Pre-Gate
                time_str = datetime.fromtimestamp(boundary).strftime("%H:%M")
                if time_str == "09:30":
                    self.execute_pre_gate()
                    
            else:
                self.last_candle_ts = boundary
                self.current_candles = {}

        # Update current candle
        if sym not in self.current_candles:
            self.current_candles[sym] = {
                "timestamp": boundary,
                "datetime": pd.to_datetime(boundary, unit="s"),
                "open": ltp,
                "high": ltp,
                "low": ltp,
                "close": ltp,
                "volume": 0, # We'll need a way to track interval volume if possible
                "_last_total_vol": vol_traded
            }
        else:
            c = self.current_candles[sym]
            c["high"] = max(c["high"], ltp)
            c["low"] = min(c["low"], ltp)
            c["close"] = ltp
            
            if vol_traded > c["_last_total_vol"]:
                c["volume"] += (vol_traded - c["_last_total_vol"])
                c["_last_total_vol"] = vol_traded

    def flush_candles(self, candles: Dict[str, Dict[str, Any]]) -> None:
        """Push completed 5m candles to BatchEngine and run scan."""
        for sym, candle in candles.items():
            self.engine.update_candle(sym, candle)
            
        logger.info("Running batched indicator screen...")
        results = self.engine.batch_screen(list(candles.keys()))
        
        for r in results:
            if r["setups_found"] > 0:
                self.dispatcher.dispatch(r)

    def execute_pre_gate(self) -> None:
        """Runs the 09:30 AM Universe Pre-Gate and unsubscribes from rejected symbols."""
        logger.info("🕒 09:30 AM Reached. Triggering Universe Pre-Gate...")
        
        in_play = self.pre_gate.filter_universe(self.engine.histories)
        
        rejected = set(self.symbols) - set(in_play)
        
        if rejected:
            logger.info(f"Unsubscribing from {len(rejected)} rejected symbols to save bandwidth.")
            # Unsubscribe requires a list of symbols
            try:
                self.ws.unsubscribe(symbols=list(rejected))
            except Exception as e:
                logger.error(f"Failed to unsubscribe symbols: {e}")
                
        self.symbols = in_play

    def on_error(self, message: dict) -> None:
        logger.error(f"WebSocket error: {message}")

    def on_close(self, message: dict) -> None:
        logger.info(f"WebSocket closed: {message}")

    def run(self) -> None:
        self.pre_fetch_history()
        logger.info("Starting WebSocket connection...")
        self.ws.connect()
        
        # Block the main thread so the Python interpreter does not shut down.
        # If the main thread exits, any background daemon threads will crash
        # when trying to schedule new tasks.
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down Live Scanner...")
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Spider-Sence Live Scanner — Fyers WebSocket"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols, e.g. NSE:SBIN-EQ,NSE:RELIANCE-EQ",
    )
    parser.add_argument(
        "--txt-file",
        type=str,
        help="Path to a text file with one symbol per line (e.g. nifty50.txt)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging level for verbose output",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG mode enabled")

    # Resolve symbol list: --symbols takes priority over --txt-file
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.txt_file:
        try:
            with open(args.txt_file, "r") as f:
                symbols = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Error: {args.txt_file} not found.")
            exit(1)
    else:
        logger.error("Error: You must provide either --symbols or --txt-file")
        exit(1)

    logger.info(f"Symbols loaded ({len(symbols)}): {symbols}")
    scanner = LiveScanner(symbols)
    scanner.run()
