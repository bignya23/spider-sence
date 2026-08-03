import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import logging
from fyers_apiv3.FyersWebsocket import data_ws
from mcp_servers.market_mcp.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

def on_connect():
    logging.info("✅ WebSocket Connected! Subscribing to MCX:CRUDEOIL26AUGFUT...")
    ws.subscribe(symbols=["MCX:CRUDEOIL26AUGFUT"], data_type="SymbolUpdate")
    ws.keep_running()

def on_message(message):
    logging.info(f"📡 Tick Received: {message}")

def on_error(message):
    logging.error(f"❌ WebSocket Error: {message}")

def on_close(message):
    logging.info(f"🚪 WebSocket Closed: {message}")

cfg = load_config()
if not cfg.client_id or not cfg.access_token:
    raise RuntimeError("Missing FYERS credentials in .env")

ws = data_ws.FyersDataSocket(
    access_token=f"{cfg.client_id}:{cfg.access_token}",
    write_to_file=False,
    reconnect=True,
    on_connect=on_connect,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

if __name__ == "__main__":
    logging.info("Starting WebSocket test...")
    ws.connect()
