import time
import json
import pandas as pd
from fyers_apiv3.fyersModel import FyersModel
from mcp_servers.market_mcp.config import load_config
from agents.technical_agent.batch_engine import BatchIndicatorEngine

if __name__ == "__main__":
    cfg = load_config()
    client = FyersModel(client_id=cfg.client_id, token=cfg.access_token, is_async=False, log_path="")

    symbol = "NSE:BAJFINANCE-EQ"
    print(f"Fetching live 15m data for {symbol}...")

    now = int(time.time())
    past = now - (30 * 24 * 60 * 60)

    res = client.history({
        "symbol": symbol,
        "resolution": "15",
        "date_format": "0",
        "range_from": str(past),
        "range_to": str(now),
        "cont_flag": "1",
    })

    if res.get("s") == "ok":
        candles = res["candles"]
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df["timestamp"].max() > 2e9:
            df["timestamp"] = df["timestamp"] // 1000
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        
        engine = BatchIndicatorEngine(max_candles=300)
        engine.initialize_symbol(symbol, df)
        
        print("Running batch_screen()...")
        results = engine.batch_screen([symbol])
        
        if results:
            res = results[0]
            print("\n" + "="*60)
            print(f"✅ {symbol} BATCH ENGINE TEST")
            print(f"  Regime: {res['regime']} | ADX: {res['adx']:.1f} | Confluence: {res['confluence']}/10")
            print(f"  Summary: {res['summary'][:150]}...")
            if res['setups_found'] > 0:
                print("\n  🚨 SETUPS FOUND:")
                print(json.dumps(res['setups'], indent=2))
            else:
                print("\n  ❌ No actionable setups found.")
            print("="*60)
    else:
        print("Error fetching data:", res)
