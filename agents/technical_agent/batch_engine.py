"""batch_engine.py

Vectorized/batched processing engine for the live scanner.

Maintains historical OHLCV data in memory for hundreds of stocks.
When a new candle closes, it appends the new row and recalculates
indicators and signals efficiently.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import pandas as pd

from .indicator_engine import calculate_indicators
from .signal_engine import SignalConfig, generate_signals

logger = logging.getLogger("batch_engine")

class BatchIndicatorEngine:
    def __init__(self, max_candles: int = 500):
        self.max_candles = max_candles
        # symbol -> pd.DataFrame
        self.histories: Dict[str, pd.DataFrame] = {}
        self.signal_config = SignalConfig()

    def initialize_symbol(self, symbol: str, df: pd.DataFrame) -> None:
        """Seed the engine with historical data (e.g. from REST API)."""
        if df.empty:
            return
        
        # Ensure we don't hold more than max_candles
        if len(df) > self.max_candles:
            df = df.iloc[-self.max_candles:]
            
        self.histories[symbol] = df.copy()
        logger.info(f"Initialized {symbol} with {len(df)} candles.")

    def update_candle(self, symbol: str, candle: Dict[str, Any]) -> None:
        """Append a new closed candle from the live websocket."""
        if symbol not in self.histories:
            # Create a new df if it doesn't exist
            self.histories[symbol] = pd.DataFrame([candle])
            return

        df = self.histories[symbol]
        
        # candle dict must match DataFrame columns: datetime, timestamp, open, high, low, close, volume
        new_row_df = pd.DataFrame([candle])
        
        # Append and prune
        df = pd.concat([df, new_row_df], ignore_index=True)
        if len(df) > self.max_candles:
            df = df.iloc[1:].reset_index(drop=True)
            
        self.histories[symbol] = df

    def get_history(self, symbol: str) -> pd.DataFrame | None:
        return self.histories.get(symbol)
        
    def _resample_ohlcv(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Resample the base 5-minute dataframe to 15m, 1h, and 1d."""
        if df.empty or "datetime" not in df.columns:
            return {"m5": df, "m15": pd.DataFrame(), "1h": pd.DataFrame(), "1d": pd.DataFrame()}
        
        # We must use the datetime column as index for resampling
        df_idx = df.set_index("datetime")
        agg_dict = {
            "timestamp": "last",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        
        m15_df = df_idx.resample("15min").agg(agg_dict).dropna().reset_index()
        h1_df = df_idx.resample("1h").agg(agg_dict).dropna().reset_index()
        d1_df = df_idx.resample("1D").agg(agg_dict).dropna().reset_index()
        
        return {
            "m5": df,
            "m15": m15_df,
            "1h": h1_df,
            "1d": d1_df
        }

    def _process_single(self, symbol: str, df: pd.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any] | None:
        """Process a single symbol. Designed to be run in a separate process."""
        try:
            if df.empty or len(df) < 55:  # Need at least 55 for Donchian 55
                return None
                
            resampled = self._resample_ohlcv(df)
            
            ind_m5 = calculate_indicators(resampled["m5"])
            ind_m15 = calculate_indicators(resampled["m15"])
            ind_hourly = calculate_indicators(resampled["1h"])
            ind_daily = calculate_indicators(resampled["1d"])
            
            timeframes = {
                "daily": ind_daily,
                "hourly": ind_hourly,
                "m15": ind_m15,
                "m5": ind_m5
            }
            
            result = generate_signals(timeframes, {}, config=self.signal_config, context=context)
            
            # Filter to see if any actionable setups were found
            valid_setups = {k: v for k, v in result["style_setups"].items() if v.get("setup")}
            
            # Generate Interactive Graph for the detected timeframe
            if valid_setups:
                try:
                    from datetime import datetime
                    from pathlib import Path
                    from .indicator_candlestick import generate_indicator_graphs
                    from .trading_styles import TRADING_STYLES
                    
                    safe_sym = symbol.replace(":", "_")
                    today_str = datetime.now().strftime("%Y%m%d")
                    
                    for style, style_data in valid_setups.items():
                        entry_tf = TRADING_STYLES[style].entry_tf
                        entry_df = resampled[entry_tf]
                        if not entry_df.empty:
                            candles_list = entry_df[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
                            
                            generate_indicator_graphs(
                                history_payload={"candles": candles_list},
                                symbol=symbol,
                                run_timestamp=today_str,
                                root=Path("companies"),
                                indicator_names=[],
                                setups={style: style_data}
                            )
                            
                            import json
                            setup_dir = Path("companies") / safe_sym / today_str / "technical" / "setups" / style
                            setup_dir.mkdir(parents=True, exist_ok=True)
                            
                            json_data = {
                                "symbol": symbol,
                                "style": style,
                                "regime": result.get("regime"),
                                "bias": result.get("confluence", {}).get("dominant_bias"),
                                "summary": result.get("summary"),
                                "details": style_data
                            }
                            
                            json_file = setup_dir / f"{safe_sym}_{style}_setup.json"
                            with open(json_file, "w") as f:
                                json.dump(json_data, f, indent=4)

                except Exception as e:
                    logger.error(f"Failed to generate graph for {symbol}: {e}", exc_info=True)
            
            
            return {
                "symbol": symbol,
                "regime": result["regime"],
                "bias": result["confluence"]["dominant_bias"],
                "adx": result["adx"],
                "confluence": result["confluence"]["score"],
                "setups_found": len(valid_setups),
                "setups": valid_setups,
                "summary": result["summary"]
            }
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            return None

    def batch_screen(self, symbols: List[str] | None = None) -> List[Dict[str, Any]]:
        """Run the 4-step filter across all specified symbols concurrently."""
        if symbols is None:
            symbols = list(self.histories.keys())
            
        results = []
        global_context = {}
        
        # 1. Process indices first to establish global context
        from .index_mapper import MAJOR_INDICES, get_parent_index
        for idx_sym in MAJOR_INDICES:
            if idx_sym in symbols and idx_sym in self.histories:
                df = self.histories[idx_sym]
                idx_res = self._process_single(idx_sym, df, context=None)
                if idx_res:
                    global_context[idx_sym] = {
                        "regime": idx_res["regime"],
                        "bias": idx_res["bias"]
                    }
                    results.append(idx_res)
                    
        # 2. Process stocks concurrently with index context
        
        # Use ThreadPoolExecutor because ProcessPoolExecutor cannot be spawned from daemon threads (WebSocket threads)
        with ThreadPoolExecutor() as executor:
            futures = []
            for sym in symbols:
                if sym in MAJOR_INDICES:
                    continue # Already processed
                    
                df = self.histories.get(sym)
                if df is not None:
                    parent_idx = get_parent_index(sym)
                    stock_context = global_context.get(parent_idx, None)
                    futures.append(executor.submit(self._process_single, sym, df, stock_context))
            
            for future in futures:
                res = future.result()
                if res is not None:
                    results.append(res)
                    
        return results

if __name__ == "__main__":
    import argparse
    import time
    import numpy as np

    parser = argparse.ArgumentParser(description="Batch Engine Benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    parser.add_argument("--n-stocks", type=int, default=200, help="Number of stocks to simulate")
    args = parser.parse_args()

    if args.benchmark:
        print(f"Generating synthetic data for {args.n_stocks} stocks...")
        np.random.seed(42)
        n_candles = 300
        dates = pd.date_range("2025-01-01 09:15", periods=n_candles, freq="15min")
        
        engine = BatchIndicatorEngine(max_candles=n_candles)
        
        for i in range(args.n_stocks):
            volatility = np.random.random() * 5
            drift = np.random.randn() * 0.5 
            closes = np.cumsum(np.random.randn(n_candles) * volatility + drift) + 1000
            vol_base = np.random.randint(100000, 500000)
            
            df = pd.DataFrame({
                "datetime": dates,
                "timestamp": (dates.astype("int64") // 1e9).astype(int),
                "open":   closes * 0.999,
                "high":   closes * 1.005,
                "low":    closes * 0.995,
                "close":  closes,
                "volume": np.abs(np.random.randn(n_candles) * vol_base * 0.5) + vol_base,
            })
            engine.initialize_symbol(f"STOCK_{i+1:03d}", df)

        print(f"Running Phase 1 analysis (Indicators + 4-Step Filter) across all {args.n_stocks} stocks...")
        start_time = time.perf_counter()
        
        results = engine.batch_screen()
        
        elapsed = time.perf_counter() - start_time
        
        regime_counts = {"trending": 0, "transition": 0, "ranging": 0}
        setups_found = 0
        for r in results:
            regime_counts[r["regime"]] += 1
            setups_found += r["setups_found"]

        print("=" * 50)
        print(f"✅ Processed {args.n_stocks} stocks in {elapsed:.3f} seconds")
        print(f"   Average per stock: {(elapsed/args.n_stocks)*1000:.1f} ms")
        print(f"   Regimes detected: {regime_counts}")
        print(f"   Total valid setups generated: {setups_found}")
        print("=" * 50)

