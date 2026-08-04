"""backtester.py

A walk-forward historical backtester to validate the Signal Engine.
Downloads historical 5m data, walks forward bar-by-bar, generates setups,
simulates fills (with slippage), tracks trades, and reports statistical metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from mcp_servers.market_mcp.config import load_config
from fyers_apiv3.fyersModel import FyersModel

from .indicator_engine import calculate_indicators
from .signal_engine import SignalConfig, generate_signals

logger = logging.getLogger("backtester")

class Trade:
    def __init__(self, symbol: str, style: str, direction: str, entry_price: float, stop_loss: float, take_profit: float, entry_time: pd.Timestamp):
        self.symbol = symbol
        self.style = style
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = entry_time
        
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[pd.Timestamp] = None
        self.exit_reason: str = ""
        self.status: str = "OPEN"
        
    @property
    def pnl_pct(self) -> float:
        if self.status == "OPEN" or self.exit_price is None:
            return 0.0
        if self.direction == "bullish":
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price

class BacktestEngine:
    def __init__(self, symbol: str, slippage_pct: float = 0.05, disable_tqdm: bool = False):
        self.symbol = symbol
        self.slippage_pct = slippage_pct
        self.disable_tqdm = disable_tqdm
        self.config = SignalConfig()
        
        self.trades: List[Trade] = []
        self.active_trades: List[Trade] = []
        
        self.df_5m: pd.DataFrame = pd.DataFrame()
        
    def fetch_data(self) -> None:
        """Fetch max 5m data from Fyers (up to 100 days)."""
        logger.info(f"Downloading 60 days of 5m data for {self.symbol} via Fyers...")
        
        cfg = load_config()
        if not cfg.client_id or not cfg.access_token:
            raise RuntimeError("Missing FYERS_CLIENT_ID or FYERS_ACCESS_TOKEN")
            
        client = FyersModel(
            client_id=cfg.client_id,
            token=cfg.access_token,
            is_async=False,
            log_path="",
        )
        
        now = int(time.time())
        past = now - (60 * 24 * 60 * 60) # 60 days
        
        res = client.history({
            "symbol": self.symbol,
            "resolution": "5",
            "date_format": "0",
            "range_from": str(past),
            "range_to": str(now),
            "cont_flag": "1",
        })
        
        if res.get("s") == "ok" and "candles" in res:
            candles = res["candles"]
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            if df["timestamp"].max() > 2e9:
                df["timestamp"] = df["timestamp"] // 1000
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
            self.df_5m = df
            logger.info(f"Data fetched: {len(self.df_5m)} candles.")
        else:
            raise ValueError(f"Failed to fetch {self.symbol} via Fyers: {res}")

    def _resample(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        df_idx = df.set_index("datetime")
        agg_dict = {"timestamp": "last", "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        
        m15_df = df_idx.resample("15min").agg(agg_dict).dropna().reset_index()
        h1_df = df_idx.resample("1h").agg(agg_dict).dropna().reset_index()
        d1_df = df_idx.resample("1D").agg(agg_dict).dropna().reset_index()
        
        return {"m5": df, "m15": m15_df, "1h": h1_df, "1d": d1_df}

    def run(self, warmup_bars: int = 500) -> None:
        """Walk forward through the data."""
        if len(self.df_5m) <= warmup_bars:
            logger.error("Not enough data to run backtest.")
            return
            
        logger.info("Starting Walk-Forward Simulation...")
        
        for i in tqdm(range(warmup_bars, len(self.df_5m)), desc=f"Backtesting {self.symbol}", disable=self.disable_tqdm):
            current_bar = self.df_5m.iloc[i]
            current_time = current_bar["datetime"]
            
            # 1. Manage Active Trades
            for trade in self.active_trades[:]:
                high = current_bar["high"]
                low = current_bar["low"]
                
                if trade.direction == "bullish":
                    if low <= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = current_time
                        trade.exit_reason = "SL Hit"
                        trade.status = "CLOSED"
                        self.active_trades.remove(trade)
                    elif high >= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.exit_time = current_time
                        trade.exit_reason = "TP Hit"
                        trade.status = "CLOSED"
                        self.active_trades.remove(trade)
                else:
                    if high >= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = current_time
                        trade.exit_reason = "SL Hit"
                        trade.status = "CLOSED"
                        self.active_trades.remove(trade)
                    elif low <= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.exit_time = current_time
                        trade.exit_reason = "TP Hit"
                        trade.status = "CLOSED"
                        self.active_trades.remove(trade)

            # 2. Slice data up to the PREVIOUS bar to simulate what the engine sees
            slice_df = self.df_5m.iloc[:i]
            resampled = self._resample(slice_df)
            
            sliced_inds = {
                "daily": calculate_indicators(resampled["1d"]),
                "hourly": calculate_indicators(resampled["1h"]),
                "m15": calculate_indicators(resampled["m15"]),
                "m5": calculate_indicators(resampled["m5"])
            }
            
            # Generate signals based on sliced indicators
            result = generate_signals(sliced_inds, {}, config=self.config)
            
            # Check setups
            for style, style_data in result.get("style_setups", {}).items():
                s = style_data.get("setup")
                if not s:
                    continue
                    
                direction = s["bias"]
                target = s["take_profit"]
                stop = s["stop_loss"]
                
                # Check if we already have an active trade for this style
                if any(t.style == style for t in self.active_trades):
                    continue
                    
                # Simulate Entry on current bar's OPEN with slippage
                open_price = current_bar["open"]
                
                # If we gap above target or below stop, it's a missed trade
                if direction == "bullish" and open_price >= target:
                    continue
                if direction == "bearish" and open_price <= target:
                    continue
                    
                slip_amt = open_price * (self.slippage_pct / 100.0)
                sim_entry = open_price + slip_amt if direction == "bullish" else open_price - slip_amt
                
                trade = Trade(
                    symbol=self.symbol,
                    style=style,
                    direction=direction,
                    entry_price=sim_entry,
                    stop_loss=stop,
                    take_profit=target,
                    entry_time=current_time
                )
                self.trades.append(trade)
                self.active_trades.append(trade)
                
        # Close any remaining active trades at market end
        last_bar = self.df_5m.iloc[-1]
        for trade in self.active_trades:
            trade.exit_price = last_bar["close"]
            trade.exit_time = last_bar["datetime"]
            trade.exit_reason = "EOD Force Close"
            trade.status = "CLOSED"
            
    def report(self) -> None:
        print("\n" + "="*60)
        print(f"📊 BACKTEST RESULTS: {self.symbol}")
        print(f"Period: 60 Days | Slippage penalty: {self.slippage_pct}%")
        print("="*60)
        
        if not self.trades:
            print("No trades generated during the period.")
            return
            
        wins = [t for t in self.trades if t.pnl_pct > 0]
        losses = [t for t in self.trades if t.pnl_pct <= 0]
        
        win_rate = (len(wins) / len(self.trades)) * 100
        gross_profit = sum([t.pnl_pct for t in wins])
        gross_loss = abs(sum([t.pnl_pct for t in losses]))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        net_return = sum([t.pnl_pct for t in self.trades]) * 100
        
        avg_win = (gross_profit / len(wins) * 100) if wins else 0.0
        avg_loss = (gross_loss / len(losses) * 100) if losses else 0.0
        
        cumulative_pnl = 0.0
        peak = 0.0
        max_drawdown = 0.0
        sorted_trades = sorted(self.trades, key=lambda t: t.exit_time or pd.Timestamp.min)
        for t in sorted_trades:
            cumulative_pnl += (t.pnl_pct * 100)
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        print(f"Total Trades:  {len(self.trades)}")
        print(f"Win Rate:      {win_rate:.1f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Avg Win:       +{avg_win:.2f}%")
        print(f"Avg Loss:      -{avg_loss:.2f}%")
        print(f"Max Drawdown:  -{max_drawdown:.2f}%")
        print(f"Net Return:    {net_return:.2f}% (Cumulative PnL %)")
        print("\nTrades Breakdown by Style:")
        
        styles = set([t.style for t in self.trades])
        for style in styles:
            s_trades = [t for t in self.trades if t.style == style]
            s_wins = [t for t in s_trades if t.pnl_pct > 0]
            s_wr = (len(s_wins) / len(s_trades)) * 100
            s_pnl = sum([t.pnl_pct for t in s_trades]) * 100
            print(f"  - {style.upper()}: {len(s_trades):<3} trades | Win Rate: {s_wr:>5.1f}% | Net PnL: {s_pnl:>6.2f}%")
        print("="*60 + "\n")

def cli():
    import argparse
    parser = argparse.ArgumentParser(description="Run the Backtester")
    parser.add_argument("symbol", type=str, help="Symbol to backtest (e.g. NSE:DRREDDY-EQ)")
    parser.add_argument("--slippage", type=float, default=0.05, help="Slippage percentage per execution")
    args = parser.parse_args()
    
    engine = BacktestEngine(symbol=args.symbol, slippage_pct=args.slippage)
    try:
        engine.fetch_data()
        engine.run()
        engine.report()
    except Exception as e:
        logger.error(f"Backtest failed: {e}")

if __name__ == "__main__":
    cli()

