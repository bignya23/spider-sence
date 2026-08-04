"""batch_backtester.py

A wrapper around the backtester to run a batch of stocks concurrently,
aggregating the trades to provide a portfolio-level statistical report.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from agents.technical_agent.backtester import BacktestEngine, Trade

logger = logging.getLogger("batch_backtester")

def run_single_backtest(symbol: str, slippage: float) -> Tuple[str, List[Trade], str]:
    """Runs a single backtest and returns the trades or an error message."""
    engine = BacktestEngine(symbol=symbol, slippage_pct=slippage, disable_tqdm=True)
    try:
        engine.fetch_data()
        engine.run()
        return symbol, engine.trades, ""
    except Exception as e:
        return symbol, [], str(e)

def cli():
    parser = argparse.ArgumentParser(description="Run Batch Backtester")
    parser.add_argument("--slippage", type=float, default=0.05, help="Slippage percentage per execution")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()
    
    # 15 highly liquid NSE names
    symbols = [
        "NSE:RELIANCE-EQ",
        "NSE:HDFCBANK-EQ",
        "NSE:ICICIBANK-EQ",
        "NSE:INFY-EQ",
        "NSE:TCS-EQ",
        "NSE:ITC-EQ",
        "NSE:LT-EQ",
        "NSE:SBIN-EQ",
        "NSE:BHARTIARTL-EQ",
        "NSE:KOTAKBANK-EQ",
        "NSE:AXISBANK-EQ",
        "NSE:ASIANPAINT-EQ",
        "NSE:BAJFINANCE-EQ",
        "NSE:MARUTI-EQ",
        "NSE:M&M-EQ"
    ]
    
    print("="*60)
    print(f"🚀 Starting Batch Backtest for {len(symbols)} symbols...")
    print(f"Workers: {args.workers} | Slippage: {args.slippage}%")
    print("="*60)
    
    all_trades: List[Trade] = []
    errors: List[Tuple[str, str]] = []
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_backtest, sym, args.slippage): sym for sym in symbols}
        
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Batch Progress"):
            sym, trades, err = future.result()
            if err:
                errors.append((sym, err))
            else:
                all_trades.extend(trades)
                
    # ── AGGREGATE REPORTING ──
    print("\n" + "="*60)
    print("📊 PORTFOLIO BATCH BACKTEST RESULTS")
    print("="*60)
    
    if errors:
        print(f"⚠️ Errors encountered on {len(errors)} symbols:")
        for sym, err in errors:
            print(f"  - {sym}: {err}")
        print("-" * 60)
        
    if not all_trades:
        print("No trades generated across any symbols.")
        return
        
    wins = [t for t in all_trades if t.pnl_pct > 0]
    losses = [t for t in all_trades if t.pnl_pct <= 0]
    
    win_rate = (len(wins) / len(all_trades)) * 100
    gross_profit = sum([t.pnl_pct for t in wins])
    gross_loss = abs(sum([t.pnl_pct for t in losses]))
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    net_return = sum([t.pnl_pct for t in all_trades]) * 100
    
    avg_win = (gross_profit / len(wins) * 100) if wins else 0.0
    avg_loss = (gross_loss / len(losses) * 100) if losses else 0.0
    
    cumulative_pnl = 0.0
    peak = 0.0
    max_drawdown = 0.0
    sorted_trades = sorted(all_trades, key=lambda t: t.exit_time or pd.Timestamp.min)
    for t in sorted_trades:
        cumulative_pnl += (t.pnl_pct * 100)
        if cumulative_pnl > peak:
            peak = cumulative_pnl
        drawdown = peak - cumulative_pnl
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            
    print(f"Total Portfolio Trades: {len(all_trades)}")
    print(f"Portfolio Win Rate:     {win_rate:.1f}%")
    print(f"Portfolio Profit Factor:{profit_factor:.2f}")
    print(f"Average Win:            +{avg_win:.2f}%")
    print(f"Average Loss:           -{avg_loss:.2f}%")
    print(f"Max Portfolio Drawdown: -{max_drawdown:.2f}%")
    print(f"Total Net Return:       {net_return:.2f}% (Cumulative PnL %)")
    
    print("\nTrades Breakdown by Style:")
    styles = set([t.style for t in all_trades])
    for style in styles:
        s_trades = [t for t in all_trades if t.style == style]
        s_wins = [t for t in s_trades if t.pnl_pct > 0]
        s_wr = (len(s_wins) / len(s_trades)) * 100
        s_pnl = sum([t.pnl_pct for t in s_trades]) * 100
        print(f"  - {style.upper()}: {len(s_trades):<4} trades | Win Rate: {s_wr:>5.1f}% | Net PnL: {s_pnl:>6.2f}%")
        
    print("\nTrades Breakdown by Symbol (Top 5 & Bottom 5):")
    symbol_pnls = {}
    for t in all_trades:
        if t.symbol not in symbol_pnls:
            symbol_pnls[t.symbol] = 0.0
        symbol_pnls[t.symbol] += (t.pnl_pct * 100)
        
    sorted_symbols = sorted(symbol_pnls.items(), key=lambda x: x[1], reverse=True)
    print("  Top Performers:")
    for sym, pnl in sorted_symbols[:5]:
        print(f"    - {sym}: +{pnl:.2f}%")
    print("  Bottom Performers:")
    for sym, pnl in sorted_symbols[-5:]:
        print(f"    - {sym}: {pnl:.2f}%")
        
    print("="*60 + "\n")
    
    # ── SAVE TO JSON ──
    output_data = {
        "summary": {
            "total_trades": len(all_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor if profit_factor != float("inf") else -1,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "max_drawdown_pct": max_drawdown,
            "net_return_pct": net_return
        },
        "errors": errors,
        "symbol_performance": sorted_symbols,
        "trades": []
    }
    
    for t in sorted_trades:
        output_data["trades"].append({
            "symbol": t.symbol,
            "style": t.style,
            "direction": t.direction,
            "entry_time": str(t.entry_time),
            "entry_price": t.entry_price,
            "exit_time": str(t.exit_time) if t.exit_time else None,
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "pnl_pct": t.pnl_pct * 100
        })
        
    os.makedirs("backtest_reports", exist_ok=True)
    filename = f"backtest_reports/batch_run_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"💾 Full report saved to: {filename}")

if __name__ == "__main__":
    cli()
