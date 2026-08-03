import asyncio
from pathlib import Path
import time
import json
import logging
from .live_scanner import LiveScanner
from .index_mapper import MAJOR_INDICES

# We reduce logging verbosity during EOD scan for cleaner output
logging.getLogger("fyers_api").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def cli():
    import argparse
    parser = argparse.ArgumentParser(description="Batch Run Technical Agent")
    parser.add_argument("--txt-file", type=str, default="nifty50.txt", help="Path to text file containing symbols")
    args = parser.parse_args()
    
    txt_path = Path(args.txt_file)
    if not txt_path.exists():
        print(f"Error: {args.txt_file} not found.")
        return
        
    symbols = txt_path.read_text().strip().split("\n")
    symbols = [s.strip() for s in symbols if s.strip()]
    
    for idx in MAJOR_INDICES:
        if idx not in symbols:
            symbols.append(idx)
    
    print(f"Starting EOD analysis for {len(symbols)} stocks using BatchIndicatorEngine...")
    print("="*60)
    
    # Initialize the scanner which holds the REST client and BatchEngine
    scanner = LiveScanner(symbols)
    
    start_time = time.time()
    
    # 1. Fetch History synchronously using the existing pre-fetch method
    scanner.pre_fetch_history()
    
    # 2. Run Engine (This triggers graph generation automatically)
    print("\nRunning Multi-Timeframe Screening & Correlation Engine...")
    results = scanner.engine.batch_screen()
    
    # 3. Flatten and Rank Setups
    flat_setups = []
    for r in results:
        if r["setups_found"] > 0:
            for style, style_data in r["setups"].items():
                s = style_data["setup"]
                confluence = r.get("confluence", 50)
                rr = s.get("risk_reward", 0.0)
                flat_setups.append({
                    "symbol": r["symbol"],
                    "bias": r.get("bias", "N/A"),
                    "regime": r.get("regime", "N/A"),
                    "style": style,
                    "confluence": confluence,
                    "rr": rr,
                    "setup": s
                })
                
    # Sort primarily by Confluence Score, then by Risk/Reward ratio
    flat_setups.sort(key=lambda x: (x["confluence"], x["rr"]), reverse=True)
    
    for rank, s in enumerate(flat_setups, 1):
        print(f"\n🏆 [RANK {rank}] {s['symbol']} | Score: {s['confluence']}/10.0 | RR: {s['rr']}x | Bias: {s['bias']} | Regime: {s['regime']}")
        print(f"   - {s['style'].upper()}: {s['setup'].get('bias', 'N/A')} @ {s['setup'].get('entry')} (Target: {s['setup'].get('take_profit')}, Stop: {s['setup'].get('stop_loss')})")
                
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    
    found_symbols = list(set([s["symbol"] for s in flat_setups]))
    print(f"✅ EOD Scan completed in {elapsed:.1f} seconds")
    print(f"✅ Total actionable setups found: {len(flat_setups)}")
    if found_symbols:
        print(f"📈 Stocks with setups: {', '.join(found_symbols)}")
    print("="*60)

if __name__ == "__main__":
    cli()
