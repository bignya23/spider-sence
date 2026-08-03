import asyncio
from pathlib import Path
import time
import json
from .technical_agent import TechnicalAgent

async def main(txt_file: str):
    txt_path = Path(txt_file)
    if not txt_path.exists():
        print(f"Error: {txt_file} not found.")
        return
        
    symbols = txt_path.read_text().strip().split("\n")
    symbols = [s.strip() for s in symbols if s.strip()]
    
    print(f"Starting batch analysis for {len(symbols)} stocks using TechnicalAgent...")
    print("="*60)
    
    # Initialize the agent once to reuse the MCP client
    agent = TechnicalAgent(generate_graphs=False)
    
    start_time = time.time()
    setups_found = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Analyzing {symbol}...")
        try:
            result = await agent.analyse(symbol)
            signals = result.get("signals", {})
            
            if signals:
                conf = signals.get("confluence", {})
                regime = signals.get("regime")
                score = conf.get("score")
                bias = conf.get("dominant_bias", "").upper()
                
                print(f"  Regime: {regime} | Bias: {bias} | Confluence: {score}/10")
                
                # Check style setups
                for style, data in signals.get("style_setups", {}).items():
                    if data.get("setup"):
                        setups_found += 1
                        print(f"  ✅ {style.upper()} SETUP FOUND")
                    else:
                        reason = str(data.get("rejection_reason"))[:80]
                        print(f"  ❌ {style.upper()}: {reason}")
            else:
                print("  ⚠️ No signal data returned.")
                
        except Exception as e:
            print(f"  ⚠️ Error analyzing {symbol}: {e}")
            
        # Give a small pause to be gentle on Fyers REST API rate limits
        await asyncio.sleep(0.5)
        
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"✅ Batch completed in {elapsed:.1f} seconds")
    print(f"✅ Total actionable setups found: {setups_found}")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch Run Technical Agent")
    parser.add_argument("--txt-file", type=str, default="nifty50.txt", help="Path to text file containing symbols")
    args = parser.parse_args()
    
    asyncio.run(main(args.txt_file))
