import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("signal_dispatcher")

class SignalDispatcher:
    def __init__(self, base_dir: str = "companies"):
        self.base_dir = Path(base_dir)
        
    def dispatch(self, signal_data: Dict[str, Any]) -> None:
        """
        Routes the final signal to console and JSONL log files.
        """
        if signal_data.get("setups_found", 0) == 0:
            return
            
        sym = signal_data["symbol"]
        
        # 1. Rich Console Formatting
        print("\n" + "="*60)
        print(f"🚨 [LIVE SIGNAL FOUND] {sym}")
        print(f"   Regime: {signal_data['regime'].upper()} | ADX: {signal_data['adx']:.1f} | Confluence: {signal_data['confluence']}/10")
        print("   Setups:")
        for style, setup in signal_data["setups"].items():
            print(f"   - {style.upper()}: {setup['direction']} @ {setup['entry']} (Target: {setup['target']}, Stop: {setup['stop']})")
        print("="*60 + "\n")
        
        # 2. Append-only JSONL Logging
        # Map NSE:RELIANCE-EQ to NSE_RELIANCE-EQ
        safe_sym = sym.replace(":", "_")
        sym_dir = self.base_dir / safe_sym
        sym_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = sym_dir / "live_signals.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(signal_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write signal to {log_file}: {e}")
