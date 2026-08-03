import json
import logging
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

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
        
        # 2. Day-Wise Folder and Logging
        # Map NSE:RELIANCE-EQ to NSE_RELIANCE-EQ
        safe_sym = sym.replace(":", "_")
        today_str = datetime.now().strftime("%Y%m%d")
        sym_dir = self.base_dir / safe_sym / today_str
        sym_dir.mkdir(parents=True, exist_ok=True)
        
        # Continuous Append Log (so everything is present)
        log_file = sym_dir / "day_signals.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(signal_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write signal to {log_file}: {e}")
            
        # Overwrite Latest State (so dashboard can fetch it easily)
        latest_file = sym_dir / "latest_signal.json"
        try:
            with open(latest_file, "w") as f:
                json.dump(signal_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to overwrite {latest_file}: {e}")

    def dispatch_exit(self, sym: str, reason: str, ltp: float, pos_data: dict) -> None:
        """
        Routes an emergency tick-level exit signal to console.
        """
        print("\n" + "="*60)
        print(f"🛑 [POSITION EXIT] {sym}")
        print(f"   Reason: {reason}")
        print(f"   Exit Price: {ltp}")
        print(f"   Original Setup: {pos_data.get('direction')} @ {pos_data.get('entry')}")
        print("="*60 + "\n")
        
        # Day-Wise Folder for Exits
        safe_sym = sym.replace(":", "_")
        today_str = datetime.now().strftime("%Y%m%d")
        sym_dir = self.base_dir / safe_sym / today_str
        sym_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = sym_dir / "day_exits.jsonl"
        exit_data = {
            "symbol": sym,
            "type": "EXIT",
            "reason": reason,
            "exit_price": ltp,
            "pos_data": pos_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Append to continuous log
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(exit_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write exit signal to {log_file}: {e}")
            
        # Overwrite Latest Exit State
        latest_exit_file = sym_dir / "latest_exit.json"
        try:
            with open(latest_exit_file, "w") as f:
                json.dump(exit_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to overwrite {latest_exit_file}: {e}")

    def dispatch_tick(self, sym: str, tick_data: dict) -> None:
        """
        Logs raw tick data into a daily JSONL file for historical replay.
        """
        safe_sym = sym.replace(":", "_")
        today_str = datetime.now().strftime("%Y%m%d")
        sym_dir = self.base_dir / safe_sym / today_str
        sym_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = sym_dir / "ticks.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(tick_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write tick data to {log_file}: {e}")
