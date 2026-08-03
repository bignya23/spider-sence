from agents.technical_agent.live_scanner import LiveScanner
import traceback

scanner = LiveScanner(["NSE:RELIANCE-EQ"])
scanner.pre_fetch_history()
try:
    df = scanner.engine.histories["NSE:RELIANCE-EQ"]
    scanner.engine._process_single("NSE:RELIANCE-EQ", df)
except Exception as e:
    traceback.print_exc()
