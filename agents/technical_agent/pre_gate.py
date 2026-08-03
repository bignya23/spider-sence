import logging
from typing import Dict, List
import pandas as pd
from datetime import datetime, time

logger = logging.getLogger("pre_gate")

class UniversePreGate:
    """
    Step 0 Filter: Executes at 09:30 AM to prune the universe of stocks down
    to only highly active, in-play symbols for the day.
    """
    
    def __init__(self, gap_threshold: float = 0.005, vol_anomaly_threshold: float = 1.5):
        self.gap_threshold = gap_threshold
        self.vol_anomaly_threshold = vol_anomaly_threshold
        
    def filter_universe(self, histories: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Evaluates the first 15 minutes of trading (09:15 - 09:30) for each symbol.
        Requires gap > 0.5% OR relative volume > 1.5x to keep the stock in play.
        """
        in_play: List[str] = []
        
        logger.info(f"🛡️ Executing 09:30 AM Pre-Gate across {len(histories)} stocks...")
        
        for sym, df in histories.items():
            if df.empty or len(df) < 50:
                continue
                
            try:
                # We need the most recent 3 candles (assuming 5m base: 09:15, 09:20, 09:25)
                # and historical data to compare against.
                
                # Extract the final 3 candles of the dataframe which represent today's open
                recent_3 = df.iloc[-3:]
                
                # Verify these are actually today's opening candles by checking their timestamps
                # (Optional strict check, but typically the last 3 candles at 09:30 *are* the opening candles)
                
                # 1. Calculate Gap %
                # We need the previous day's close. Since this is a 5m chart, we look back before today's first candle.
                # A simple heuristic: get the close of the 4th candle from the end.
                prev_close = df.iloc[-4]['close']
                today_open = recent_3.iloc[0]['open']
                
                gap_pct = abs((today_open - prev_close) / prev_close)
                
                # 2. Calculate Volume Anomaly
                # Sum the volume of the first 3 5-min candles (15m total)
                today_15m_vol = recent_3['volume'].sum()
                
                # Calculate the average 15-minute volume historically.
                # To be precise, we calculate average volume per 5m candle over the whole history, then multiply by 3.
                avg_5m_vol = df['volume'].mean()
                avg_15m_vol = avg_5m_vol * 3
                
                rvol = today_15m_vol / avg_15m_vol if avg_15m_vol > 0 else 0
                
                # Pre-Gate Logic (OR Condition)
                passed = False
                reasons = []
                
                if gap_pct > self.gap_threshold:
                    passed = True
                    reasons.append(f"Gap {gap_pct*100:.2f}%")
                    
                if rvol > self.vol_anomaly_threshold:
                    passed = True
                    reasons.append(f"RVOL {rvol:.2f}x")
                    
                if passed:
                    in_play.append(sym)
                    logger.debug(f"[IN PLAY] {sym} passed Pre-Gate: {' | '.join(reasons)}")
                else:
                    logger.debug(f"[REJECTED] {sym} failed Pre-Gate (Gap: {gap_pct*100:.2f}%, RVOL: {rvol:.2f}x)")
                    
            except Exception as e:
                logger.error(f"Error processing Pre-Gate for {sym}: {e}")
                
        logger.info(f"✅ Pre-Gate complete. {len(in_play)}/{len(histories)} stocks remain in play.")
        return in_play
