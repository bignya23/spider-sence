import os
from pathlib import Path
from typing import List

MAJOR_INDICES = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX",
    "NSE:FINNIFTY-INDEX",
    "NSE:NIFTYIT-INDEX",
    "NSE:NIFTYAUTO-INDEX",
    "BSE:SENSEX-INDEX"
]

BANK_NIFTY_SYMBOLS = {
    "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:AXISBANK-EQ", 
    "NSE:KOTAKBANK-EQ", "NSE:SBIN-EQ", "NSE:INDUSINDBK-EQ", 
    "NSE:BANKBARODA-EQ", "NSE:PNB-EQ", "NSE:AUBANK-EQ", 
    "NSE:BANDHANBNK-EQ", "NSE:FEDERALBNK-EQ", "NSE:IDFCFIRSTB-EQ",
}

FINNIFTY_SYMBOLS = {
    "NSE:BAJFINANCE-EQ", "NSE:BAJAJFINSV-EQ", "NSE:CHOLAFIN-EQ",
    "NSE:SHRIRAMFIN-EQ", "NSE:MUTHOOTFIN-EQ", "NSE:PFC-EQ", "NSE:RECLTD-EQ"
}

IT_SYMBOLS = {
    "NSE:TCS-EQ", "NSE:INFY-EQ", "NSE:HCLTECH-EQ", "NSE:WIPRO-EQ",
    "NSE:TECHM-EQ", "NSE:LTIM-EQ", "NSE:PERSISTENT-EQ", "NSE:COFORGE-EQ"
}

AUTO_SYMBOLS = {
    "NSE:MARUTI-EQ", "NSE:TATAMOTORS-EQ", "NSE:M&M-EQ", "NSE:BAJAJ-AUTO-EQ",
    "NSE:HEROMOTOCO-EQ", "NSE:EICHERMOT-EQ", "NSE:TVSMOTOR-EQ", "NSE:ASHOKLEY-EQ"
}

def get_parent_index(symbol: str) -> str:
    """
    Returns the appropriate parent index for correlation filtering.
    Defaults to NIFTY 50 if the stock is not explicitly mapped to a specific sector.
    """
    if symbol in BANK_NIFTY_SYMBOLS:
        return "NSE:NIFTYBANK-INDEX"
    elif symbol in FINNIFTY_SYMBOLS:
        return "NSE:FINNIFTY-INDEX"
    elif symbol in IT_SYMBOLS:
        return "NSE:NIFTYIT-INDEX"
    elif symbol in AUTO_SYMBOLS:
        return "NSE:NIFTYAUTO-INDEX"
    
    return "NSE:NIFTY50-INDEX"
