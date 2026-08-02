"""trading_styles.py

Defines trading style profiles for the technical analysis pipeline.

Each style specifies:
  - Which timeframes to use as (bias, structure, entry) TF hierarchy
  - Minimum confluence score to generate a setup
  - Minimum ADX on the bias TF to confirm trend strength
  - Display name and description

Usage:
    from agents.technical_agent.trading_styles import TRADING_STYLES, TradingStyleProfile
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class TradingStyleProfile:
    """Configuration for a specific trading style."""

    name: str
    display_name: str
    description: str

    # 3-TF hierarchy: (bias_tf, structure_tf, entry_tf)
    # These keys must exist in the fetched timeframes dict
    bias_tf: str
    structure_tf: str
    entry_tf: str

    # Quality thresholds
    min_confluence: float = 3.0    # Minimum score (0–10) to generate any setup
    min_adx: float = 20.0          # ADX on bias TF confirming trend exists
    min_rr: float = 1.5            # Minimum risk/reward ratio to accept setup

    # Graph subfolder name
    graph_folder: str = "graphs"


TRADING_STYLES: Dict[str, TradingStyleProfile] = {
    "scalp": TradingStyleProfile(
        name="scalp",
        display_name="Scalp",
        description=(
            "Very short-term trades held for minutes. "
            "Uses hourly for bias, 15-min for structure, 5-min for precise entry. "
            "Suited for liquid stocks with tight spreads."
        ),
        bias_tf="hourly",
        structure_tf="m15",
        entry_tf="m5",
        min_confluence=4.0,   # Higher bar for scalps – need cleaner alignment
        min_adx=25.0,
        min_rr=1.5,
        graph_folder="setups/scalp",
    ),
    "intraday": TradingStyleProfile(
        name="intraday",
        display_name="Intraday",
        description=(
            "Same-day trades, typically held 1–4 hours. "
            "Uses daily for macro bias, hourly for structure, 15-min for entry. "
            "Positions must be squared off by end of session."
        ),
        bias_tf="daily",
        structure_tf="hourly",
        entry_tf="m15",
        min_confluence=3.0,
        min_adx=20.0,
        min_rr=1.5,
        graph_folder="setups/intraday",
    ),
    "swing": TradingStyleProfile(
        name="swing",
        display_name="Swing",
        description=(
            "Trades held 2–10 days to capture a full price swing. "
            "Uses daily for trend, hourly for structure, and 15-min for precise entry. "
            "Relies heavily on Ichimoku cloud, Bollinger Bands, and Pivot Points."
        ),
        bias_tf="daily",
        structure_tf="daily",
        entry_tf="hourly",
        min_confluence=4.0,
        min_adx=22.0,
        min_rr=2.0,
        graph_folder="setups/swing",
    ),
    "positional": TradingStyleProfile(
        name="positional",
        display_name="Positional",
        description=(
            "Medium-term positions held weeks to months. "
            "Uses weekly/daily trend alignment as the primary filter. "
            "Enters on the daily chart at key support/resistance. "
            "Requires very strong multi-week trend confirmation."
        ),
        bias_tf="daily",
        structure_tf="daily",
        entry_tf="daily",
        min_confluence=5.0,  # Needs the strongest alignment
        min_adx=25.0,
        min_rr=2.5,
        graph_folder="setups/positional",
    ),
}
