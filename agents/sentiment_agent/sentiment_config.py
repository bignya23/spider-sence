"""sentiment_config.py — Configuration for the Sentiment Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class SentimentConfig:
    # Azure OpenAI
    azure_endpoint: str
    azure_key: str
    azure_deployment: str

    # News MCP URL
    news_mcp_url: str

    # Data lookback
    news_days: int = 3
    disclosures_days: int = 30
    social_limit: int = 50

    # Score weights (must sum to 1.0)
    weight_news: float = 0.40
    weight_filings: float = 0.25
    weight_tone: float = 0.15
    weight_social: float = 0.15
    weight_alpha_vantage: float = 0.05


def load_config() -> SentimentConfig:
    return SentimentConfig(
        azure_endpoint=os.getenv("AZURE_ENDPOINT", ""),
        azure_key=os.getenv("AZURE_KEY", ""),
        azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
        news_mcp_url=os.getenv("NEWS_MCP_URL", "http://127.0.0.1:8000/mcp/news"),
    )
