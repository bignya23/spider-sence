"""config.py — News MCP server configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class NewsConfig:
    news_api_key: str
    alpha_vantage_key: str
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str


def load_config() -> NewsConfig:
    return NewsConfig(
        news_api_key=os.getenv("NEWS_API_KEY", ""),
        alpha_vantage_key=os.getenv("ALPHA_VANTAGE_KEY", ""),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "SpiderSence/1.0"),
    )
