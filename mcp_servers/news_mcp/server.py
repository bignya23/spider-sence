"""News MCP server.

Exposes 4 tools:
  - get_news(symbol, company_name, days)
  - get_corporate_disclosures(symbol, company_name, days)
  - get_social_posts(symbol, company_name, limit)
  - get_alpha_vantage_sentiment(ticker, limit)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastmcp import FastMCP

from .config import load_config
from .rss_parser import fetch_rss_articles
from .nse_scraper import fetch_corporate_disclosures
from .reddit_client import fetch_reddit_posts

logger = logging.getLogger("news_mcp")
mcp = FastMCP("news")


def _build_keywords(symbol: str, company_name: str) -> list[str]:
    """Build search keyword list from symbol and company name."""
    keywords = []
    if company_name:
        keywords.append(company_name)
        # Add first word of company name (e.g. "Reliance" from "Reliance Industries")
        first_word = company_name.split()[0]
        if len(first_word) > 3:
            keywords.append(first_word)
    # Extract ticker from symbol e.g. "NSE:RELIANCE-EQ" → "RELIANCE"
    try:
        ticker = symbol.split(":")[1].split("-")[0]
        if ticker not in keywords:
            keywords.append(ticker)
    except (IndexError, AttributeError):
        pass
    return keywords


@mcp.tool()
def get_news(
    symbol: str,
    company_name: str = "",
    days: int = 3,
) -> list[dict]:
    """
    Fetch recent news articles about a stock from Indian financial RSS feeds
    and NewsAPI.org.

    Args:
        symbol: Fyers symbol e.g. 'NSE:RELIANCE-EQ'
        company_name: Human-readable company name e.g. 'Reliance Industries'
        days: Number of days to look back (default: 3)

    Returns:
        List of Article dicts with keys: title, description, url, source, published_at
    """
    cfg = load_config()
    keywords = _build_keywords(symbol, company_name)

    # 1. RSS feeds (always available)
    articles = fetch_rss_articles(keywords, days=days)

    # 2. NewsAPI (if key available)
    if cfg.news_api_key:
        try:
            query = " OR ".join(f'"{kw}"' for kw in keywords)
            resp = httpx.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "apiKey": cfg.news_api_key,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("articles", []):
                    articles.append({
                        "title": item.get("title") or "",
                        "description": (item.get("description") or "")[:500],
                        "url": item.get("url") or "",
                        "source": item.get("source", {}).get("name") or "NewsAPI",
                        "published_at": item.get("publishedAt") or "",
                    })
        except Exception as exc:
            logger.warning("NewsAPI fetch failed: %s", exc)

    # Deduplicate by title
    seen = set()
    unique = []
    for art in articles:
        key = art["title"].strip().lower()[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(art)

    return unique


@mcp.tool()
def get_corporate_disclosures(
    symbol: str,
    company_name: str = "",
    days: int = 30,
) -> list[dict]:
    """
    Fetch recent corporate announcements from BSE India for a stock.

    Args:
        symbol: Fyers symbol e.g. 'NSE:RELIANCE-EQ'
        company_name: Human-readable name e.g. 'Reliance Industries'
        days: How many days back to look (default: 30)

    Returns:
        List of Disclosure dicts with keys: type, subject, date, attachment_url, is_significant
    """
    return fetch_corporate_disclosures(symbol, company_name=company_name, days=days)


@mcp.tool()
def get_social_posts(
    symbol: str,
    company_name: str = "",
    limit: int = 50,
) -> list[dict]:
    """
    Fetch recent Reddit posts mentioning the stock from Indian investment subreddits.

    Args:
        symbol: Fyers symbol e.g. 'NSE:RELIANCE-EQ'
        company_name: Human-readable name e.g. 'Reliance Industries'
        limit: Max posts to return (default: 50)

    Returns:
        List of Post dicts with keys: title, body_snippet, subreddit, score, num_comments, created_utc
    """
    cfg = load_config()
    keywords = _build_keywords(symbol, company_name)
    return fetch_reddit_posts(
        keywords,
        limit=limit,
        client_id=cfg.reddit_client_id,
        client_secret=cfg.reddit_client_secret,
        user_agent=cfg.reddit_user_agent,
    )


@mcp.tool()
def get_alpha_vantage_sentiment(
    ticker: str,
    limit: int = 20,
) -> list[dict]:
    """
    Fetch pre-scored news sentiment from Alpha Vantage.

    Args:
        ticker: NSE ticker symbol e.g. 'RELIANCE' (without exchange prefix/suffix)
        limit: Max articles to return (default: 20)

    Returns:
        List of ScoredArticle dicts with keys: title, url, sentiment_score, sentiment_label,
        relevance_score, source, published_at
    """
    cfg = load_config()
    if not cfg.alpha_vantage_key:
        logger.info("ALPHA_VANTAGE_KEY not set — skipping")
        return []

    try:
        resp = httpx.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "limit": limit,
                "apikey": cfg.alpha_vantage_key,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        feed = data.get("feed", [])
    except Exception as exc:
        logger.warning("Alpha Vantage fetch failed: %s", exc)
        return []

    results = []
    for item in feed[:limit]:
        # Find relevance/sentiment for our specific ticker
        ticker_upper = ticker.upper()
        ticker_sentiment = next(
            (t for t in item.get("ticker_sentiment", []) if t.get("ticker") == ticker_upper),
            {},
        )
        results.append({
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "source": item.get("source") or "",
            "published_at": item.get("time_published") or "",
            "sentiment_score": float(item.get("overall_sentiment_score") or 0),
            "sentiment_label": item.get("overall_sentiment_label") or "Neutral",
            "relevance_score": float(ticker_sentiment.get("relevance_score") or 0),
            "ticker_sentiment_score": float(ticker_sentiment.get("ticker_sentiment_score") or 0),
        })

    return results
