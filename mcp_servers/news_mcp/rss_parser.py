"""rss_parser.py

Fetches and filters articles from major Indian financial RSS feeds.
No API key needed — all feeds are publicly available.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx

logger = logging.getLogger("news_mcp.rss")

# Major Indian financial news RSS feeds
RSS_FEEDS = {
    "Economic Times — Markets": "https://economictimes.indiatimes.com/markets/rss.cms",
    "Economic Times — Stocks": "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    "Moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
    "Mint — Markets": "https://www.livemint.com/rss/markets",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


def _entry_to_article(entry: Any, source: str) -> dict:
    """Convert a feedparser entry dict to our standard Article dict."""
    published = None
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        except Exception:
            published = None

    title = getattr(entry, "title", "") or ""
    description = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
    url = getattr(entry, "link", "") or ""

    # Strip HTML from description
    import re
    description = re.sub(r"<[^>]+>", "", description).strip()

    return {
        "title": title,
        "description": description[:500],
        "url": url,
        "source": source,
        "published_at": published,
    }


def _is_relevant(article: dict, keywords: list[str]) -> bool:
    """Return True if any keyword appears in the title or description."""
    text = (article["title"] + " " + article["description"]).lower()
    return any(kw.lower() in text for kw in keywords)


def _is_recent(article: dict, days: int) -> bool:
    """Return True if the article was published within `days` days."""
    if not article.get("published_at"):
        return True  # include if no date (can't filter)
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        pub = datetime.fromisoformat(article["published_at"])
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        return pub >= cutoff
    except Exception:
        return True


def fetch_rss_articles(
    keywords: list[str],
    days: int = 3,
    max_per_feed: int = 20,
) -> list[dict]:
    """
    Fetch and filter articles from all RSS feeds.

    Args:
        keywords: Company name, symbol fragments, or search terms.
        days: Only return articles published in the last N days.
        max_per_feed: Maximum articles to parse per feed.

    Returns:
        Deduplicated list of matching Article dicts.
    """
    articles: list[dict] = []
    seen_titles: set[str] = set()

    for source_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url, agent=_HEADERS["User-Agent"])
            for entry in feed.entries[:max_per_feed]:
                art = _entry_to_article(entry, source_name)
                title_key = art["title"].strip().lower()[:80]
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                if _is_relevant(art, keywords) and _is_recent(art, days):
                    articles.append(art)
        except Exception as exc:
            logger.warning("RSS fetch failed for %s: %s", source_name, exc)

    # Add Google News specific search
    if keywords:
        try:
            import urllib.parse
            primary_kw = keywords[0]
            q = urllib.parse.quote(primary_kw)
            gnews_url = f"https://news.google.com/rss/search?q={q}+stock&hl=en-IN&gl=IN&ceid=IN:en"
            
            # Google blocks feedparser default agent, so we fetch with httpx first
            resp = httpx.get(gnews_url, headers=_HEADERS, timeout=10)
            if resp.status_code == 200:
                feed = feedparser.parse(resp.text)
                for entry in feed.entries[:max_per_feed]:
                    art = _entry_to_article(entry, "Google News")
                    title_key = art["title"].strip().lower()[:80]
                    if title_key in seen_titles:
                        continue
                    seen_titles.add(title_key)
                    # No need to filter relevance since Google News did it, just check recency
                    if _is_recent(art, days):
                        articles.append(art)
        except Exception as exc:
            logger.warning("Google News RSS fetch failed: %s", exc)

    return articles
