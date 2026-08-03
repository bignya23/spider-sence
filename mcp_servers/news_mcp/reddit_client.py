"""reddit_client.py

Searches Reddit for stock mentions using PRAW.
Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("news_mcp.reddit")

SUBREDDITS = [
    "IndianStockMarket",
    "DalalStreet",
    "Nifty",
    "IndiaInvestments",
]


def fetch_reddit_posts(
    keywords: list[str],
    limit: int = 50,
    *,
    client_id: str,
    client_secret: str,
    user_agent: str,
) -> list[dict]:
    """
    Search subreddits for posts mentioning the stock.

    Args:
        keywords: Company name / ticker fragments to search.
        limit: Max posts to fetch per subreddit.
        client_id: Reddit app client ID.
        client_secret: Reddit app client secret.
        user_agent: User-agent string for Reddit API.

    Returns:
        List of Post dicts.
    """
    if not client_id or not client_secret:
        logger.info("Reddit credentials not set — skipping social data")
        return []

    try:
        import praw
    except ImportError:
        logger.warning("praw not installed — run: pip install praw")
        return []

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            ratelimit_seconds=2,
        )
    except Exception as exc:
        logger.warning("Reddit init failed: %s", exc)
        return []

    query = " OR ".join(f'"{kw}"' for kw in keywords)
    posts: list[dict] = []
    seen_ids: set[str] = set()

    for sub_name in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub_name)
            results = subreddit.search(query, sort="relevance", time_filter="week", limit=limit)
            for post in results:
                if post.id in seen_ids:
                    continue
                seen_ids.add(post.id)
                body_snippet = (post.selftext or "")[:400].strip()
                posts.append({
                    "id": post.id,
                    "title": post.title,
                    "body_snippet": body_snippet,
                    "url": f"https://www.reddit.com{post.permalink}",
                    "subreddit": f"r/{sub_name}",
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "upvote_ratio": post.upvote_ratio,
                    "created_utc": str(post.created_utc),
                })
        except Exception as exc:
            logger.warning("Reddit search failed for r/%s: %s", sub_name, exc)

    # Sort by score descending
    posts.sort(key=lambda p: p["score"], reverse=True)
    return posts[:limit]
