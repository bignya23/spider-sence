"""sentiment_agent.py

Agent 3: Sentiment Analysis Agent.

Collects news, filings, and social data via the news_mcp server,
runs a 3-call LLM pipeline, and saves:
  - <SYMBOL>_sentiment.md   (rich Markdown analyst report)
  - <SYMBOL>_sentiment.json (machine-readable score sidecar)

Usage:
    python -m agents.sentiment_agent.sentiment_agent NSE:RELIANCE-EQ
    python -m agents.sentiment_agent.sentiment_agent NSE:RELIANCE-EQ --company "Reliance Industries"
    python -m agents.sentiment_agent.sentiment_agent NSE:RELIANCE-EQ --days 7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.exceptions import ToolError

from .sentiment_config import SentimentConfig, load_config
from .sentiment_engine import run_sentiment_pipeline
from orchestrator.output_paths import (
    ensure_category_directory,
    format_run_timestamp,
    normalise_symbol,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentiment_agent")


class NewsClient:
    """Thin async wrapper around the news_mcp FastMCP server."""

    def __init__(self, url: str):
        self.client = Client(url)

    async def call(self, tool: str, **kwargs) -> Any:
        async with self.client:
            result = await self.client.call_tool(tool, arguments=kwargs)
            return result.data


class SentimentAgent:
    def __init__(
        self,
        *,
        config: SentimentConfig | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.config = config or load_config()
        self.news = NewsClient(self.config.news_mcp_url)
        self.output_root = output_root

    async def analyse(
        self,
        symbol: str,
        *,
        company_name: str = "",
        days: int | None = None,
        run_timestamp: str | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        run_ts = run_timestamp or format_run_timestamp()
        cfg = self.config
        n_days = days or cfg.news_days

        if not company_name:
            company_name = symbol.split(":")[1].split("-")[0].replace("_", " ").title()

        logger.info("Starting sentiment analysis for %s (%s)", symbol, company_name)

        # ── Collect data concurrently ─────────────────────────────────────────
        async def safe_call(tool: str, **kwargs):
            try:
                return await self.news.call(tool, **kwargs) or []
            except (ToolError, Exception) as exc:
                logger.warning("Tool '%s' failed: %s", tool, exc)
                return []

        # Extract ticker for Alpha Vantage (e.g. "NSE:RELIANCE-EQ" → "RELIANCE")
        av_ticker = symbol.split(":")[1].split("-")[0]

        news_articles, disclosures, social_posts, av_articles = await asyncio.gather(
            safe_call("get_news", symbol=symbol, company_name=company_name, days=n_days),
            safe_call("get_corporate_disclosures", symbol=symbol, company_name=company_name, days=cfg.disclosures_days),
            safe_call("get_social_posts", symbol=symbol, company_name=company_name, limit=cfg.social_limit),
            safe_call("get_alpha_vantage_sentiment", ticker=av_ticker, limit=20),
        )

        logger.info(
            "Collected: %d news articles, %d filings, %d social posts, %d AV articles",
            len(news_articles), len(disclosures), len(social_posts), len(av_articles),
        )

        # ── Run LLM pipeline ──────────────────────────────────────────────────
        now_ist = datetime.now(timezone.utc).astimezone(
            tz=timezone(timedelta(hours=5, minutes=30), name="IST")
        )
        generated_at = now_ist.strftime("%Y-%m-%d %H:%M")

        markdown_report, sidecar = run_sentiment_pipeline(
            symbol=symbol,
            company_name=company_name,
            news_articles=news_articles,
            disclosures=disclosures,
            social_posts=social_posts,
            av_articles=av_articles,
            config=cfg,
            days=n_days,
            generated_at=generated_at,
        )

        # ── Save outputs ──────────────────────────────────────────────────────
        safe_sym = normalise_symbol(symbol)
        out_dir, _ = ensure_category_directory(
            symbol, "sentiment", run_timestamp=run_ts, root=self.output_root
        )

        md_path = out_dir / f"{safe_sym}_sentiment.md"
        json_path = out_dir / f"{safe_sym}_sentiment.json"

        md_path.write_text(markdown_report, encoding="utf-8")
        json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Sentiment report saved → %s", md_path)

        return {
            "symbol": symbol,
            "run_timestamp": run_ts,
            "sidecar": sidecar,
            "md_path": str(md_path),
            "json_path": str(json_path),
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Spider-Sence Sentiment Agent (Agent 3)"
    )
    parser.add_argument("symbol", help="Fyers symbol e.g. NSE:RELIANCE-EQ")
    parser.add_argument(
        "--company", "-c",
        default="",
        help="Company name hint e.g. 'Reliance Industries' (improves news search)",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=3,
        help="Days of news to look back (default: 3)",
    )
    args = parser.parse_args()

    agent = SentimentAgent()
    result = await agent.analyse(
        args.symbol,
        company_name=args.company,
        days=args.days,
    )

    sidecar = result["sidecar"]
    sent = sidecar.get("sentiment", {})
    score = sent.get("score", 0)
    label = sent.get("label", "N/A")
    confidence = sent.get("confidence", "N/A")
    catalyst = sent.get("key_catalyst") or "N/A"

    # Pretty CLI summary
    emoji_map = {"STRONGLY_BULLISH": "🟢🟢", "MODERATELY_BULLISH": "🟢",
                 "SLIGHTLY_BULLISH": "🟡", "NEUTRAL": "⚪",
                 "SLIGHTLY_BEARISH": "🟠", "MODERATELY_BEARISH": "🔴",
                 "STRONGLY_BEARISH": "🔴🔴"}

    print("\n" + "═" * 68)
    print(f"  🧠 SENTIMENT ANALYSIS  ({args.symbol})")
    print("═" * 68)
    print(f"  Score:      {score:+.1f} / 10  {emoji_map.get(label, '')}")
    print(f"  Verdict:    {label}")
    print(f"  Confidence: {confidence.upper()}")
    print(f"  Catalyst:   {catalyst}")

    bd = sidecar.get("breakdown", {})
    print("\n  Score Breakdown:")
    print(f"    📰 News:            {bd.get('news_score', 0):+.2f}")
    print(f"    🗂️ Filings:          {bd.get('filings_score', 0):+.2f}")
    print(f"    🎤 Tone:            {bd.get('tone_score', 0):+.2f}")
    print(f"    💬 Social:          {bd.get('social_score', 0):+.2f}")
    print(f"    📈 Alpha Vantage:   {bd.get('alpha_vantage_score', 0):+.2f}")
    src = sidecar.get("sources_used", {})
    print(f"\n  Sources: {src.get('news_articles',0)} articles  |  "
          f"{src.get('corporate_filings',0)} filings  |  "
          f"{src.get('social_posts',0)} posts")
    print("═" * 68)
    print(f"\n📝 Report → {result['md_path']}")
    print(f"💾 JSON   → {result['json_path']}\n")


if __name__ == "__main__":
    asyncio.run(_main())
