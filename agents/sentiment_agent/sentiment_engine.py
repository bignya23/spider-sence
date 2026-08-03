"""sentiment_engine.py

3-call LLM pipeline using Azure OpenAI GPT-5.

Call 1 — Event Classification:  headline list → structured events JSON
Call 2 — Tone Analysis:          article bodies → tone/confidence scores
Call 3 — MD Report Generation:  all signals → full Markdown analyst report

The final output is a rich Markdown report + a minimal JSON sidecar.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from orchestrator.llm_client import LLMClient

from .sentiment_config import SentimentConfig

logger = logging.getLogger("sentiment_agent.engine")

# ── Score label mapping ────────────────────────────────────────────────────────
def score_to_label(score: float) -> str:
    if score >= 7:
        return "STRONGLY_BULLISH"
    elif score >= 4:
        return "MODERATELY_BULLISH"
    elif score >= 1:
        return "SLIGHTLY_BULLISH"
    elif score >= -1:
        return "NEUTRAL"
    elif score >= -4:
        return "SLIGHTLY_BEARISH"
    elif score >= -7:
        return "MODERATELY_BEARISH"
    else:
        return "STRONGLY_BEARISH"


def score_to_emoji(score: float) -> str:
    if score >= 4:
        return "🟢"
    elif score >= 1:
        return "🟡"
    elif score >= -1:
        return "⚪"
    elif score >= -4:
        return "🟠"
    else:
        return "🔴"


# ── Prompt templates ───────────────────────────────────────────────────────────

_EVENT_SYSTEM = """\
You are a professional Indian equity analyst trained to detect market-moving events in financial text.
Your task is to analyse a list of news headlines and corporate disclosures, and identify all significant events.

For each event, output a JSON object with these exact fields:
- type: one of [EARNINGS_BEAT, EARNINGS_MISS, REVENUE_SURPRISE_POSITIVE, REVENUE_SURPRISE_NEGATIVE,
  CONTRACT_WIN, REGULATORY_ACTION, MANAGEMENT_CHANGE_POSITIVE, MANAGEMENT_CHANGE_NEGATIVE,
  STAKE_PLEDGE_INCREASE, STAKE_PLEDGE_DECREASE, DEBT_REDUCTION, FUND_RAISE, QIP_RIGHTS_ISSUE,
  ANALYST_UPGRADE, ANALYST_DOWNGRADE, BUYBACK, DIVIDEND_INCREASE, FII_OUTFLOW, FII_INFLOW,
  MACRO_RISK, MACRO_POSITIVE, CONTEXTUAL_BEAT, CONTEXTUAL_MISS, OTHER]
- headline: the original headline text
- source: publication name
- impact: "positive", "negative", or "neutral"
- weight: float 0.0 to 1.0 (magnitude/importance of this event)
- reasoning: one sentence explaining why this is positive or negative

Focus on events directly relevant to the target company or its sector.
Be especially careful with contextual beats — a "2% revenue drop" may be bullish if analysts expected 5%.
Return ONLY a valid JSON array. No markdown, no explanation outside the array.
"""

_TONE_SYSTEM = """\
You are an expert in analysing the tone and language of financial documents.
Your task is to evaluate the tone of management commentary, analyst reports, or news articles.

Output a JSON object with these exact fields:
- management_confidence: float 1.0 to 5.0 (1=highly defensive/legal, 5=assertive with concrete guidance)
- forward_guidance: one of ["positive", "neutral", "cautious", "withdrawn"]
- language_flags: list of applicable flags from:
  ["growth_affirmations", "capex_commitment", "buyback_signals", "dividend_increase",
   "hedging_terms", "legal_boilerplate", "withdrawn_guidance", "auditor_qualification",
   "promoter_concern", "debt_stress", "margin_pressure"]
- tone_summary: 2-3 sentence plain English summary of the overall tone
- key_quote: the single most revealing phrase or sentence from the text

Return ONLY a valid JSON object. No markdown, no explanation.
"""

_REPORT_TEMPLATE = """\
You are a senior equity research analyst writing a professional sentiment report.
Using the data below, write a complete Markdown report following the EXACT template provided.
Be analytical, specific, and professional. Use actual numbers and names from the data.
Do NOT add extra sections. Do NOT omit any section. Fill every placeholder.

---
TEMPLATE START
---

# 📊 Sentiment Report: {symbol}
**Generated:** {generated_at} IST  |  **Confidence:** {confidence}  |  **Data window:** Last {days} days

---

## Overall Verdict

### {emoji} {label}  |  Score: {score} / 10

> *"[Write a 2-3 sentence analytical narrative summarising WHY this score was reached, 
> referencing specific events, tone, and market context. Be direct and professional.]*"*

**Key Catalyst:** [Most impactful single event detected]

---

## Events Detected

| # | Event Type | Headline | Source | Impact |
|---|---|---|---|---|
[For each event, add a row: | N | `TYPE` | headline | source | 🟢/🔴/🟡 |]
[If no events: | — | No significant events detected | — | — | ⚪ Neutral |]

---

## Management & Analyst Tone

| Dimension | Assessment |
|---|---|
| **Confidence Level** | [X/5 with bar: ████████░░] |
| **Forward Guidance** | [positive/neutral/cautious/withdrawn] |
| **Language Signals** | [comma-separated flags] |

> *"[tone_summary from tone analysis]*"*

---

## Social Pulse (Reddit)

- **Posts Analysed:** [N] across r/IndianStockMarket, r/DalalStreet, r/Nifty
- **Community Bias:** [🟢/🔴/⚪] [bullish/bearish/neutral] (avg vote score: +N)
- **Top Post:** *"[title of highest-voted post]"* (↑ N votes)

[If no Reddit data: *No Reddit data available — social signal skipped.*]

---

## Score Breakdown

| Source | Score | Weight | Contribution |
|---|---|---|---|
| 📰 News Events | {news_score}/10 | 40% | {news_contrib} |
| 🗂️ Corporate Filings | {filings_score}/10 | 25% | {filings_contrib} |
| 🎤 Management Tone | {tone_score}/10 | 15% | {tone_contrib} |
| 💬 Social Sentiment | {social_score}/10 | 15% | {social_contrib} |
| 📈 Alpha Vantage | {av_score}/10 | 5% | {av_contrib} |
| **FINAL SCORE** | **{final_score}/10** | **100%** | **{label}** |

---

## Risk Flags

[List any negative signals, concerns, or caveats as bullet points. 
If none: *No significant risk flags detected in this analysis window.*]

---

## Sources Used
- [N] news articles ([breakdown by outlet])
- [N] corporate filings (BSE disclosures)
- [N] Reddit posts
- [N] Alpha Vantage scored articles

---
TEMPLATE END
---

Now write the report using this data:

SYMBOL: {symbol}
EVENTS: {events_json}
TONE ANALYSIS: {tone_json}
SOCIAL DATA: {social_json}
SCORE BREAKDOWN: {scores_json}
GENERATED AT: {generated_at}
DAYS: {days}
"""


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _llm_call(
    client: LLMClient,
    system: str,
    user: str,
) -> str:
    """Single LLM call, returns the text content of the response."""
    return client.complete(system_prompt=system, user_prompt=user)


def _safe_json(text: str) -> Any:
    """Try to parse JSON from an LLM response, tolerating minor formatting issues."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON, returning empty fallback")
        return None


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _score_events(events: list[dict]) -> float:
    """Convert event list to a -10..+10 score."""
    if not events:
        return 0.0
    total_weight = sum(e.get("weight", 0.5) for e in events)
    if total_weight == 0:
        return 0.0
    weighted_sum = 0.0
    for e in events:
        w = e.get("weight", 0.5)
        impact = e.get("impact", "neutral")
        direction = 1 if impact == "positive" else (-1 if impact == "negative" else 0)
        weighted_sum += direction * w
    raw = weighted_sum / total_weight  # -1 to +1
    return round(raw * 10, 2)  # -10 to +10


def _score_tone(tone: dict | None) -> float:
    """Convert tone analysis to a -10..+10 score."""
    if not tone:
        return 0.0
    conf = float(tone.get("management_confidence") or 3.0)  # 1-5
    conf_score = (conf - 3) / 2 * 5  # -5 to +5, centred on 3

    guidance = tone.get("forward_guidance", "neutral")
    guidance_map = {"positive": 3.0, "neutral": 0.0, "cautious": -3.0, "withdrawn": -6.0}
    guidance_score = guidance_map.get(guidance, 0.0)

    flags = tone.get("language_flags") or []
    positive_flags = {"growth_affirmations", "capex_commitment", "buyback_signals", "dividend_increase"}
    negative_flags = {"hedging_terms", "legal_boilerplate", "withdrawn_guidance",
                      "auditor_qualification", "promoter_concern", "debt_stress", "margin_pressure"}
    flag_score = sum(1 for f in flags if f in positive_flags) - sum(1 for f in flags if f in negative_flags)
    flag_score = max(-3.0, min(3.0, flag_score))  # cap at ±3

    raw = conf_score + guidance_score + flag_score
    return round(max(-10, min(10, raw)), 2)


def _score_social(posts: list[dict]) -> float:
    """Convert Reddit posts to a -10..+10 sentiment score."""
    if not posts:
        return 0.0
    # Use average upvote ratio as proxy; above 0.6 = bullish, below 0.4 = bearish
    avg_ratio = sum(p.get("upvote_ratio", 0.5) for p in posts) / len(posts)
    raw = (avg_ratio - 0.5) * 20  # 0.5 ratio = 0, 1.0 = +10, 0.0 = -10
    return round(raw, 2)


def _score_alpha_vantage(articles: list[dict]) -> float:
    """Convert Alpha Vantage scores to -10..+10."""
    if not articles:
        return 0.0
    relevant = [a for a in articles if a.get("relevance_score", 0) > 0.3]
    if not relevant:
        return 0.0
    avg = sum(a.get("ticker_sentiment_score") or a.get("sentiment_score") or 0 for a in relevant) / len(relevant)
    # Alpha Vantage scores are roughly -1 to +1
    return round(avg * 10, 2)


def _score_filings(disclosures: list[dict]) -> float:
    """Heuristic score for corporate filings — significant ones carry more weight."""
    if not disclosures:
        return 0.0  # neutral if no data

    POSITIVE_TYPES = {
        "financial results", "dividend", "buyback", "bonus", "contract", "order",
        "credit rating upgrade", "release of pledge", "debt reduction",
    }
    NEGATIVE_TYPES = {
        "pledging", "legal proceedings", "auditor", "cessation", "resignation",
        "fund raising", "qip", "rights issue", "rating downgrade",
    }

    score = 0.0
    count = 0
    for d in disclosures:
        text = (d.get("type") or "" + " " + (d.get("subject") or "")).lower()
        pos = any(t in text for t in POSITIVE_TYPES)
        neg = any(t in text for t in NEGATIVE_TYPES)
        if pos:
            score += 4.0
            count += 1
        elif neg:
            score -= 4.0
            count += 1
        elif d.get("is_significant"):
            count += 1

    if count == 0:
        return 0.0
    return round(max(-10, min(10, score / count)), 2)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_sentiment_pipeline(
    symbol: str,
    company_name: str,
    news_articles: list[dict],
    disclosures: list[dict],
    social_posts: list[dict],
    av_articles: list[dict],
    config: SentimentConfig,
    days: int,
    generated_at: str,
) -> tuple[str, dict]:
    """
    Run the full 3-call LLM sentiment pipeline.

    Returns:
        (markdown_report: str, sidecar_json: dict)
    """
    client = LLMClient()

    # ── Call 1: Event Classification ──────────────────────────────────────────
    events: list[dict] = []
    if news_articles or disclosures:
        headlines = [
            {"title": a["title"], "source": a.get("source", ""), "type": "news"}
            for a in news_articles[:30]
        ] + [
            {"title": d["subject"], "source": "BSE Disclosure", "type": "filing"}
            for d in disclosures[:20]
        ]
        user_msg = (
            f"Company: {company_name} ({symbol})\n\n"
            f"Headlines and disclosures:\n{json.dumps(headlines, indent=2)}"
        )
        raw = _llm_call(client, _EVENT_SYSTEM, user_msg)
        parsed = _safe_json(raw)
        if isinstance(parsed, list):
            events = parsed
        logger.info("Event classification found %d events", len(events))

    # ── Call 2: Tone Analysis ─────────────────────────────────────────────────
    tone: dict | None = None
    article_bodies = [
        a["description"] for a in news_articles[:5] if a.get("description")
    ]
    if article_bodies:
        user_msg = (
            f"Analyse the tone of these articles/excerpts about {company_name}:\n\n"
            + "\n\n---\n\n".join(article_bodies)
        )
        raw = _llm_call(client, _TONE_SYSTEM, user_msg)
        parsed = _safe_json(raw)
        if isinstance(parsed, dict):
            tone = parsed
        logger.info("Tone analysis complete: confidence=%.1f", tone.get("management_confidence", 0) if tone else 0)

    # ── Compute Sub-scores ────────────────────────────────────────────────────
    news_score = _score_events(events)
    filings_score = _score_filings(disclosures)
    tone_score = _score_tone(tone)
    social_score = _score_social(social_posts)
    av_score = _score_alpha_vantage(av_articles)

    w = config
    final_score = round(
        news_score * w.weight_news
        + filings_score * w.weight_filings
        + tone_score * w.weight_tone
        + social_score * w.weight_social
        + av_score * w.weight_alpha_vantage,
        2,
    )

    label = score_to_label(final_score)
    emoji = score_to_emoji(final_score)
    confidence = "high" if abs(final_score) >= 6 else "medium" if abs(final_score) >= 3 else "low"

    # Social summary
    social_summary = {}
    if social_posts:
        avg_score = sum(p.get("score", 0) for p in social_posts) / len(social_posts)
        avg_ratio = sum(p.get("upvote_ratio", 0.5) for p in social_posts) / len(social_posts)
        top = max(social_posts, key=lambda p: p.get("score", 0), default={})
        social_summary = {
            "posts_analyzed": len(social_posts),
            "avg_vote_score": round(avg_score, 0),
            "avg_upvote_ratio": round(avg_ratio, 2),
            "community_bias": "bullish" if avg_ratio > 0.6 else "bearish" if avg_ratio < 0.4 else "neutral",
            "top_post": top.get("title", "N/A"),
            "top_post_score": top.get("score", 0),
        }

    scores_json = {
        "news_score": news_score,
        "filings_score": filings_score,
        "tone_score": tone_score,
        "social_score": social_score,
        "alpha_vantage_score": av_score,
        "final_score": final_score,
        "label": label,
        "confidence": confidence,
        "weights": {
            "news": f"{int(w.weight_news*100)}%",
            "filings": f"{int(w.weight_filings*100)}%",
            "tone": f"{int(w.weight_tone*100)}%",
            "social": f"{int(w.weight_social*100)}%",
            "alpha_vantage": f"{int(w.weight_alpha_vantage*100)}%",
        },
    }

    # ── Call 3: Markdown Report ───────────────────────────────────────────────
    def _contrib(score: float, weight: float) -> str:
        return f"{round(score * weight, 2):+.2f}"

    user_msg = _REPORT_TEMPLATE.format(
        symbol=symbol,
        generated_at=generated_at,
        days=days,
        confidence=confidence,
        emoji=emoji,
        label=label,
        score=final_score,
        events_json=json.dumps(events, indent=2) if events else "[]",
        tone_json=json.dumps(tone, indent=2) if tone else "{}",
        social_json=json.dumps(social_summary, indent=2),
        scores_json=json.dumps(scores_json, indent=2),
        news_score=news_score,
        filings_score=filings_score,
        tone_score=tone_score,
        social_score=social_score,
        av_score=av_score,
        final_score=final_score,
        news_contrib=_contrib(news_score, w.weight_news),
        filings_contrib=_contrib(filings_score, w.weight_filings),
        tone_contrib=_contrib(tone_score, w.weight_tone),
        social_contrib=_contrib(social_score, w.weight_social),
        av_contrib=_contrib(av_score, w.weight_alpha_vantage),
    )

    markdown_report = _llm_call(
        client,
        "You are a professional financial analyst. Write exactly as instructed. No deviations.",
        user_msg,
    )

    # ── Sidecar JSON ──────────────────────────────────────────────────────────
    sidecar = {
        "symbol": symbol,
        "generated_at": generated_at,
        "sentiment": {
            "score": final_score,
            "label": label,
            "confidence": confidence,
            "key_catalyst": events[0].get("headline") if events else None,
        },
        "breakdown": {
            "news_score": news_score,
            "filings_score": filings_score,
            "tone_score": tone_score,
            "social_score": social_score,
            "alpha_vantage_score": av_score,
        },
        "events_count": len(events),
        "sources_used": {
            "news_articles": len(news_articles),
            "corporate_filings": len(disclosures),
            "social_posts": len(social_posts),
            "alpha_vantage_articles": len(av_articles),
        },
    }

    return markdown_report, sidecar
