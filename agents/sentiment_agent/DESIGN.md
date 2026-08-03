# Agent 3: Sentiment Analysis Agent — Design Specification

**Status:** Draft — Pending review before implementation  
**Location:** `agents/sentiment_agent/`

---

## 1. What Is This Agent?

The Sentiment Agent reads everything a human analyst would — news, regulatory filings, and community discussion — and converts that text into a structured quantitative output: a **Sentiment Score from -10 to +10**.

It does this through a 3-stage pipeline:
1. **Collect** raw text from multiple sources (news, filings, Reddit)
2. **Analyse** that text using an LLM with specialised prompts
3. **Synthesise** everything into a final scored JSON report

---

## 2. Folder Structure

```
agents/sentiment_agent/
├── __init__.py
├── sentiment_agent.py       ← Entry point / orchestrator (mirrors technical_agent.py)
├── sentiment_engine.py      ← All LLM prompt calls and response parsing
├── event_classifier.py      ← Classifies detected events into standard types
└── sentiment_config.py      ← Config dataclass, weights, thresholds

mcp_servers/news_mcp/
├── __init__.py
├── server.py                ← FastMCP server exposing 4 tools
├── config.py                ← Loads env vars (NEWS_API_KEY, REDDIT_*, etc.)
├── rss_parser.py            ← Fetches and parses Indian financial RSS feeds
├── nse_scraper.py           ← Fetches BSE corporate filings
└── reddit_client.py         ← Reddit PRAW wrapper
```

---

## 3. Data Sources

### 3.1 Financial News — RSS Feeds (Free, Unlimited)

| Feed | URL |
|---|---|
| Economic Times — Markets | `https://economictimes.indiatimes.com/markets/rss.cms` |
| Economic Times — Stocks | `https://economictimes.indiatimes.com/markets/stocks/rss.cms` |
| Moneycontrol — Markets | `https://www.moneycontrol.com/rss/latestnews.xml` |
| Mint — Markets | `https://www.livemint.com/rss/markets` |
| Business Standard | `https://www.business-standard.com/rss/markets-106.rss` |

The RSS parser will filter articles to those **mentioning the target stock symbol or company name** in the title or description.

### 3.2 Structured News — NewsAPI.org (Free: 100 req/day)

- Endpoint: `https://newsapi.org/v2/everything?q=<company>&language=en&sortBy=publishedAt`
- Returns structured JSON with title, description, source, publishedAt
- Used alongside RSS for coverage

**API Key needed:** `NEWS_API_KEY` → sign up free at newsapi.org

### 3.3 Market News Sentiment — Alpha Vantage (Free: 25 req/day)

- Endpoint: `https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=<symbol>`
- Already returns a primitive `overall_sentiment_score` per article
- Used as a **cross-check signal** rather than primary source

**API Key needed:** `ALPHA_VANTAGE_KEY` → sign up free at alphavantage.co

### 3.4 Corporate Disclosures — BSE India (Free, Public)

- BSE public API: `https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w`
- Query by BSE scrip code to get all recent announcements
- Key disclosure types to flag: Board Results, Pledging, Auditor Resignation, QIP

**No API key needed.** Public endpoint.

### 3.5 Social Sentiment — Reddit (Free)

- Subreddits: `r/IndianStockMarket`, `r/DalalStreet`, `r/Nifty`, `r/IndiaInvestments`
- Uses **PRAW library** (Python Reddit API Wrapper)
- Retrieves top 50 posts from the last 7 days sorted by relevance

**API Keys needed (free Reddit app — create at reddit.com/prefs/apps):**
```
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=SpiderSence/1.0
```

---

## 4. MCP Server Tools (`news_mcp`)

### `get_news(symbol, company_name, days=3) → list[Article]`
Aggregates articles from RSS feeds + NewsAPI. Deduplicates by title.

### `get_corporate_disclosures(symbol, days=30) → list[Disclosure]`
Fetches BSE corporate announcements for the symbol.

### `get_social_posts(symbol, company_name, limit=50) → list[Post]`
Searches Reddit for the most relevant recent posts.

### `get_alpha_vantage_sentiment(ticker, limit=20) → list[ScoredArticle]`
Returns pre-scored articles from Alpha Vantage as a cross-check.

---

## 5. LLM Analysis Pipeline (`sentiment_engine.py`)

Uses **Azure OpenAI GPT-5** (already in `.env`). Makes **3 sequential LLM calls**.

### Call 1 — Event Classification

**Input:** List of news headlines + disclosure subjects  
**Output:** JSON list of detected events

| Event Type | Example | Direction |
|---|---|---|
| `EARNINGS_BEAT` | "TCS Q1 profit beats by 8%" | positive |
| `EARNINGS_MISS` | "HDFC Bank misses NIM estimates" | negative |
| `REVENUE_SURPRISE_POSITIVE` | "Revenue down 2% vs expected 5% drop" | positive |
| `CONTRACT_WIN` | "Infosys wins $2B deal" | positive |
| `REGULATORY_ACTION` | "SEBI imposes ban on promoter" | negative |
| `MANAGEMENT_CHANGE_NEGATIVE` | "CFO resigns unexpectedly" | negative |
| `MANAGEMENT_CHANGE_POSITIVE` | "Industry veteran appointed CEO" | positive |
| `STAKE_PLEDGE_INCREASE` | "Promoter pledges 8% stake" | negative |
| `STAKE_PLEDGE_DECREASE` | "Promoter releases pledge on 5% shares" | positive |
| `DEBT_REDUCTION` | "Company repays ₹2000 Cr debt early" | positive |
| `QIP_RIGHTS_ISSUE` | "Company raises ₹5000 Cr via QIP" | negative (dilution) |
| `ANALYST_UPGRADE` | "Goldman raises target to ₹1500" | positive |
| `ANALYST_DOWNGRADE` | "Morgan Stanley cuts target" | negative |
| `FII_OUTFLOW` | "FIIs sell ₹4000 Cr in Banking" | negative |
| `MACRO_RISK` | "INR hits all-time low" | negative |

### Call 2 — Tone Analysis

**Input:** Full article body text from 3–5 most relevant articles  
**Output:**
```json
{
  "management_confidence": 4.2,
  "forward_guidance": "positive",
  "language_flags": ["growth_affirmations", "capex_commitment"],
  "tone_summary": "Management tone is assertive..."
}
```

**Confidence scale:** 1 = defensive/legal boilerplate, 5 = assertive with concrete guidance

**Positive flags:** `growth_affirmations`, `capex_commitment`, `buyback_signals`, `dividend_increase`  
**Negative flags:** `hedging_terms`, `legal_boilerplate`, `withdrawn_guidance`, `auditor_qualification`

### Call 3 — Final Synthesis

**Input:** Events list + tone scores + social vote scores + Alpha Vantage scores  
**Weighting model:**

```
News Events:          40%
Corporate Filings:    25%
Tone Analysis:        15%
Social Sentiment:     15%
Alpha Vantage:         5%
```

**Output:**
```json
{
  "score": 6.5,
  "label": "MODERATELY_BULLISH",
  "confidence": "medium",
  "key_catalyst": "Contract win worth ₹4,200 Cr with ONGC",
  "narrative": "Sentiment is moderately positive..."
}
```

**Score labels:**
| Range | Label |
|---|---|
| +7 to +10 | `STRONGLY_BULLISH` |
| +4 to +7 | `MODERATELY_BULLISH` |
| +1 to +4 | `SLIGHTLY_BULLISH` |
| -1 to +1 | `NEUTRAL` |
| -4 to -1 | `SLIGHTLY_BEARISH` |
| -7 to -4 | `MODERATELY_BEARISH` |
| -10 to -7 | `STRONGLY_BEARISH` |

---

## 6. Output JSON Schema

Saved to:
```
companies/<SYMBOL>/<run_ts>/sentiment/<SYMBOL>_sentiment.json
```

```json
{
  "symbol": "NSE:RELIANCE-EQ",
  "generated_at": "2026-08-02T07:30:00Z",
  "run_timestamp": "20260802_073000_UTC",

  "sentiment": {
    "score": 6.5,
    "score_range": [-10, 10],
    "label": "MODERATELY_BULLISH",
    "confidence": "medium",
    "key_catalyst": "...",
    "narrative": "..."
  },

  "breakdown": {
    "news_score": 7.2,
    "social_score": 5.8,
    "filings_score": 6.0,
    "tone_score": 4.5,
    "alpha_vantage_score": 6.0
  },

  "events_detected": [
    {
      "type": "CONTRACT_WIN",
      "headline": "Reliance wins ₹4,200 Cr LNG deal with ONGC",
      "source": "Economic Times",
      "published_at": "2026-08-01",
      "impact": "positive",
      "weight": 0.8
    }
  ],

  "tone_analysis": {
    "management_confidence": 4.2,
    "forward_guidance": "positive",
    "language_flags": ["growth_affirmations", "capex_commitment"],
    "tone_summary": "..."
  },

  "social_summary": {
    "posts_analyzed": 23,
    "avg_vote_score": 142,
    "community_bias": "bullish",
    "top_post": "..."
  },

  "sources_used": {
    "news_articles": 18,
    "corporate_filings": 3,
    "social_posts": 23,
    "alpha_vantage_articles": 10
  }
}
```

---

## 7. CLI Usage

```bash
# Basic run
python -m agents.sentiment_agent.sentiment_agent NSE:RELIANCE-EQ

# With company name hint (improves news search accuracy)
python -m agents.sentiment_agent.sentiment_agent NSE:RELIANCE-EQ --company "Reliance Industries"

# Look back 7 days of news
python -m agents.sentiment_agent.sentiment_agent NSE:RELIANCE-EQ --days 7
```

---

## 8. New `.env` Variables Needed

```env
# NewsAPI.org — free at newsapi.org
NEWS_API_KEY=

# Alpha Vantage — free at alphavantage.co
ALPHA_VANTAGE_KEY=

# Reddit API — free, create app at reddit.com/prefs/apps
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=SpiderSence/1.0
```

---

## 9. New Dependencies

```
praw>=7.7.0          # Reddit API
feedparser>=6.0.0    # RSS feed parsing
openai>=1.40.0       # Azure OpenAI client
beautifulsoup4>=4.12 # HTML parsing for BSE scraper
```

---

## Review Checklist (confirm before coding starts)

- [ ] Data sources look correct?
- [ ] Weighting model (40% news / 25% filings / 15% tone / 15% social / 5% AV) agreed?
- [ ] Score range (-10 to +10) agreed?
- [ ] Output JSON structure looks good?
- [ ] Reddit API keys ready? (`reddit.com/prefs/apps`)
- [ ] NewsAPI key ready? (`newsapi.org`)
- [ ] Alpha Vantage key ready? (`alphavantage.co`)
