# spider-sence

A Investing/Trading MultiAgent system that fetches and analyzes stock fundamentals from [screener.in](https://www.screener.in).

## Installation

```bash
# Clone and install in editable mode
git clone git@github.com:bignya23/spider-sence.git
cd spider-sence
pip install -e .
```

This registers five CLI commands (see below).

## LLM Configuration (for Report Generation)

The report generation feature uses an OpenAI-compatible API to generate
fundamental analysis reports. You can use **OpenRouter**, **OpenAI**, or any
compatible provider.

### Option A: Environment Variables

```bash
export LLM_API_KEY="sk-or-v1-your-key-here"
export LLM_BASE_URL="https://openrouter.ai/api/v1"   # optional, this is the default
export LLM_MODEL="cline-free/glm-5.2"                 # optional, this is the default (free)
```

### Option B: Config File

Copy the example config and add your API key:

```bash
cp orchestrator/llm_config.example.json orchestrator/llm_config.json
# Edit orchestrator/llm_config.json and replace the api_key
```

> **Note:** `orchestrator/llm_config.json` is in `.gitignore` and will never
> be committed. Only `llm_config.example.json` is tracked.

### Option C: Pass Config Path

```bash
spider-fundamental RELIANCE --skip-fetch --report --config /path/to/config.json
```

## Fundamental Agent

The fundamental agent lives in `agents/fundamental_agent/` and runs a two-stage pipeline:

1. **Fetch** (`screener_output.py`) — Downloads company data from screener.in, parses the HTML, and exports to Excel (`.xlsx`) + JSONL (`.jsonl`).
2. **Enrich** (`screener_enricher.py`) — Reads the JSONL, computes derived fundamental metrics (CAGR, PEG, ROCE, ROE, cash conversion, institutional flows, etc.), and saves a derived JSON payload for downstream LLM analysis.

### Pipeline (`pipeline.py`)

Orchestrates both stages in a single command and prints a summary report.

### CLI Commands

After `pip install -e .`, the following commands are available:

| Command | Description |
|---|---|
| `spider-fundamental SYMBOL` | Run the full pipeline (fetch + enrich + optional report) |
| `spider-fundamental-report SYMBOL` | Generate an LLM fundamental analysis report |
| `spider-screener-fetch SYMBOL` | Fetch company data from screener.in → Excel + JSONL |
| `spider-screener-enrich SYMBOL` | Enrich existing JSONL → derived JSON |
| `spider-mcp` | Start MCP server for Cline integration |

#### Examples

```bash
# Full pipeline: fetch from screener.in + compute derivatives
spider-fundamental RELIANCE

# Skip the network fetch; only enrich existing JSONL data
spider-fundamental RELIANCE --skip-fetch

# Fetch consolidated company page
spider-fundamental RELIANCE --consolidated

# Full pipeline + generate LLM report
spider-fundamental RELIANCE --skip-fetch --report

# Generate only the LLM report (requires existing derived JSON)
spider-fundamental-report RELIANCE

# Run individual stages
spider-screener-fetch RELIANCE
spider-screener-enrich RELIANCE
```

You can also run via Python module:

```bash
python3 -m agents.fundamental_agent.pipeline RELIANCE --skip-fetch --report
python3 -m agents.fundamental_agent.report_generator RELIANCE
python3 -m agents.fundamental_agent.screener_output RELIANCE
python3 -m agents.fundamental_agent.screener_enricher RELIANCE
```

### Output

Each run writes to `companies/<SYMBOL>/<RUN_TIMESTAMP>/`, where the timestamp is
formatted as `YYYYMMDD_HHMMSS_UTC`.

- `fundamental/`
  - `<SYMBOL>.xlsx` — Raw screener.in export in Excel format.
  - `<symbol>.jsonl` — One JSON object per metric/period (for embeddings or search).
  - `<SYMBOL>_derived.json` — Computed fundamentals plus raw baseline inputs.
  - `<SYMBOL>_fundamental_report.md` — LLM-generated Markdown report (when enabled).
- `technical/`
  - `<SYMBOL>_candles.png` — Candlestick chart rendered via Matplotlib.
  - `<SYMBOL>_interactive.html` — Plotly-based interactive chart (when generated).

The derived JSON contains:

- **`raw_baseline_inputs`** — Absolute data points (price, market cap, borrowings, reserves, cash flows, etc.)
- **`calculated_fundamental_derivatives`** — Computed metrics across 5 pillars:
  - Growth Velocity (CAGR)
  - Valuation Realism (PE, PEG, dividend yield)
  - Solvency & Leverage (debt-to-reserves, ROCE, ROE)
  - Liquidity & Earnings Quality (cash conversion, FCF/CFO)
  - Smart Money Momentum (promoter/FII/DII deltas)
- **`extracted_qualitative_context`** — Verified pros & cons from screener.in
- **`field_definitions`** — One-line descriptions of every field for the downstream LLM agent

### Testing

```bash
python3 -m pytest tests/test_screener_cleaner.py -v
```

## Project Structure

```
spider-sence/
├── agents/
│   └── fundamental_agent/
│       ├── screener_output.py     # Stage 1: Fetch + Excel/JSONL export
│       ├── screener_enricher.py   # Stage 2: Compute derived fundamentals
│       ├── prompt_template.py     # LLM prompt templates for report generation
│       ├── report_generator.py    # Stage 3: LLM report generation
│       ├── pipeline.py            # Orchestrates all stages
│       └── exports/               # (legacy) retained for compatibility
├── companies/                     # Runtime outputs grouped by symbol/run
├── orchestrator/
│   ├── __init__.py
│   ├── llm_config.py              # LLM configuration (env vars / JSON file)
│   ├── llm_client.py              # OpenAI-compatible API client
│   └── llm_config.example.json    # Example config (copy to llm_config.json)
├── tests/
├── pyproject.toml
└── requirements.txt
```

## MCP Server (Cline Integration)

The MCP server exposes the pipeline as tools that Cline (or any MCP-compatible
client) can call directly — no need to run CLI commands manually.

### Setup

1. Install the package: `pip install -e .`

2. Add to your Cline MCP settings (`cline_mcp_settings.json`):

```json
{
  "mcpServers": {
    "spider-sence": {
      "command": "python3",
      "args": ["-m", "mcp_servers.screener_mcp.server"],
      "cwd": "/Users/bignyagogoi/Desktop/spider-sence"
    }
  }
}
```

3. Restart Cline. You can now ask Cline things like:
   - *"Analyze RELIANCE fundamentally"*
   - *"Fetch TCS data and generate a report"*
   - *"Show me the derived fundamentals for INFY"*

### Available MCP Tools

| Tool | Description |
|---|---|
| `run_pipeline` | Fetch & enrich Screener data, optionally generate report |
| `compute_derivatives` | Compute derived fundamentals from existing JSONL |
| `generate_report` | Generate an LLM fundamental analysis report |

### Cline-Based Report Generation (No API Key Needed)

The `cline-free/glm-5.2` model is only available through the Cline extension,
not via external API. To generate reports using Cline's free model:

1. Cline calls `enrich_fundamentals` (or `fetch_company` first if data doesn't exist)
2. Cline calls `get_analysis_prompt` to get the system prompt + derived data
3. **Cline generates the report itself** using its own LLM (e.g. `cline-free/glm-5.2`)
4. Cline calls `save_report` to save the generated Markdown report

This means **no API key is needed** — Cline IS the LLM.

## Dependencies

- `requests` — HTTP client for screener.in
- `beautifulsoup4` — HTML parsing
- `pandas` — Data manipulation
- `openpyxl` — Excel export
- `lxml` — XML/HTML parser backend
- `openai` — LLM API client (for report generation)
- `mcp` — Model Context Protocol server (for Cline integration)
