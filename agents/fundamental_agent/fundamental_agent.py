#!/usr/bin/env python3
"""Fundamental Agent Pipeline.

Runs the full fundamental analysis pipeline for a given stock symbol:
  1. Fetch company data from screener.in and export to Excel + JSONL
  2. Compute derived fundamentals from the JSONL and save the derived JSON

Usage:
  python3 -m agents.fundamental_agent.pipeline RELIANCE
  python3 -m agents.fundamental_agent.pipeline RELIANCE --skip-fetch
  python3 -m agents.fundamental_agent.pipeline RELIANCE --consolidated

After installing the project (pip install -e .), you can also run:
  spider-fundamental RELIANCE
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcp_servers.screener_mcp.screener_output import export_company, write_jsonl_from_excel
from mcp_servers.screener_mcp.screener_enricher import load_jsonl, build_lookup, compute_derivatives, save_result
from orchestrator.output_paths import (
    ensure_category_directory,
    get_category_directory,
    get_companies_root,
    normalise_symbol,
)

logger = logging.getLogger(__name__)


def _try_generate_report(
    symbol: str,
    companies_root: Path,
    config_path: Optional[Path],
    run_timestamp: Optional[str],
) -> Optional[Path]:
    """Attempt to generate an LLM report; return path or None on failure."""
    try:
        from agents.fundamental_agent.report_generator import run_report_generation
        return run_report_generation(
            symbol,
            exports_dir=companies_root,
            config_path=config_path,
            run_timestamp=run_timestamp,
        )
    except SystemExit:
        # Config validation failed — skip gracefully
        return None
    except Exception as exc:
        logger.warning("Report generation skipped: %s", exc)
        return None

DEFAULT_EXPORTS_DIR = Path(__file__).resolve().parents[2] / "companies"


def run_pipeline(
    symbol: str,
    *,
    exports_dir: Path = DEFAULT_EXPORTS_DIR,
    skip_fetch: bool = False,
    consolidated: bool = False,
    generate_report: bool = False,
    config_path: Optional[Path] = None,
    run_timestamp: Optional[str] = None,
    use_screener_mcp: bool = False,
    screener_mcp_url: Optional[str] = None,
) -> Path:
    """Run the full fundamental agent pipeline.

    Args:
        symbol: Stock symbol (e.g. RELIANCE).
        exports_dir: Directory where exports are stored.
        skip_fetch: If True, skip the screener.in fetch step and only enrich
            the existing JSONL.
        consolidated: If True, fetch the consolidated company page.
        generate_report: If True, generate an LLM fundamental report after
            computing derivatives.
        config_path: Optional path to an LLM config JSON file for report
            generation.

    Returns:
        Path to the saved derived JSON file.
    """
    symbol = symbol.strip().upper()
    normalised_symbol = normalise_symbol(symbol)
    companies_root = get_companies_root(exports_dir)

    run_time = datetime.now(timezone.utc)

    if skip_fetch:
        try:
            fundamental_dir, resolved_timestamp = get_category_directory(
                symbol,
                "fundamental",
                root=companies_root,
                run_timestamp=run_timestamp,
            )
        except FileNotFoundError:
            logger.error(
                "Skipping fetch but no existing data found for %s in %s",
                symbol,
                companies_root / normalised_symbol,
            )
            raise SystemExit(2)
    else:
        fundamental_dir, resolved_timestamp = ensure_category_directory(
            symbol,
            "fundamental",
            root=companies_root,
            run_datetime=run_time,
            run_timestamp=run_timestamp,
        )

    # ------------------------------------------------------------------
    # Stage 1: Fetch from screener.in -> Excel + JSONL
    # ------------------------------------------------------------------
    excel_path = fundamental_dir / f"{normalised_symbol}.xlsx"
    jsonl_path = fundamental_dir / f"{normalised_symbol.lower()}.jsonl"

    if skip_fetch:
        if not jsonl_path.exists() and not excel_path.exists():
            logger.error(
                "Skipping fetch but no existing data found for %s in %s",
                symbol,
                fundamental_dir,
            )
            raise SystemExit(2)
        logger.info("Skipping fetch step; using existing data for %s", symbol)
    else:
        logger.info("Stage 1: Fetching %s from screener.in ...", symbol)
        t0 = time.time()
        try:
            if use_screener_mcp:
                from fastmcp import Client

                url = screener_mcp_url or os.getenv(
                    "SCREENER_MCP_URL", "http://127.0.0.1:8000/mcp/screener"
                )
                async def _call():
                    async with Client(url) as client:
                        res = await client.call_tool(
                            "fetch_company_details",
                            arguments={
                                "symbol": symbol,
                                "consolidated": consolidated,
                                "exports_dir": str(exports_dir),
                                "run_timestamp": resolved_timestamp,
                            },
                        )
                        return res.data
                import asyncio

                payload = asyncio.run(_call())
                excel_path = Path(payload["excel_path"])
                jsonl_path = Path(payload["jsonl_path"])
                logger.info("  Screener MCP Excel saved: %s", excel_path)
            else:
                excel_path = export_company(symbol, fundamental_dir)
                logger.info("  Excel saved: %s", excel_path)
        except Exception as exc:
            logger.error("Failed to fetch company data for %s: %s", symbol, exc)
            raise

        # Generate JSONL from the Excel if it wasn't already created
        if not jsonl_path.exists():
            try:
                jsonl_path = write_jsonl_from_excel(excel_path)
                logger.info("  JSONL saved: %s", jsonl_path)
            except Exception as exc:
                logger.error("Failed to write JSONL: %s", exc)
                raise

        elapsed = time.time() - t0
        logger.info("Stage 1 complete (%.1fs)", elapsed)

    # ------------------------------------------------------------------
    # Stage 2: Enrich JSONL -> derived JSON
    # ------------------------------------------------------------------
    logger.info("Stage 2: Computing derived fundamentals for %s ...", symbol)

    # locate JSONL
    jsonl1 = fundamental_dir / f"{normalised_symbol.lower()}.jsonl"
    jsonl2 = fundamental_dir / f"{normalised_symbol}.jsonl"
    if jsonl1.exists():
        jpath = jsonl1
    elif jsonl2.exists():
        jpath = jsonl2
    else:
        logger.error("JSONL file not found for %s in %s", symbol, fundamental_dir)
        raise SystemExit(2)

    items = load_jsonl(jpath)
    logger.info("  Loaded %d items from %s", len(items), jpath.name)

    lookup = build_lookup(items)
    result = compute_derivatives(lookup, symbol)
    outp = save_result(symbol, result, fundamental_dir)
    logger.info("  Derived JSON saved: %s", outp)
    logger.info("Stage 2 complete.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    deriv = result.get("calculated_fundamental_derivatives", {})
    raw = result.get("raw_baseline_inputs", {})
    qual = result.get("extracted_qualitative_context", {})

    print("\n" + "=" * 60)
    print(f"  FUNDAMENTAL AGENT PIPELINE — {symbol}")
    print("=" * 60)
    print(f"  Data as of: {result.get('data_as_of')}")
    print(f"  Run timestamp: {resolved_timestamp}")
    print("-" * 60)
    print("  Raw Baseline Inputs:")
    for k, v in raw.items():
        print(f"    {k:<35} {v}")
    print("-" * 60)
    print("  Calculated Derivatives:")
    for section, values in deriv.items():
        print(f"    [{section}]")
        for k, v in values.items():
            print(f"      {k:<33} {v}")
    print("-" * 60)
    print("  Qualitative Context:")
    print(f"    Pros: {qual.get('verified_pros', [])}")
    print(f"    Cons: {qual.get('verified_cons', [])}")
    print("=" * 60)
    print(f"  Output: {outp}")
    print(f"  Run folder: {fundamental_dir.parent}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Stage 3 (optional): Generate LLM report
    # ------------------------------------------------------------------
    report_path = None
    if generate_report:
        logger.info("Stage 3: Generating LLM fundamental report for %s ...", symbol)
        report_path = _try_generate_report(symbol, companies_root, config_path, resolved_timestamp)
        if report_path:
            logger.info("Stage 3 complete.")
        else:
            logger.info("Stage 3 skipped (no LLM config or error).")

    payload = {
        "derived_json": str(outp),
        "run_folder": str(fundamental_dir.parent),
        "run_timestamp": resolved_timestamp,
    }
    if report_path:
        payload["report_path"] = str(report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full fundamental agent pipeline (fetch + enrich)."
    )
    parser.add_argument("symbol", help="Stock symbol (e.g. RELIANCE)")
    parser.add_argument(
        "--exports-dir",
        default=str(DEFAULT_EXPORTS_DIR),
        help="Root directory to store company data (default: ./companies)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip the screener.in fetch step; only enrich existing JSONL",
    )
    parser.add_argument(
        "--consolidated",
        action="store_true",
        help="Fetch the consolidated company page from screener.in",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate an LLM fundamental analysis report after enrichment",
    )
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help="Use an existing run timestamp (default: create new run)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to LLM config JSON file (for --report)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(
            args.symbol,
            exports_dir=Path(args.exports_dir),
            skip_fetch=args.skip_fetch,
            consolidated=args.consolidated,
            generate_report=args.report,
            config_path=Path(args.config) if args.config else None,
            run_timestamp=args.run_timestamp,
        )


if __name__ == "__main__":
    main()
