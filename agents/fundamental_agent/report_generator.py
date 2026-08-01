#!/usr/bin/env python3
"""Fundamental Agent Report Generator.

Takes the derived JSON from the screener enricher, builds a prompt using
the prompt template, sends it to the LLM via the orchestrator's LLM client,
and saves the generated Markdown report.

Usage:
  python3 -m agents.fundamental_agent.report_generator RELIANCE
  spider-fundamental-report RELIANCE
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agents.fundamental_agent.prompt_template import build_prompts
from orchestrator.llm_client import LLMClient
from orchestrator.llm_config import LLMConfig, load_config
from orchestrator.output_paths import get_category_directory, get_companies_root, normalise_symbol

logger = logging.getLogger(__name__)

DEFAULT_EXPORTS_DIR = Path(__file__).resolve().parents[2] / "companies"


def _resolve_fundamental_dir(
    symbol: str,
    exports_dir: Path = DEFAULT_EXPORTS_DIR,
    run_timestamp: Optional[str] = None,
) -> tuple[Path, str]:
    root = get_companies_root(exports_dir)
    return get_category_directory(symbol, "fundamental", root=root, run_timestamp=run_timestamp)


def _load_derived_json_from_dir(symbol: str, fundamental_dir: Path) -> Dict[str, Any]:
    safe_symbol = normalise_symbol(symbol)
    derived_path = fundamental_dir / f"{safe_symbol}_derived.json"
    if not derived_path.exists():
        raise FileNotFoundError(
            f"Derived JSON not found: {derived_path}\n"
            f"Run the pipeline first: spider-fundamental {symbol} --skip-fetch"
        )
    with derived_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_derived_json(
    symbol: str,
    exports_dir: Path = DEFAULT_EXPORTS_DIR,
    run_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the derived JSON for a given symbol.

    Args:
        symbol: Stock symbol (e.g. RELIANCE).
        exports_dir: Directory containing the exports.
        run_timestamp: Optional run timestamp folder to load; defaults to latest run.

    Returns:
        The derived JSON as a dict.

    Raises:
        FileNotFoundError: If the derived JSON does not exist.
    """
    symbol = symbol.strip().upper()
    fundamental_dir, _ = _resolve_fundamental_dir(symbol, exports_dir, run_timestamp)
    return _load_derived_json_from_dir(symbol, fundamental_dir)


def generate_report(
    derived_data: Dict[str, Any],
    config: Optional[LLMConfig] = None,
) -> str:
    """Generate a fundamental analysis report using the LLM.

    Args:
        derived_data: The derived JSON dict from compute_derivatives.
        config: Optional LLM configuration override.

    Returns:
        The generated report as a Markdown string.
    """
    system_prompt, user_prompt = build_prompts(derived_data)

    logger.info("Building LLM client ...")
    client = LLMClient(config=config)

    logger.info("Sending prompt to LLM ...")
    report_text = client.complete(system_prompt, user_prompt)

    logger.info("Report generated (%d chars)", len(report_text))
    return report_text


def _save_report_to_dir(symbol: str, report_text: str, fundamental_dir: Path) -> Path:
    fundamental_dir.mkdir(parents=True, exist_ok=True)
    report_path = fundamental_dir / f"{normalise_symbol(symbol)}_fundamental_report.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(report_text)
    return report_path


def save_report(
    symbol: str,
    report_text: str,
    exports_dir: Path = DEFAULT_EXPORTS_DIR,
    run_timestamp: Optional[str] = None,
) -> Path:
    """Save the report as a Markdown file.

    Args:
        symbol: Stock symbol.
        report_text: The report content.
        exports_dir: Directory containing the exports.
        run_timestamp: Optional run timestamp folder; defaults to latest run.

    Returns:
        Path to the saved report file.
    """
    symbol = symbol.strip().upper()
    fundamental_dir, _ = _resolve_fundamental_dir(symbol, exports_dir, run_timestamp)
    return _save_report_to_dir(symbol, report_text, fundamental_dir)


def run_report_generation(
    symbol: str,
    *,
    exports_dir: Path = DEFAULT_EXPORTS_DIR,
    config_path: Optional[Path] = None,
    run_timestamp: Optional[str] = None,
) -> Path:
    """Full report generation pipeline: load derived JSON → LLM → save report.

    Args:
        symbol: Stock symbol (e.g. RELIANCE).
        exports_dir: Directory containing the exports.
        config_path: Optional path to an LLM config JSON file.

    Returns:
        Path to the saved report file.
    """
    symbol = symbol.strip().upper()

    # Step 1: Resolve run folder & load derived JSON
    logger.info("Resolving run folder for %s ...", symbol)
    fundamental_dir, resolved_timestamp = _resolve_fundamental_dir(symbol, exports_dir, run_timestamp)

    logger.info("Loading derived JSON for %s ...", symbol)
    derived_data = _load_derived_json_from_dir(symbol, fundamental_dir)

    # Step 2: Load LLM config
    config = load_config(config_path) if config_path else load_config()
    if not config.is_valid():
        logger.error(
            "LLM configuration is invalid. Please configure your API key:\n"
            "  Option A: Set env var  export LLM_API_KEY=sk-...\n"
            "  Option B: Create file  orchestrator/llm_config.json\n"
            "  Option C: Pass --config path/to/config.json\n"
            "See README.md for full setup instructions."
        )
        raise SystemExit(2)

    logger.info("Using LLM config: %s", config)

    # Step 3: Generate report
    report_text = generate_report(derived_data, config=config)

    # Step 4: Save report
    report_path = _save_report_to_dir(symbol, report_text, fundamental_dir)
    logger.info("Report saved: %s", report_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"  FUNDAMENTAL REPORT GENERATED — {symbol}")
    print("=" * 60)
    print(f"  Report file: {report_path}")
    print(f"  Report size: {len(report_text)} chars")
    print(f"  Run folder: {fundamental_dir.parent}")
    print(f"  Run timestamp: {resolved_timestamp}")
    print("=" * 60 + "\n")

    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a fundamental analysis report using an LLM."
    )
    parser.add_argument("symbol", help="Stock symbol (e.g. RELIANCE)")
    parser.add_argument(
        "--exports-dir",
        default=str(DEFAULT_EXPORTS_DIR),
        help="Root directory containing company data (default: ./companies)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to LLM config JSON file (overrides env vars)",
    )
    parser.add_argument(
        "--run-timestamp",
        type=str,
        default=None,
        help="Run timestamp folder to use (default: latest)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run_report_generation(
        args.symbol,
        exports_dir=Path(args.exports_dir),
        config_path=Path(args.config) if args.config else None,
        run_timestamp=args.run_timestamp,
    )


if __name__ == "__main__":
    main()
