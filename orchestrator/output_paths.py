from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple


_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_UTC"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_companies_root(root: Path | str | None = None) -> Path:
    base = Path(root) if root is not None else _project_root() / "companies"
    base.mkdir(parents=True, exist_ok=True)
    return base


def normalise_symbol(symbol: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9._-]+", "_", symbol.strip().upper())
    return cleaned or "UNKNOWN"


def format_run_timestamp(run_datetime: datetime | None = None) -> str:
    dt = (run_datetime or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return dt.strftime(_TIMESTAMP_FORMAT)


def ensure_run_directory(
    symbol: str,
    *,
    root: Path | str | None = None,
    run_datetime: datetime | None = None,
    run_timestamp: str | None = None,
) -> Tuple[Path, str]:
    base = get_companies_root(root)
    symbol_folder = base / normalise_symbol(symbol)
    symbol_folder.mkdir(parents=True, exist_ok=True)

    timestamp = run_timestamp or format_run_timestamp(run_datetime)
    run_dir = symbol_folder / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, timestamp


def list_run_directories(symbol: str, *, root: Path | str | None = None) -> Iterable[Path]:
    symbol_folder = get_companies_root(root) / normalise_symbol(symbol)
    if not symbol_folder.exists():
        return []
    return sorted(p for p in symbol_folder.iterdir() if p.is_dir())


def find_run_directory(
    symbol: str,
    *,
    root: Path | str | None = None,
    run_timestamp: str | None = None,
) -> Path | None:
    symbol_folder = get_companies_root(root) / normalise_symbol(symbol)
    if not symbol_folder.exists():
        return None

    if run_timestamp:
        candidate = symbol_folder / run_timestamp
        return candidate if candidate.exists() else None

    runs = list(list_run_directories(symbol, root=root))
    return runs[-1] if runs else None


def ensure_category_directory(
    symbol: str,
    category: str,
    *,
    root: Path | str | None = None,
    run_datetime: datetime | None = None,
    run_timestamp: str | None = None,
) -> Tuple[Path, str]:
    run_dir, timestamp = ensure_run_directory(
        symbol,
        root=root,
        run_datetime=run_datetime,
        run_timestamp=run_timestamp,
    )
    category_dir = run_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    return category_dir, timestamp


def get_category_directory(
    symbol: str,
    category: str,
    *,
    root: Path | str | None = None,
    run_timestamp: str | None = None,
) -> Tuple[Path, str]:
    run_dir = find_run_directory(symbol, root=root, run_timestamp=run_timestamp)
    if run_dir is None:
        raise FileNotFoundError(
            f"No run data found for {symbol.strip().upper()} in companies directory"
        )

    category_dir = run_dir / category
    if not category_dir.exists():
        raise FileNotFoundError(
            f"Category '{category}' missing for run {run_dir.name} (symbol {symbol.strip().upper()})"
        )
    return category_dir, run_dir.name

