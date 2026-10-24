"""Confirm the 16-column PPD layout and that the London-flat filter is sane."""

from __future__ import annotations

import sys

import pandas as pd

from src import config
from src.download_data import filter_london_flats, load_london_flats, load_raw_csv


def check_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with the raw column layout (empty == ok)."""
    problems: list[str] = []
    if list(df.columns) != config.PPD_COLUMNS:
        problems.append(
            f"expected {len(config.PPD_COLUMNS)} PPD columns {config.PPD_COLUMNS}, "
            f"got {list(df.columns)}"
        )
    return problems


def check_filter(raw: pd.DataFrame) -> list[str]:
    """Return problems with the filtered frame (empty == ok)."""
    problems: list[str] = []
    london = filter_london_flats(raw)
    if london.empty:
        problems.append("London-flat filter produced zero rows")
        return problems
    if not (london["property_type"].str.upper() == config.PROPERTY_TYPE_FLAT).all():
        problems.append("non-flat rows survived the filter")
    if not (london["county"].str.upper() == config.COUNTY_GREATER_LONDON).all():
        problems.append("non-Greater-London rows survived the filter")
    return problems


def main() -> int:
    raw = load_raw_csv(config.SAMPLE_CSV) if config.SAMPLE_CSV.exists() else None
    if raw is None:
        print("No sample CSV to check; run `make data` first.")
        return 1
    problems = check_columns(raw) + check_filter(raw)
    if problems:
        for p in problems:
            print(f"✗ {p}")
        return 1
    london = load_london_flats(use_sample=True)
    print(f"✓ 16 PPD columns confirmed; London-flat filter keeps {len(london):,} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
