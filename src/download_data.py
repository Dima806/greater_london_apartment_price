"""Fetch HM Land Registry Price Paid Data and filter it to Greater London flats.

Downloads are keyless. For offline Codespace runs and CI, a frozen sample in
``data/sample/`` stands in for the full download, so nothing here needs the
network at test time.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from src import config


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Read a headerless PPD CSV into a frame with the 16 named columns."""
    return pd.read_csv(path, header=None, names=config.PPD_COLUMNS, dtype=str)


def filter_london_flats(
    df: pd.DataFrame,
    min_price: int = config.MIN_PRICE,
    max_price: int = config.MAX_PRICE,
) -> pd.DataFrame:
    """Keep standard-category flat sales in Greater London within price bounds."""
    price = pd.to_numeric(df["price"], errors="coerce")
    mask = (
        (df["property_type"].str.upper() == config.PROPERTY_TYPE_FLAT)
        & (df["county"].str.upper() == config.COUNTY_GREATER_LONDON)
        & (df["ppd_category_type"].str.upper() == config.PPD_CATEGORY_STANDARD)
        & price.between(min_price, max_price)
    )
    out = df.loc[mask].copy()
    out["price"] = price.loc[mask].astype(int)
    return out.reset_index(drop=True)


def download_year(year: int, dest_dir: Path = config.RAW_DIR) -> Path:
    """Download one year's PPD CSV to ``dest_dir`` (cached; skips if present)."""
    url = config.PPD_URLS[year]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"pp-{year}.csv"
    if dest.exists():
        return dest
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310  (trusted gov.uk host)
    return dest


def load_london_flats(use_sample: bool = False) -> pd.DataFrame:
    """Return filtered Greater London flats, from the raw download or the sample."""
    if use_sample or not config.RAW_DIR.exists() or not any(config.RAW_DIR.glob("pp-*.csv")):
        return filter_london_flats(load_raw_csv(config.SAMPLE_CSV))
    frames = [load_raw_csv(p) for p in sorted(config.RAW_DIR.glob("pp-*.csv"))]
    return filter_london_flats(pd.concat(frames, ignore_index=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + filter Land Registry PPD.")
    parser.add_argument("--years", nargs="+", type=int, default=config.YEARS)
    args = parser.parse_args()

    frames = []
    for year in args.years:
        path = download_year(year)
        frames.append(load_raw_csv(path))
    london = filter_london_flats(pd.concat(frames, ignore_index=True))

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / "london_flats.csv"
    london.to_csv(out, index=False)
    print(f"Wrote {len(london):,} Greater London flat sales -> {out}")


if __name__ == "__main__":
    main()
