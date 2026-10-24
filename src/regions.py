"""Borough boundaries and the precomputed per-borough statistics the map colors by.

``build_region_stats`` runs inside training and writes ``config/region_stats.csv``
with one row per borough polygon, so the choropleth needs no live model calls.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config

STATS_COLUMNS = [
    "borough",
    "median_price",
    "predicted_price",
    "yoy_pct",
    "vs_median_pct",
]


def load_geojson(path: Path = config.BOROUGHS_GEOJSON) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def borough_names(geojson: dict[str, Any]) -> list[str]:
    """Borough names from the GeoJSON ``properties`` (``borough`` or ``name``)."""
    names = []
    for feature in geojson["features"]:
        props = feature.get("properties", {})
        names.append(props.get("borough") or props.get("name"))
    return names


def build_region_stats(
    features: pd.DataFrame,
    geojson: dict[str, Any],
    reference_predictor: Callable[[str], float],
    recent_year: int = 2026,
    prior_year: int = 2025,
) -> pd.DataFrame:
    """One row per GeoJSON borough; gaps fall back to London-wide values."""
    recent = features[features["year"] == recent_year]
    prior = features[features["year"] == prior_year]
    london_median = float(recent["price"].median()) if not recent.empty else float("nan")

    med_recent = recent.groupby("borough")["price"].median()
    med_prior = prior.groupby("borough")["price"].median()

    rows = []
    for name in borough_names(geojson):
        median_price = float(med_recent.get(name, london_median))
        prior_price = med_prior.get(name, np.nan)
        yoy = (
            float((median_price - prior_price) / prior_price * 100.0)
            if pd.notna(prior_price) and prior_price > 0
            else 0.0
        )
        vs_median = (
            float((median_price - london_median) / london_median * 100.0)
            if london_median and not np.isnan(london_median)
            else 0.0
        )
        rows.append(
            {
                "borough": name,
                "median_price": round(median_price),
                "predicted_price": round(reference_predictor(name)),
                "yoy_pct": round(yoy, 2),
                "vs_median_pct": round(vs_median, 2),
            }
        )
    return pd.DataFrame(rows, columns=STATS_COLUMNS)


def compute_bins(values: list[float] | np.ndarray, k: int = 5) -> list[float]:
    """Monotonic bin edges (quantile breaks) of length ``k + 1`` for a legend."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return list(np.linspace(0.0, 1.0, k + 1))
    edges = np.quantile(arr, np.linspace(0.0, 1.0, k + 1))
    edges = np.unique(np.round(edges, 6))
    if edges.size < 2:
        edges = np.array([arr.min(), arr.max() + 1.0])
    return [float(e) for e in edges]


def load_stats(path: Path = config.REGION_STATS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


# ONS Open Geography Portal — generalised, clipped Local Authority Districts
# (Open Government Licence). Returns GeoJSON directly, so no shapefile reader
# is needed, and the layer is already generalised (the "simplify" step).
BOUNDARIES_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Local_Authority_Districts_May_2023_UK_BGC_V2/FeatureServer/0/query"
)
# Envelope around Greater London as west,south,east,north (lon/lat).
LONDON_ENVELOPE = "-0.52,51.28,0.34,51.70"


def _build_boundaries(output: Path | None = None) -> None:
    """Fetch the 33 London borough boundaries as a small committed GeoJSON."""
    import urllib.parse
    import urllib.request

    params = {
        "where": "1=1",
        "geometry": LONDON_ENVELOPE,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "f": "geojson",
        "resultRecordCount": "2000",
    }
    url = BOUNDARIES_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310  (trusted ONS host)
        collection = json.load(resp)

    wanted = set(config.BOROUGHS)

    def borough_of(props: dict[str, Any]) -> str | None:
        return next((v for v in props.values() if isinstance(v, str) and v in wanted), None)

    collected: list[tuple[str, Any]] = []
    for feature in collection.get("features", []):
        name = borough_of(feature.get("properties", {}))
        if name:
            collected.append((name, feature["geometry"]))
    collected.sort(key=lambda item: item[0])

    missing = wanted - {name for name, _ in collected}
    if missing:
        raise RuntimeError(f"No boundary returned for boroughs: {sorted(missing)}")

    features = [
        {"type": "Feature", "properties": {"borough": name}, "geometry": geometry}
        for name, geometry in collected
    ]
    dest = Path(output) if output else config.BOROUGHS_GEOJSON
    with dest.open("w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh)
    print(f"Wrote {len(features)} borough polygons -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Borough boundaries and stats.")
    parser.add_argument("--build-boundaries", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.build_boundaries:
        _build_boundaries(Path(args.output) if args.output else None)
    else:
        print(load_stats().head())


if __name__ == "__main__":
    main()
