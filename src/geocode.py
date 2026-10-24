"""Postcode-district geocoding, reusing the keyless crow-flies stack.

Centroids are precomputed and committed to ``config/postcode_centroids.csv``
so the Space makes no live geocoding calls. Crow-flies distances still use
geopy (geodesic). ``--build`` refreshes the centroids from the keyless
postcodes.io outcode API, falling back to the borough's boundary centroid for
any outward code it cannot resolve, so the builder never fails on one lookup.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from geopy.distance import geodesic

from src import config

CENTROID_COLUMNS = ["district", "lat", "lon", "borough"]


@functools.lru_cache(maxsize=1)
def load_centroids() -> pd.DataFrame:
    """Load the committed postcode-district centroid lookup."""
    df = pd.read_csv(config.CENTROIDS_CSV)
    return df[CENTROID_COLUMNS]


def centroid(district: str) -> tuple[float, float] | None:
    """Return ``(lat, lon)`` for a postcode district, or ``None`` if unknown."""
    df = load_centroids()
    hit = df.loc[df["district"] == str(district).upper()]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return float(row["lat"]), float(row["lon"])


def distance_to_centre(lat: float, lon: float) -> float:
    """Geodesic (crow-flies) distance in km from a point to central London."""
    return float(geodesic((lat, lon), config.CENTRAL_LONDON).km)


def nearest_district(lat: float, lon: float, centroids: pd.DataFrame | None = None) -> str:
    """Snap a clicked coordinate to the nearest committed postcode district."""
    df = load_centroids() if centroids is None else centroids
    coords = df[["lat", "lon"]].to_numpy(dtype=float)
    dists = [geodesic((lat, lon), (float(a), float(b))).km for a, b in coords]
    return str(df.iloc[int(np.argmin(dists))]["district"])


def district_to_borough() -> dict[str, str]:
    """Map every known postcode district to its borough."""
    df = load_centroids()
    return dict(zip(df["district"], df["borough"], strict=False))


def _lookup_outcode(outcode: str) -> tuple[float, float] | None:
    """Look up an outward code's centroid via the keyless postcodes.io API."""
    import urllib.parse
    import urllib.request

    url = f"https://api.postcodes.io/outcodes/{urllib.parse.quote(outcode)}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310 (trusted host)
            result = (json.load(resp) or {}).get("result") or {}
    except (OSError, ValueError):
        return None
    lat, lon = result.get("latitude"), result.get("longitude")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


def _borough_centroids() -> dict[str, tuple[float, float]]:
    """Representative (lat, lon) per borough, from the committed boundary polygons."""
    from src.regions import load_geojson

    def walk(node: Any, lats: list[float], lons: list[float]) -> None:
        if node and isinstance(node[0], list):
            for child in node:
                walk(child, lats, lons)
        else:
            lons.append(float(node[0]))
            lats.append(float(node[1]))

    centres: dict[str, tuple[float, float]] = {}
    for feature in load_geojson()["features"]:
        props = feature.get("properties", {})
        name = props.get("borough") or props.get("name")
        lats: list[float] = []
        lons: list[float] = []
        walk(feature["geometry"]["coordinates"], lats, lons)
        centres[name] = (sum(lats) / len(lats), sum(lons) / len(lons))
    return centres


def _build(output: str | None = None) -> None:
    """Build the outward-code centroid lookup for the districts seen in the data."""
    import time

    from src.download_data import load_london_flats

    london = load_london_flats()
    district_borough: dict[str, str] = {}
    for postcode, district in zip(london["postcode"], london["district"], strict=False):
        district_borough.setdefault(
            config.outward_code(postcode), config.normalize_borough(district)
        )

    borough_centre = _borough_centroids()
    per_borough: dict[str, int] = {}
    rows = []
    resolved = 0
    for outcode, borough in sorted(district_borough.items()):
        coord = _lookup_outcode(outcode)
        if coord is not None:
            resolved += 1
            lat, lon = coord
        else:
            # Spread fallbacks slightly so districts in one borough stay distinct.
            base = borough_centre.get(borough, config.CENTRAL_LONDON)
            i = per_borough.get(borough, 0)
            per_borough[borough] = i + 1
            lat = base[0] + 0.006 * math.cos(i)
            lon = base[1] + 0.006 * math.sin(i)
        rows.append(
            {"district": outcode, "lat": round(lat, 5), "lon": round(lon, 5), "borough": borough}
        )
        time.sleep(0.1)  # be polite to postcodes.io

    out = pd.DataFrame(rows, columns=CENTROID_COLUMNS)
    dest = output or config.CENTROIDS_CSV
    out.to_csv(dest, index=False)
    print(
        f"Wrote {len(out)} centroids -> {dest} "
        f"({resolved} via postcodes.io, {len(out) - resolved} from borough centroid)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Postcode-district centroids.")
    parser.add_argument("--build", action="store_true", help="Refresh centroids (postcodes.io).")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.build:
        _build(args.output)
    else:
        print(load_centroids().head())


if __name__ == "__main__":
    main()
