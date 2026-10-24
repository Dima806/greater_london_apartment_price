"""Every borough polygon gets stats, and the choropleth bins are valid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.regions import STATS_COLUMNS, borough_names, compute_bins, load_geojson


def test_every_borough_polygon_has_stats(trained: dict[str, Path]) -> None:
    stats = pd.read_csv(trained["stats"])
    geojson = load_geojson()
    polygons = set(borough_names(geojson))

    assert list(stats.columns) == STATS_COLUMNS
    assert polygons.issubset(set(stats["borough"]))
    assert len(stats) == len(polygons)
    assert stats[["median_price", "predicted_price"]].notna().all().all()
    assert (stats["median_price"] > 0).all()


def test_compute_bins_are_sorted_and_sized() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    edges = compute_bins(values, k=5)

    assert len(edges) <= 6
    assert edges == sorted(edges)
    assert edges[0] <= values.min()
    assert edges[-1] >= values.max()


def test_compute_bins_handles_empty() -> None:
    assert len(compute_bins([], k=5)) == 6
