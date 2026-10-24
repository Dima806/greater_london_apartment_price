"""Every valid postcode district has a centroid, and clicks snap to a district."""

from __future__ import annotations

from src.geocode import centroid, distance_to_centre, load_centroids, nearest_district


def test_every_catalog_district_has_a_centroid(catalog: dict) -> None:
    centroids = load_centroids()
    known = set(centroids["district"])
    for district in catalog["districts"]:
        assert district in known, f"no centroid for {district}"
        assert centroid(district) is not None


def test_click_snaps_to_nearest_district() -> None:
    centroids = load_centroids()
    row = centroids.iloc[10]
    # A point a few metres from a known centroid snaps back to that district.
    snapped = nearest_district(row["lat"] + 0.0003, row["lon"] - 0.0003, centroids)
    assert snapped == row["district"]


def test_distance_to_centre_is_positive() -> None:
    lat, lon = load_centroids().iloc[0][["lat", "lon"]]
    assert distance_to_centre(float(lat), float(lon)) >= 0.0
