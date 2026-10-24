"""predict_price returns a positive GBP point inside a valid conformal range."""

from __future__ import annotations

from src.geocode import load_centroids
from src.infer import Artifacts, get_fx, predict_districts, predict_price
from src.schema import PropertyInput


def _validated(catalog: dict, **kwargs: object) -> PropertyInput:
    return PropertyInput.model_validate(kwargs, context={"catalog": catalog})


def test_predict_price_returns_positive_range(catalog: dict, artifacts: Artifacts) -> None:
    inp = _validated(
        catalog,
        borough="Camden",
        postcode_district="NW1",
        tenure="Leasehold",
        new_build=False,
        year=2026,
    )
    result = predict_price(inp, artifacts)

    assert result.point > 0
    assert 0 < result.low <= result.point <= result.high
    assert result.currency == "GBP"
    assert abs(result.coverage - 0.90) < 1e-9


def test_cheaper_borough_predicts_lower(catalog: dict, artifacts: Artifacts) -> None:
    dear = predict_price(
        _validated(catalog, borough="Kensington and Chelsea", postcode_district="SW3"), artifacts
    )
    cheap = predict_price(
        _validated(catalog, borough="Barking and Dagenham", postcode_district="RM8"), artifacts
    )
    assert dear.point > cheap.point


def test_predict_districts_prices_every_district(catalog: dict, artifacts: Artifacts) -> None:
    centroids = load_centroids()
    inp = _validated(catalog, borough="Camden", postcode_district="NW1")
    frame = predict_districts(inp, artifacts, centroids)

    assert len(frame) == len(centroids)
    assert {"district", "borough", "lat", "lon", "price"}.issubset(frame.columns)
    assert (frame["price"] > 0).all()


def test_get_fx_converts() -> None:
    assert get_fx(100.0, "GBP") == 100.0
    assert get_fx(100.0, "EUR") != 100.0
