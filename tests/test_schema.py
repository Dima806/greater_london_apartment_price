"""PropertyInput validates and normalizes categories against the catalog."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schema import PropertyInput


def test_valid_input_passes(catalog: dict) -> None:
    inp = PropertyInput.model_validate(
        {"borough": "Camden", "postcode_district": "NW1", "year": 2025},
        context={"catalog": catalog},
    )
    assert inp.borough == "Camden"
    assert inp.tenure == "Leasehold"  # default


def test_borough_is_normalized(catalog: dict) -> None:
    inp = PropertyInput.model_validate(
        {"borough": "camden", "postcode_district": "nw1 2ab"},
        context={"catalog": catalog},
    )
    assert inp.borough == "Camden"
    assert inp.postcode_district == "NW1"


def test_unknown_borough_rejected(catalog: dict) -> None:
    with pytest.raises(ValidationError):
        PropertyInput.model_validate(
            {"borough": "Atlantis", "postcode_district": "NW1"},
            context={"catalog": catalog},
        )


def test_unknown_district_rejected(catalog: dict) -> None:
    with pytest.raises(ValidationError):
        PropertyInput.model_validate(
            {"borough": "Camden", "postcode_district": "ZZ99"},
            context={"catalog": catalog},
        )


def test_month_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        PropertyInput(borough="Camden", postcode_district="NW1", month=13)


def test_without_catalog_skips_category_check() -> None:
    # No context → structural validation only, so an arbitrary borough is allowed.
    inp = PropertyInput(borough="Anywhere", postcode_district="AB1")
    assert inp.borough == "Anywhere"
