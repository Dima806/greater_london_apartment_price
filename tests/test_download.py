"""The London-flat filter keeps only standard-category flats in Greater London."""

from __future__ import annotations

import pandas as pd

from src import config
from src.download_data import filter_london_flats


def _row(**overrides: object) -> dict[str, object]:
    base = {
        "transaction_id": "{T}",
        "price": "500000",
        "date_of_transfer": "2026-05-15 00:00",
        "postcode": "SW1 1AA",
        "property_type": "F",
        "old_new": "N",
        "duration": "L",
        "paon": "1",
        "saon": "",
        "street": "SAMPLE STREET",
        "locality": "",
        "town_city": "LONDON",
        "district": "WESTMINSTER",
        "county": "GREATER LONDON",
        "ppd_category_type": "A",
        "record_status": "A",
    }
    base.update(overrides)
    return base


def test_filter_keeps_only_london_flats() -> None:
    raw = pd.DataFrame(
        [
            _row(),  # valid flat — kept
            _row(property_type="D"),  # a house — dropped
            _row(county="SURREY"),  # outside Greater London — dropped
            _row(ppd_category_type="B"),  # non-standard category — dropped
            _row(price="10000"),  # below the price floor — dropped
        ],
        columns=config.PPD_COLUMNS,
    )

    out = filter_london_flats(raw)

    assert len(out) == 1
    assert (out["property_type"].str.upper() == config.PROPERTY_TYPE_FLAT).all()
    assert (out["county"].str.upper() == config.COUNTY_GREATER_LONDON).all()
    assert out["price"].dtype.kind in "iu"


def test_sample_passes_the_filter() -> None:
    from src.download_data import load_london_flats

    london = load_london_flats(use_sample=True)
    assert len(london) > 1000
    assert (london["property_type"].str.upper() == "F").all()
