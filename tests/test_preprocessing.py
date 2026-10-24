"""Out-of-fold target encoding must not leak a row's own price into its encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import encode_matrix, out_of_fold_target_encode


def test_oof_encoding_does_not_see_own_price() -> None:
    rng = np.random.default_rng(0)
    rows = (
        [{"postcode_district": "AA1", "log_price": v} for v in rng.normal(10.0, 0.1, 60)]
        + [{"postcode_district": "BB2", "log_price": v} for v in rng.normal(12.0, 0.1, 60)]
        # A single, extreme-priced district: its OOF value cannot come from itself.
        + [{"postcode_district": "ZZ9", "log_price": 20.0}]
    )
    df = pd.DataFrame(rows)

    encoded = out_of_fold_target_encode(df, n_splits=5, smoothing=0.0, seed=0)
    zz9 = encoded[df["postcode_district"] == "ZZ9"].iloc[0]

    global_mean = df["log_price"].mean()
    # If the encoding leaked, zz9 would sit near 20; instead it falls back to the mean.
    assert abs(zz9 - 20.0) > 4.0
    assert abs(zz9 - global_mean) < 2.0
    assert encoded.notna().all()


def test_encode_matrix_reindexes_to_feature_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "borough": "Westminster",
                "tenure": "Leasehold",
                "year": 2026,
                "new_build": 0,
                "month": 6,
                "pd_te": 13.0,
                "distance_to_centre": 2.5,
            }
        ]
    )
    columns = ["borough_Camden", "borough_Westminster", "tenure_Leasehold", "year_2026", "pd_te"]
    matrix = encode_matrix(frame, feature_columns=columns)

    assert list(matrix.columns) == columns
    assert matrix.loc[0, "borough_Westminster"] == 1.0
    assert matrix.loc[0, "borough_Camden"] == 0.0  # absent category filled with 0
    assert matrix.loc[0, "pd_te"] == 13.0
