"""Turn filtered Land Registry rows into model features.

Low-cardinality categoricals (borough, tenure, year) are one-hot encoded; the
high-cardinality postcode district is captured with out-of-fold target
encoding so the model never sees a row's own price through the encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import config
from src.geocode import distance_to_centre, load_centroids

ONE_HOT_COLUMNS = ["borough", "tenure", "year"]
NUMERIC_COLUMNS = ["new_build", "month", "pd_te", "distance_to_centre"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the raw feature columns from filtered PPD rows.

    Expects the columns produced by :func:`download_data.filter_london_flats`
    and returns a frame carrying ``price``/``log_price`` plus the raw (not yet
    encoded) feature columns.
    """
    out = pd.DataFrame(index=df.index)
    out["price"] = pd.to_numeric(df["price"], errors="coerce")
    out["log_price"] = np.log(out["price"])
    out["borough"] = df["district"].map(config.normalize_borough)
    out["postcode_district"] = df["postcode"].map(config.outward_code)
    out["tenure"] = (
        df["duration"].str.upper().map({"L": "Leasehold", "F": "Freehold"}).fillna("Leasehold")
    )
    out["new_build"] = (df["old_new"].str.upper() == "Y").astype(int)
    dates = pd.to_datetime(df["date_of_transfer"], errors="coerce")
    out["year"] = dates.dt.year
    out["month"] = dates.dt.month

    centroids = load_centroids()
    coords = centroids.set_index("district")[["lat", "lon"]]
    joined = out.join(coords, on="postcode_district")
    out["distance_to_centre"] = [
        distance_to_centre(lat, lon) if pd.notna(lat) else np.nan
        for lat, lon in zip(joined["lat"], joined["lon"], strict=False)
    ]
    median_dist = out["distance_to_centre"].median()
    out["distance_to_centre"] = out["distance_to_centre"].fillna(median_dist)
    return out.dropna(subset=["price", "log_price", "year", "month"]).reset_index(drop=True)


def target_encoding_map(
    df: pd.DataFrame,
    cat_col: str = "postcode_district",
    target_col: str = "log_price",
    smoothing: float = 20.0,
) -> tuple[dict[str, float], float]:
    """Full-data smoothed mean of ``target_col`` per category, for inference."""
    global_mean = float(df[target_col].mean())
    grouped = df.groupby(cat_col)[target_col]
    counts = grouped.count()
    means = grouped.mean()
    smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
    return {str(k): float(v) for k, v in smoothed.items()}, global_mean


def out_of_fold_target_encode(
    df: pd.DataFrame,
    cat_col: str = "postcode_district",
    target_col: str = "log_price",
    n_splits: int = 5,
    smoothing: float = 20.0,
    seed: int = 42,
) -> pd.Series:
    """Out-of-fold encoding: each row is encoded from folds that exclude it.

    A row never contributes to its own encoded value, so the encoding cannot
    leak the row's own price into the model's training features.
    """
    values = np.full(len(df), np.nan, dtype=float)
    kf = KFold(n_splits=min(n_splits, len(df)), shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(df):
        fold = df.iloc[train_idx]
        mapping, global_mean = target_encoding_map(fold, cat_col, target_col, smoothing)
        val_cats = df.iloc[val_idx][cat_col].astype(str)
        values[val_idx] = [mapping.get(cat, global_mean) for cat in val_cats]
    return pd.Series(values, index=df.index, name="pd_te")


def encode_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """One-hot the categoricals and align to ``feature_columns`` (fill 0)."""
    design = frame.reset_index(drop=True).copy()
    design["year"] = design["year"].astype(int).astype(str)
    dummies = pd.get_dummies(design[ONE_HOT_COLUMNS], prefix=ONE_HOT_COLUMNS)
    matrix = pd.concat([dummies, design[NUMERIC_COLUMNS]], axis=1)
    matrix = matrix.astype(float)
    if feature_columns is not None:
        matrix = matrix.reindex(columns=feature_columns, fill_value=0.0)
    return matrix
