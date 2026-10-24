"""Train XGBoost on log price, calibrate conformal intervals, write all artifacts.

Produces ``models/model.pkl`` plus the committed map inputs
(``config/region_stats.csv``, ``config/valid_categories.yaml``). Runs offline
on the frozen sample when the full download is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src import config
from src.download_data import load_london_flats
from src.geocode import centroid, distance_to_centre, district_to_borough, load_centroids
from src.intervals import MondrianConformal, SplitConformal
from src.preprocessing import (
    add_features,
    encode_matrix,
    out_of_fold_target_encode,
    target_encoding_map,
)
from src.regions import borough_names, build_region_stats, load_geojson

RECENT_YEAR = 2026
PRIOR_YEAR = 2025


def _matrix(df: pd.DataFrame, pd_te_col: str, feature_columns: list[str] | None) -> pd.DataFrame:
    tmp = df.copy()
    tmp["pd_te"] = df[pd_te_col].to_numpy()
    return encode_matrix(tmp, feature_columns)


def train(
    output_model: Path = config.MODEL_PATH,
    output_stats: Path = config.REGION_STATS_CSV,
    output_catalog: Path = config.VALID_CATEGORIES_YAML,
    use_sample: bool = False,
) -> dict[str, Any]:
    params = config.load_model_parameters()
    seed = int(params["seed"])
    np.random.seed(seed)

    london = load_london_flats(use_sample=use_sample)
    feats = add_features(london).reset_index(drop=True)
    print(f"Training on {len(feats):,} Greater London flat sales.")

    # Target encoding: OOF for the model's own training rows (no self-leak),
    # a full-data map for serving and for the calibration/test rows.
    te_cfg = params["target_encoding"]
    full_map, full_mean = target_encoding_map(feats, smoothing=te_cfg["smoothing"])
    feats["pd_te_oof"] = out_of_fold_target_encode(
        feats, n_splits=te_cfg["n_splits"], smoothing=te_cfg["smoothing"], seed=seed
    )
    feats["pd_te_map"] = [full_map.get(d, full_mean) for d in feats["postcode_district"]]

    feature_columns = list(_matrix(feats, "pd_te_map", None).columns)

    # Random split (train 60% | calibrate 20% | test 20%). A random split keeps
    # both years balanced in training, so the model learns the year effect, and
    # it satisfies the exchangeability that split conformal assumes.
    train_df, tmp_df = train_test_split(feats, test_size=0.4, random_state=seed)
    calib_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=seed)

    x_train = _matrix(train_df, "pd_te_oof", feature_columns)
    x_calib = _matrix(calib_df, "pd_te_map", feature_columns)
    x_test = _matrix(test_df, "pd_te_map", feature_columns)

    model = XGBRegressor(**params["xgboost"])
    model.fit(x_train, train_df["log_price"])

    # Conformal calibration on the held-out calibration slice.
    alpha = float(params["conformal"]["alpha"])
    min_group = int(params["conformal"]["min_group"])
    calib_pred = model.predict(x_calib)
    split = SplitConformal(alpha).fit(calib_df["log_price"].to_numpy(), calib_pred)
    mondrian = MondrianConformal(alpha, min_group).fit(
        calib_df["log_price"].to_numpy(), calib_pred, calib_df["borough"].to_numpy()
    )

    # Report coverage + a borough-median baseline on the held-out test period.
    test_pred = model.predict(x_test)
    cov_split = split.coverage(test_df["log_price"].to_numpy(), test_pred)
    cov_mond = mondrian.coverage(
        test_df["log_price"].to_numpy(), test_pred, test_df["borough"].to_numpy()
    )
    model_mae = float(np.mean(np.abs(np.exp(test_pred) - test_df["price"].to_numpy())))
    borough_median = train_df.groupby("borough")["price"].median()
    baseline = test_df["borough"].map(borough_median).fillna(train_df["price"].median())
    baseline_mae = float(np.mean(np.abs(baseline.to_numpy() - test_df["price"].to_numpy())))
    print(
        f"Coverage: split={cov_split:.3f}, mondrian={cov_mond:.3f} (target {1 - alpha:.2f}).\n"
        f"MAE: model=£{model_mae:,.0f} vs borough-median baseline=£{baseline_mae:,.0f}."
    )

    # Reference predictor for the per-borough map statistics.
    centroids = load_centroids()
    median_distance = float(feats["distance_to_centre"].median())
    rep = (
        feats.groupby("borough")["postcode_district"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    d2b = district_to_borough()
    borough_to_district = {b: d for d, b in d2b.items()}

    def reference_predictor(name: str) -> float:
        district = rep.get(name) or borough_to_district.get(name)
        pd_te = full_map.get(district, full_mean) if district else full_mean
        coord = centroid(district) if district else None
        dist = distance_to_centre(*coord) if coord else median_distance
        frame = pd.DataFrame(
            [
                {
                    "borough": name,
                    "tenure": "Leasehold",
                    "year": RECENT_YEAR,
                    "new_build": 0,
                    "month": 6,
                    "pd_te": pd_te,
                    "distance_to_centre": dist,
                }
            ]
        )
        return float(np.exp(model.predict(encode_matrix(frame, feature_columns))[0]))

    geojson = load_geojson()
    stats = build_region_stats(feats, geojson, reference_predictor, RECENT_YEAR, PRIOR_YEAR)
    output_stats.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_stats, index=False)
    print(f"Wrote region stats for {len(stats)} boroughs -> {output_stats}")

    catalog = {
        "boroughs": sorted(
            set(feats["borough"]) | set(borough_names(geojson)) | set(centroids["borough"])
        ),
        "districts": sorted(set(feats["postcode_district"]) | set(centroids["district"])),
        "district_to_borough": d2b,
        "tenures": config.TENURES,
        "years": config.YEARS,
    }
    config.save_yaml(catalog, output_catalog)
    print(f"Wrote valid categories -> {output_catalog}")

    artifacts = {
        "model": model,
        "feature_columns": feature_columns,
        "target_encoding": full_map,
        "target_global_mean": full_mean,
        "median_distance": median_distance,
        "conformal": {
            "alpha": alpha,
            "min_group": min_group,
            "global_q": mondrian.global_q,
            "group_q": mondrian.group_q,
            "coverage_split": cov_split,
            "coverage_mondrian": cov_mond,
        },
    }
    from src.infer import save_artifacts

    save_artifacts(artifacts, output_model)
    print(f"Wrote model artifact -> {output_model}")
    return artifacts


def main() -> None:
    train()


if __name__ == "__main__":
    main()
