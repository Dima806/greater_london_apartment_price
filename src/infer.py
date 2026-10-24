"""Inference: validate input, predict a price, attach a conformal range + SHAP.

Mirrors the salary project's ``infer.py``: a thin, serving-only module that
loads the pickled artifact and answers one request at a time.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.explain import explain_prediction
from src.geocode import centroid, distance_to_centre
from src.preprocessing import encode_matrix
from src.schema import PropertyInput


@dataclass
class Artifacts:
    """Everything needed to serve predictions, loaded from ``model.pkl``."""

    model: Any
    feature_columns: list[str]
    target_encoding: dict[str, float]
    target_global_mean: float
    median_distance: float
    conformal: dict[str, Any]
    _explainer: Any = field(default=None, repr=False)

    @property
    def explainer(self) -> Any:
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer


@dataclass
class Prediction:
    """A priced flat: the point estimate and its calibrated range, in GBP."""

    point: float
    low: float
    high: float
    currency: str
    borough: str
    coverage: float
    shap: list[tuple[str, float]]


def save_artifacts(artifacts: dict[str, Any], path: Path = config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(artifacts, fh)


@lru_cache(maxsize=4)
def load_artifacts(path: Path = config.MODEL_PATH) -> Artifacts:
    with Path(path).open("rb") as fh:
        data = pickle.load(fh)  # noqa: S301  (our own trusted artifact)
    return Artifacts(
        model=data["model"],
        feature_columns=list(data["feature_columns"]),
        target_encoding=dict(data["target_encoding"]),
        target_global_mean=float(data["target_global_mean"]),
        median_distance=float(data["median_distance"]),
        conformal=dict(data["conformal"]),
    )


def _feature_frame(inp: PropertyInput, artifacts: Artifacts) -> pd.DataFrame:
    """Build the one-row raw feature frame for a request."""
    pd_te = artifacts.target_encoding.get(inp.postcode_district, artifacts.target_global_mean)
    coord = centroid(inp.postcode_district)
    dist = distance_to_centre(*coord) if coord else artifacts.median_distance
    return pd.DataFrame(
        [
            {
                "borough": inp.borough,
                "tenure": inp.tenure,
                "year": inp.year,
                "new_build": int(inp.new_build),
                "month": inp.month,
                "pd_te": pd_te,
                "distance_to_centre": dist,
            }
        ]
    )


def predict_price(inp: PropertyInput, artifacts: Artifacts | None = None) -> Prediction:
    """Predict a flat's typical price with a conformal range and SHAP reasons."""
    artifacts = artifacts or load_artifacts()
    frame = _feature_frame(inp, artifacts)
    x_row = encode_matrix(frame, artifacts.feature_columns)

    log_point = float(artifacts.model.predict(x_row)[0])
    conformal = artifacts.conformal
    q = conformal.get("group_q", {}).get(inp.borough, conformal["global_q"])
    point = float(np.exp(log_point))
    low = float(np.exp(log_point - q))
    high = float(np.exp(log_point + q))

    shap_values = explain_prediction(artifacts.explainer, x_row)
    return Prediction(
        point=point,
        low=low,
        high=high,
        currency="GBP",
        borough=inp.borough,
        coverage=1.0 - conformal["alpha"],
        shap=shap_values,
    )


def predict_districts(
    inp: PropertyInput, artifacts: Artifacts, centroids: pd.DataFrame
) -> pd.DataFrame:
    """Predicted point price for every postcode district, for the given spec.

    One batched model call, so the per-district map layer is cheap to refresh.
    Returns columns: district, borough, lat, lon, price (GBP).
    """
    districts = centroids["district"].tolist()
    boroughs = centroids["borough"].tolist()
    lats = [float(v) for v in centroids["lat"]]
    lons = [float(v) for v in centroids["lon"]]
    pd_te = [artifacts.target_encoding.get(d, artifacts.target_global_mean) for d in districts]
    dist = [distance_to_centre(la, lo) for la, lo in zip(lats, lons, strict=False)]
    frame = pd.DataFrame(
        {
            "borough": boroughs,
            "tenure": inp.tenure,
            "year": inp.year,
            "new_build": int(inp.new_build),
            "month": inp.month,
            "pd_te": pd_te,
            "distance_to_centre": dist,
        }
    )
    x = encode_matrix(frame, artifacts.feature_columns)
    prices = np.exp(artifacts.model.predict(x))
    return pd.DataFrame(
        {"district": districts, "borough": boroughs, "lat": lats, "lon": lons, "price": prices}
    )


def get_fx(price: float, currency: str) -> float:
    """Convert a GBP price for display using committed reference rates."""
    if currency.upper() == "GBP":
        return price
    rates = config.load_yaml(config.FX_RATES_YAML) if config.FX_RATES_YAML.exists() else {}
    return price * float(rates.get(currency.upper(), 1.0))
