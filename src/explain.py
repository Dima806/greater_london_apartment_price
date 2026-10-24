"""Tree SHAP contributions for a single prediction (the 'why this price' panel)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def prettify_feature(column: str) -> str:
    """Human-readable label for an encoded feature column."""
    mapping = {
        "pd_te": "Postcode district (typical level)",
        "distance_to_centre": "Distance to central London",
        "new_build": "New build",
        "month": "Month",
    }
    if column in mapping:
        return mapping[column]
    for prefix, label in (("borough_", "Borough"), ("tenure_", "Tenure"), ("year_", "Year")):
        if column.startswith(prefix):
            return f"{label}: {column[len(prefix) :]}"
    return column


def explain_prediction(
    explainer: shap.TreeExplainer,
    x_row: pd.DataFrame,
    top_k: int = 8,
) -> list[tuple[str, float]]:
    """Return the ``top_k`` features by absolute SHAP value for one row.

    SHAP values are in log-price space (the model's target); the sign shows
    whether a feature pushes the estimate up or down.
    """
    values = np.asarray(explainer.shap_values(x_row))
    row = values[0] if values.ndim == 2 else values
    contributions = [
        (prettify_feature(col), float(val))
        for col, val in zip(x_row.columns, row, strict=False)
        if abs(float(val)) > 1e-9
    ]
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    return contributions[:top_k]
