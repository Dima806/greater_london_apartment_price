"""Conformal prediction intervals, implemented from scratch (no ``mapie``).

Ported from ``prediction_intervals_arena``: split conformal as the baseline
and adaptive (Mondrian) conformal as the upgrade. Everything works on
log-price residuals; the caller exponentiates the bounds back to GBP.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def conformal_quantile(residuals: Sequence[float], alpha: float) -> float:
    """Finite-sample conformal quantile of absolute residuals.

    Uses the ``ceil((n+1)(1-alpha)) / n`` order statistic, which gives the
    split-conformal coverage guarantee of at least ``1 - alpha``.
    """
    res = np.sort(np.asarray(residuals, dtype=float))
    n = res.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(res[k - 1])


class SplitConformal:
    """Symmetric split-conformal interval around a point prediction."""

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.q: float = float("inf")

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> SplitConformal:
        residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        self.q = conformal_quantile(residuals, self.alpha)
        return self

    def interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float)
        return y_pred - self.q, y_pred + self.q

    def coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        lo, hi = self.interval(y_pred)
        y_true = np.asarray(y_true, dtype=float)
        return float(np.mean((y_true >= lo) & (y_true <= hi)))


class MondrianConformal:
    """Adaptive conformal: a separate quantile per group (e.g. per borough).

    Groups with fewer than ``min_group`` calibration points fall back to the
    pooled global quantile, so the interval widens in volatile areas while
    keeping the coverage guarantee.
    """

    def __init__(self, alpha: float = 0.10, min_group: int = 50) -> None:
        self.alpha = alpha
        self.min_group = min_group
        self.global_q: float = float("inf")
        self.group_q: dict[str, float] = {}

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Sequence[str],
    ) -> MondrianConformal:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = np.abs(y_true - y_pred)
        group_arr = np.asarray(groups)
        self.global_q = conformal_quantile(residuals, self.alpha)
        for group in np.unique(group_arr):
            mask = group_arr == group
            if int(np.count_nonzero(mask)) >= self.min_group:
                self.group_q[str(group)] = conformal_quantile(residuals[mask], self.alpha)
        return self

    def q_for(self, group: str) -> float:
        return self.group_q.get(str(group), self.global_q)

    def interval(self, y_pred: float, group: str) -> tuple[float, float]:
        q = self.q_for(group)
        return y_pred - q, y_pred + q

    def coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Sequence[str],
    ) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        qs = np.array([self.q_for(g) for g in groups])
        return float(np.mean((y_true >= y_pred - qs) & (y_true <= y_pred + qs)))
