"""Split and adaptive conformal intervals reach near-nominal coverage."""

from __future__ import annotations

import numpy as np

from src.intervals import MondrianConformal, SplitConformal, conformal_quantile


def test_conformal_quantile_grows_as_alpha_shrinks() -> None:
    residuals = np.abs(np.random.default_rng(1).normal(0, 1, 2000))
    q90 = conformal_quantile(residuals, 0.10)
    q95 = conformal_quantile(residuals, 0.05)
    assert q95 > q90 > 0


def test_split_conformal_covers_near_nominal() -> None:
    rng = np.random.default_rng(42)
    errors = rng.normal(0, 1.0, 8000)
    calib, test = errors[:4000], errors[4000:]
    zeros_calib = np.zeros_like(calib)

    model = SplitConformal(alpha=0.10).fit(calib, zeros_calib)
    coverage = model.coverage(test, np.zeros_like(test))

    assert 0.86 <= coverage <= 0.94
    assert abs(model.q - 1.645) < 0.15  # ~90% quantile of a half-normal


def test_mondrian_widens_for_volatile_groups() -> None:
    rng = np.random.default_rng(7)
    calm = rng.normal(0, 0.5, 3000)
    wild = rng.normal(0, 2.0, 3000)
    y_true = np.concatenate([calm, wild])
    y_pred = np.zeros_like(y_true)
    groups = ["calm"] * 3000 + ["wild"] * 3000

    mondrian = MondrianConformal(alpha=0.10, min_group=100).fit(y_true, y_pred, groups)

    assert mondrian.q_for("wild") > mondrian.q_for("calm")
    assert 0.87 <= mondrian.coverage(y_true, y_pred, groups) <= 0.93
    assert mondrian.q_for("unseen") == mondrian.global_q  # fallback for unknown groups
