"""Shared fixtures: train once per session on the frozen sample, into a temp dir.

Keeps the suite fully offline and independent of any committed binary artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src import train as train_module
from src.infer import Artifacts, load_artifacts


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("artifacts")
    paths = {
        "model": out / "model.pkl",
        "stats": out / "region_stats.csv",
        "catalog": out / "valid_categories.yaml",
    }
    train_module.train(
        output_model=paths["model"],
        output_stats=paths["stats"],
        output_catalog=paths["catalog"],
        use_sample=True,
    )
    return paths


@pytest.fixture(scope="session")
def artifacts(trained: dict[str, Path]) -> Artifacts:
    return load_artifacts(trained["model"])


@pytest.fixture(scope="session")
def catalog(trained: dict[str, Path]) -> dict:
    with trained["catalog"].open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
