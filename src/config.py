"""Paths, constants, and small config loaders shared across the project.

Everything resolves relative to the repository root so the code runs the same
way in a Codespace, in CI, and on the Hugging Face Space. Output locations can
be redirected with ``GLAP_*`` environment variables, which the test suite uses
to train into a temporary directory without touching committed artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# --- Repository layout -------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def _dir(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override) if override else default


CONFIG_DIR = _dir("GLAP_CONFIG_DIR", ROOT / "config")
MODELS_DIR = _dir("GLAP_MODELS_DIR", ROOT / "models")
DATA_DIR = _dir("GLAP_DATA_DIR", ROOT / "data")
SAMPLE_DIR = DATA_DIR / "sample"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_PATH = MODELS_DIR / "model.pkl"
BOROUGHS_GEOJSON = CONFIG_DIR / "london_boroughs.geojson"
REGION_STATS_CSV = CONFIG_DIR / "region_stats.csv"
CENTROIDS_CSV = CONFIG_DIR / "postcode_centroids.csv"
VALID_CATEGORIES_YAML = CONFIG_DIR / "valid_categories.yaml"
MODEL_PARAMETERS_YAML = CONFIG_DIR / "model_parameters.yaml"
FX_RATES_YAML = CONFIG_DIR / "fx_rates.yaml"
SAMPLE_CSV = SAMPLE_DIR / "pp_sample.csv"

# --- Land Registry Price Paid Data -------------------------------------------

# The CSV has no header; these are the 16 columns in order.
PPD_COLUMNS = [
    "transaction_id",
    "price",
    "date_of_transfer",
    "postcode",
    "property_type",
    "old_new",
    "duration",
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category_type",
    "record_status",
]

PPD_URLS = {
    2025: "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2025.csv",
    2026: "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2026.csv",
}

# London-flat filter.
PROPERTY_TYPE_FLAT = "F"
COUNTY_GREATER_LONDON = "GREATER LONDON"
PPD_CATEGORY_STANDARD = "A"
MIN_PRICE = 50_000
MAX_PRICE = 5_000_000

TENURES = ["Leasehold", "Freehold"]
YEARS = [2025, 2026]

# Charing Cross — the conventional centre of London, for distance-to-centre.
CENTRAL_LONDON = (51.5074, -0.1278)

# The 32 London boroughs plus the City of London (33 local authorities),
# in the canonical casing used by the GeoJSON, centroids, and region stats.
BOROUGHS = [
    "City of London",
    "Barking and Dagenham",
    "Barnet",
    "Bexley",
    "Brent",
    "Bromley",
    "Camden",
    "Croydon",
    "Ealing",
    "Enfield",
    "Greenwich",
    "Hackney",
    "Hammersmith and Fulham",
    "Haringey",
    "Harrow",
    "Havering",
    "Hillingdon",
    "Hounslow",
    "Islington",
    "Kensington and Chelsea",
    "Kingston upon Thames",
    "Lambeth",
    "Lewisham",
    "Merton",
    "Newham",
    "Redbridge",
    "Richmond upon Thames",
    "Southwark",
    "Sutton",
    "Tower Hamlets",
    "Waltham Forest",
    "Wandsworth",
    "Westminster",
]

_BOROUGH_BY_UPPER = {name.upper(): name for name in BOROUGHS}

# HM Land Registry district names that differ from the ONS borough names.
_BOROUGH_ALIASES = {
    "CITY OF WESTMINSTER": "Westminster",
}


def normalize_borough(raw: str) -> str:
    """Map a raw (upper-case) Land Registry district to a canonical borough name."""
    key = str(raw).strip().upper()
    if key in _BOROUGH_ALIASES:
        return _BOROUGH_ALIASES[key]
    return _BOROUGH_BY_UPPER.get(key, str(raw).strip().title())


def outward_code(postcode: str) -> str:
    """Return the outward code (postcode district), e.g. ``'SW11 1AA' -> 'SW11'``."""
    return str(postcode).strip().upper().split(" ")[0]


# --- Tiny YAML helpers -------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def load_model_parameters() -> dict[str, Any]:
    """Model hyper-parameters, filters, and the conformal alpha (with defaults)."""
    defaults: dict[str, Any] = {
        "xgboost": {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "random_state": 42,
        },
        "filters": {
            "min_price": MIN_PRICE,
            "max_price": MAX_PRICE,
        },
        "conformal": {
            "alpha": 0.10,
            "min_group": 50,
        },
        "target_encoding": {
            "n_splits": 5,
            "smoothing": 20.0,
        },
        "seed": 42,
    }
    if MODEL_PARAMETERS_YAML.exists():
        loaded = load_yaml(MODEL_PARAMETERS_YAML)
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(defaults.get(key), dict):
                defaults[key].update(value)
            else:
                defaults[key] = value
    return defaults
