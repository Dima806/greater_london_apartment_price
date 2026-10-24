"""The ``PropertyInput`` request model and the category catalog it validates against."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from src import config


def load_catalog() -> dict[str, Any]:
    """Load the valid-category catalog written by training."""
    if not config.VALID_CATEGORIES_YAML.exists():
        return {}
    return config.load_yaml(config.VALID_CATEGORIES_YAML)


class PropertyInput(BaseModel):
    """A single flat to price.

    Pass ``context={"catalog": load_catalog()}`` to
    :meth:`~pydantic.BaseModel.model_validate` to reject boroughs and postcode
    districts the model was not trained on.
    """

    borough: str
    postcode_district: str
    tenure: Literal["Leasehold", "Freehold"] = "Leasehold"
    new_build: bool = False
    year: int = Field(default=2026)
    month: int = Field(default=6, ge=1, le=12)

    @field_validator("borough", mode="before")
    @classmethod
    def _norm_borough(cls, v: Any) -> str:
        return config.normalize_borough(str(v))

    @field_validator("postcode_district", mode="before")
    @classmethod
    def _norm_district(cls, v: Any) -> str:
        return config.outward_code(str(v))

    @model_validator(mode="after")
    def _check_against_catalog(self, info: ValidationInfo) -> PropertyInput:
        catalog = (info.context or {}).get("catalog") if info.context else None
        if not catalog:
            return self
        boroughs = set(catalog.get("boroughs", []))
        districts = set(catalog.get("districts", []))
        if boroughs and self.borough not in boroughs:
            raise ValueError(f"unknown borough: {self.borough!r}")
        if districts and self.postcode_district not in districts:
            raise ValueError(f"unknown postcode district: {self.postcode_district!r}")
        return self
