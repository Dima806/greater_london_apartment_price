"""Session-state defaults and two-way sync between map clicks and selectors.

``st_folium`` reruns the whole script on every interaction, so all selection
state lives here. The map returns only the latest click; the selectors keep
their values across reruns, and the two are reconciled every pass, which is the
documented way to stop the map or inputs silently resetting.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.geocode import nearest_district
from src.schema import PropertyInput

DEFAULT_BOROUGH = "Westminster"
COLOR_MODES = {
    "Price level": "level",
    "Change 2025 → 2026": "yoy",
    "Vs London median": "vs_median",
}


def districts_for(catalog: dict[str, Any], borough: str) -> list[str]:
    """Postcode districts within a borough (all districts as a fallback)."""
    d2b = catalog.get("district_to_borough", {})
    districts = sorted(d for d, b in d2b.items() if b == borough)
    return districts or sorted(catalog.get("districts", []))


def init_state(catalog: dict[str, Any]) -> None:
    """Populate sensible defaults so the first load shows a real estimate."""
    boroughs = catalog.get("boroughs", [])
    if "borough" not in st.session_state:
        st.session_state.borough = DEFAULT_BOROUGH if DEFAULT_BOROUGH in boroughs else boroughs[0]
    if "postcode_district" not in st.session_state:
        st.session_state.postcode_district = districts_for(catalog, st.session_state.borough)[0]
    st.session_state.setdefault("tenure", "Leasehold")
    st.session_state.setdefault("new_build", False)
    st.session_state.setdefault("year", 2026)
    st.session_state.setdefault("month", 6)
    st.session_state.setdefault("color_by", "level")
    st.session_state.setdefault("compare_years", False)


def reset_state(catalog: dict[str, Any]) -> None:
    """Return every control and the map to defaults."""
    for key in (
        "borough",
        "postcode_district",
        "tenure",
        "new_build",
        "year",
        "month",
        "color_by",
        "compare_years",
    ):
        st.session_state.pop(key, None)
    init_state(catalog)


def apply_map_click(
    click: dict[str, Any] | None, catalog: dict[str, Any], centroids: pd.DataFrame
) -> None:
    """Reconcile a map click into the selectors (borough + nearest district)."""
    if not click:
        return
    lat, lon = click.get("lat"), click.get("lng")
    if lat is None or lon is None:
        return
    district = nearest_district(lat, lon, centroids)
    st.session_state.postcode_district = district
    st.session_state.borough = catalog.get("district_to_borough", {}).get(
        district, st.session_state.borough
    )


def ensure_consistent(catalog: dict[str, Any]) -> None:
    """Keep the chosen district inside the chosen borough."""
    districts = districts_for(catalog, st.session_state.borough)
    if st.session_state.postcode_district not in districts:
        st.session_state.postcode_district = districts[0]


def current_input() -> PropertyInput:
    return PropertyInput(
        borough=st.session_state.borough,
        postcode_district=st.session_state.postcode_district,
        tenure=st.session_state.tenure,
        new_build=st.session_state.new_build,
        year=st.session_state.year,
        month=st.session_state.month,
    )
