"""Sidebar selectors, kept in two-way sync with the map through session state."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src import config
from ui.state import COLOR_MODES, districts_for, reset_state


def render_controls(catalog: dict[str, Any]) -> None:
    """Draw the sidebar controls, writing choices back into session state."""
    boroughs = list(catalog.get("boroughs", []))

    st.sidebar.header("Location")
    b_idx = boroughs.index(st.session_state.borough) if st.session_state.borough in boroughs else 0
    st.session_state.borough = st.sidebar.selectbox("Borough", boroughs, index=b_idx)

    districts = districts_for(catalog, st.session_state.borough)
    if st.session_state.postcode_district not in districts:
        st.session_state.postcode_district = districts[0]
    d_idx = districts.index(st.session_state.postcode_district)
    st.session_state.postcode_district = st.sidebar.selectbox(
        "Postcode district", districts, index=d_idx
    )

    st.sidebar.header("Attributes")
    st.session_state.tenure = st.sidebar.radio(
        "Tenure", config.TENURES, index=config.TENURES.index(st.session_state.tenure)
    )
    st.session_state.new_build = st.sidebar.toggle("New build", value=st.session_state.new_build)
    st.session_state.year = st.sidebar.radio(
        "Year", config.YEARS, index=config.YEARS.index(st.session_state.year), horizontal=True
    )
    st.session_state.compare_years = st.sidebar.checkbox(
        "Compare 2025 and 2026", value=st.session_state.compare_years
    )

    st.sidebar.header("Map")
    labels = list(COLOR_MODES.keys())
    current = next(lbl for lbl, mode in COLOR_MODES.items() if mode == st.session_state.color_by)
    chosen = st.sidebar.selectbox(
        "Colour by",
        labels,
        index=labels.index(current),
        disabled=st.session_state.compare_years,
        help="While comparing years, the map shows the 2025 → 2026 change.",
    )
    st.session_state.color_by = COLOR_MODES[chosen]

    if st.sidebar.button("Reset", use_container_width=True):
        reset_state(catalog)
        st.rerun()
