"""Greater London Apartment Price — a map-first Streamlit app.

The map is the interface: pick a borough by clicking or with the sidebar
selectors, and read the estimate as a range in context rather than a lone
number. This file stays thin; the work lives in ``src/`` and ``ui/``.
"""

from __future__ import annotations

import streamlit as st

from src.geocode import load_centroids
from src.infer import get_fx, load_artifacts, predict_districts, predict_price
from src.regions import load_geojson, load_stats
from src.schema import PropertyInput, load_catalog
from ui.controls import render_controls
from ui.map_view import build_map, render
from ui.state import apply_map_click, current_input, ensure_consistent, init_state


@st.cache_resource
def _load_backend() -> tuple:
    return (
        load_catalog(),
        load_stats(),
        load_geojson(),
        load_centroids(),
        load_artifacts(),
    )


def _gbp(value: float) -> str:
    return f"£{value:,.0f}"


def _predict(inp: PropertyInput, artifacts, catalog):
    validated = PropertyInput.model_validate(inp.model_dump(), context={"catalog": catalog})
    return predict_price(validated, artifacts)


def main() -> None:
    st.set_page_config(page_title="Greater London Apartment Price", page_icon="🏠", layout="wide")
    catalog, stats, geojson, centroids, artifacts = _load_backend()
    init_state(catalog)
    ensure_consistent(catalog)

    st.title("🏠 Greater London Apartment Price")
    st.caption(
        "Typical sold prices for flats, from HM Land Registry. The estimate is a range, "
        "not a single figure — the open data has no floor area, so this is the typical "
        "price for a location, tenure, and period, not one specific flat."
    )

    render_controls(catalog)

    inp = current_input()
    prediction = _predict(inp, artifacts, catalog)

    # Headline estimate with the conformal range (above the map).
    left, right = st.columns([2, 3])
    with left:
        st.metric(f"{inp.postcode_district} · {inp.borough}", _gbp(prediction.point))
        pct = int(round(prediction.coverage * 100))
        st.markdown(
            f"**Typically {_gbp(prediction.point)}**, with a {pct}% range of "
            f"**{_gbp(prediction.low)} – {_gbp(prediction.high)}** for comparable flats."
        )
        if st.session_state.compare_years:
            other = PropertyInput.model_validate(
                {**inp.model_dump(), "year": 2025 if inp.year == 2026 else 2026},
                context={"catalog": catalog},
            )
            other_pred = predict_price(other, artifacts)
            a, b = sorted([(inp.year, prediction.point), (other.year, other_pred.point)])
            delta = (b[1] - a[1]) / a[1] * 100 if a[1] else 0.0
            st.markdown(f"**{a[0]}:** {_gbp(a[1])}  →  **{b[0]}:** {_gbp(b[1])}  ({delta:+.1f}%)")

    with right:
        # Comparing years overrides the map to the change layer, without
        # disturbing the user's own "Colour by" choice.
        map_color_by = "yoy" if st.session_state.compare_years else st.session_state.color_by
        district_prices = predict_districts(inp, artifacts, centroids)
        if map_color_by == "yoy":
            d2025 = predict_districts(inp.model_copy(update={"year": 2025}), artifacts, centroids)
            d2026 = predict_districts(inp.model_copy(update={"year": 2026}), artifacts, centroids)
            a, b = d2025["price"].to_numpy(), d2026["price"].to_numpy()
            district_prices = district_prices.assign(change_pct=(b - a) / a * 100.0)
        fmap = build_map(
            geojson,
            stats,
            map_color_by,
            st.session_state.borough,
            st.session_state.postcode_district,
            marker_price=prediction.point,
            district_prices=district_prices,
        )
        click = render(fmap)

    if click:
        before = st.session_state.postcode_district
        apply_map_click(click, catalog, centroids)
        if st.session_state.postcode_district != before:
            st.rerun()

    # Progressive disclosure: detail lives in expanders.
    with st.expander("Why this price (SHAP)"):
        if prediction.shap:
            st.caption("Contribution to the estimate, largest first (log-price space).")
            for label, value in prediction.shap:
                st.write(f"{'▲' if value > 0 else '▼'} {label}: {value:+.3f}")
        else:
            st.write("No explanation available.")

    with st.expander("Convert currency"):
        currency = st.selectbox("Currency", ["GBP", "EUR", "USD"])
        st.write(
            f"{_gbp(prediction.point)} ≈ {get_fx(prediction.point, currency):,.0f} {currency}"
        )

    with st.expander("About & caveats"):
        st.markdown(
            "- **Sold prices, not asking prices** — HM Land Registry Price Paid Data.\n"
            "- **No floor area, bedrooms, or bathrooms** in the data, so this is a typical "
            "location–tenure–period price, not a specific unit.\n"
            "- The range is a **conformal interval**; it assumes exchangeability, and a sharp "
            "market move can drift it, so calibration uses the most recent period.\n"
            "- Data: HM Land Registry & London Datastore, Open Government Licence."
        )


if __name__ == "__main__":
    main()
