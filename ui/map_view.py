"""The interactive Folium choropleth: borders, colour scales, tooltips, marker."""

from __future__ import annotations

import copy
from typing import Any

import folium
import pandas as pd
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template
from streamlit_folium import st_folium

from src import config
from src.geocode import centroid
from src.regions import compute_bins

LONDON_CENTRE = [51.5074, -0.1278]

# Keyless, neutral light-grey basemap so the coloured regions stand out
# (CartoDB tiles now require an API key).
_BASEMAP_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
_BASEMAP_ATTR = "Tiles © Esri — Esri, DeLorme, NAVTEQ"

# ColorBrewer, chosen to be colourblind-safe.
_SEQUENTIAL = ["#ffffd9", "#c7e9b4", "#41b6c4", "#1d91c0", "#225ea8", "#081d58"]
_DIVERGING = ["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"]

_COLOR_FIELDS = {
    "level": ("median_price", "Typical flat price (£)"),
    "yoy": ("yoy_pct", "Change 2025 → 2026 (%)"),
    "vs_median": ("vs_median_pct", "Difference from London median (%)"),
}


def _fmt_gbp(value: float) -> str:
    """Compact money label, e.g. 245000 -> '£245k', 1200000 -> '£1.2m'."""
    if abs(value) >= 1_000_000:
        return f"£{value / 1_000_000:.1f}m"
    return f"£{value / 1_000:.0f}k"


def _colormap(
    stats: pd.DataFrame, color_by: str
) -> tuple[Any, str, str, list[float] | None, list[str] | None]:
    """Return (colormap, field, caption, bins, class_colors).

    ``bins``/``class_colors`` are set only for the sequential 'level' scale,
    which gets a discrete swatch legend instead of a crowded colorbar.
    """
    field, caption = _COLOR_FIELDS[color_by]
    values = stats[field].to_numpy(dtype=float)
    if color_by == "level":
        bins = compute_bins(values, k=5)
        cmap: Any = LinearColormap(_SEQUENTIAL, vmin=bins[0], vmax=bins[-1]).to_step(index=bins)
        cmap.caption = caption
        class_colors = [cmap((bins[i] + bins[i + 1]) / 2) for i in range(len(bins) - 1)]
        return cmap, field, caption, bins, class_colors
    bound = max(abs(float(values.min())), abs(float(values.max())), 1.0)
    cmap = LinearColormap(_DIVERGING, vmin=-bound, vmax=bound)
    cmap.caption = caption
    return cmap, field, caption, None, None


def _discrete_legend(bins: list[float], class_colors: list[str], caption: str) -> MacroElement:
    """A compact bottom-left legend with one equal-width swatch per price band."""
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;white-space:nowrap;">'
        f'<span style="background:{class_colors[i]};width:14px;height:14px;'
        f'display:inline-block;margin-right:6px;border:1px solid #999;"></span>'
        f"{_fmt_gbp(bins[i])} – {_fmt_gbp(bins[i + 1])}</div>"
        for i in range(len(class_colors))
    )
    html = (
        '<div style="position:absolute;bottom:18px;left:12px;z-index:9999;'
        "background:rgba(255,255,255,0.92);padding:8px 10px;border-radius:6px;"
        'font:12px/1.3 system-ui,sans-serif;color:#222;box-shadow:0 1px 4px rgba(0,0,0,0.3);">'
        f'<div style="font-weight:600;margin-bottom:4px;">{caption}</div>{rows}</div>'
    )
    legend = MacroElement()
    legend._template = Template("{% macro html(this, kwargs) %}" + html + "{% endmacro %}")
    return legend


def _annotate(geojson: dict[str, Any], stats: pd.DataFrame) -> dict[str, Any]:
    """Copy the GeoJSON and inject per-borough stat values into properties."""
    lookup = stats.set_index("borough").to_dict("index")
    annotated = copy.deepcopy(geojson)
    for feature in annotated["features"]:
        props = feature.setdefault("properties", {})
        name = props.get("borough") or props.get("name")
        props["borough"] = name
        row = lookup.get(name, {})
        props["median_price"] = f"£{row.get('median_price', 0):,.0f}"
        props["yoy_pct"] = f"{row.get('yoy_pct', 0):+.1f}%"
        props["vs_median_pct"] = f"{row.get('vs_median_pct', 0):+.1f}%"
    return annotated


def _add_district_layer(fmap: folium.Map, district_prices: pd.DataFrame, color_by: str) -> None:
    """Add a toggleable layer of per-district dots.

    Coloured by the 2025 → 2026 change (diverging) when comparing years and the
    change is available, otherwise by the predicted price (sequential).
    """
    change = color_by == "yoy" and "change_pct" in district_prices.columns
    value_col = "change_pct" if change else "price"
    values = district_prices[value_col].to_numpy(dtype=float)
    if change:
        bound = max(abs(float(values.min())), abs(float(values.max())), 1.0)
        dmap = LinearColormap(_DIVERGING, vmin=-bound, vmax=bound)
        name = "District change"
    else:
        dmap = LinearColormap(_SEQUENTIAL, vmin=float(values.min()), vmax=float(values.max()))
        name = "District prices"
    group = folium.FeatureGroup(name=name, show=True)
    for district, borough, lat, lon, value in zip(
        district_prices["district"],
        district_prices["borough"],
        district_prices["lat"],
        district_prices["lon"],
        district_prices[value_col],
        strict=False,
    ):
        v = float(value)
        text = f"{v:+.1f}%" if change else f"£{v:,.0f}"
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=5,
            color="#444444",
            weight=0.5,
            fill=True,
            fill_color=dmap(v),
            fill_opacity=0.9,
            tooltip=f"{district} · {borough}: {text}",
        ).add_to(group)
    group.add_to(fmap)


def build_map(
    geojson: dict[str, Any],
    stats: pd.DataFrame,
    color_by: str,
    selected_borough: str | None,
    marker_district: str | None,
    marker_price: float | None = None,
    district_prices: pd.DataFrame | None = None,
) -> folium.Map:
    """Build the choropleth with borders, fill, legend, tooltips, and a marker.

    The marker shows only the predicted point price (the range stays in the
    text beside the map, to keep the map uncluttered). When ``district_prices``
    is given, a toggleable layer of per-district price dots is added.
    """
    cmap, field, caption, bins, class_colors = _colormap(stats, color_by)
    color_lookup = stats.set_index("borough")[field].to_dict()
    annotated = _annotate(geojson, stats)

    fmap = folium.Map(
        location=LONDON_CENTRE, zoom_start=10, tiles=_BASEMAP_TILES, attr=_BASEMAP_ATTR
    )

    def style(feature: dict[str, Any]) -> dict[str, Any]:
        name = feature["properties"].get("borough")
        value = color_lookup.get(name)
        selected = name == selected_borough
        return {
            "fillColor": cmap(value) if value is not None else "#cccccc",
            "color": "#111111" if selected else "#666666",
            "weight": 3 if selected else 1,
            "fillOpacity": 0.75,
        }

    folium.GeoJson(
        annotated,
        style_function=style,
        highlight_function=lambda _f: {"weight": 3, "color": "#111111"},
        tooltip=folium.GeoJsonTooltip(
            fields=["borough", "median_price", "yoy_pct", "vs_median_pct"],
            aliases=["Borough", "Typical price", "YoY change", "Vs median"],
            sticky=True,
        ),
        name="boroughs",
    ).add_to(fmap)

    # A continuous colorbar for the diverging scales; a discrete swatch legend
    # for the sequential price level (its quantile edges would crowd a colorbar).
    if bins is not None and class_colors is not None:
        fmap.get_root().add_child(_discrete_legend(bins, class_colors, caption))
    else:
        cmap.add_to(fmap)

    if district_prices is not None and not district_prices.empty:
        _add_district_layer(fmap, district_prices, color_by)

    if marker_district:
        coord = centroid(marker_district)
        if coord:
            label = f"{marker_district} · {selected_borough}"
            if marker_price is not None:
                label += f" — £{marker_price:,.0f}"
            folium.Marker(
                location=list(coord),
                tooltip=label,
                popup=folium.Popup(label, max_width=250),
                icon=folium.Icon(color="darkblue", icon="home", prefix="fa"),
            ).add_to(fmap)
            folium.PolyLine(
                [list(coord), list(config.CENTRAL_LONDON)],
                color="#333333",
                weight=1,
                dash_array="5",
                opacity=0.5,
            ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def render(fmap: folium.Map, key: str = "map") -> dict[str, Any] | None:
    """Render the map in Streamlit and return the last interaction payload."""
    result = st_folium(fmap, width=None, height=520, key=key, returned_objects=["last_clicked"])
    return result.get("last_clicked") if result else None
