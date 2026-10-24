---
title: Greater London Apartment Price
emoji: 🏠
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
app_port: 7860
pinned: false
license: apache-2.0
---

# greater_london_apartment_price

A map-first Streamlit app, deployed as a Hugging Face Space, that predicts the price of a
Greater London apartment (flat) and shows it in context on an interactive choropleth of London.

Pick a location by clicking a borough or using the searchable selectors (the postcode-district
list is scoped to the chosen borough), set a few attributes (tenure, new-build, year), and get a
predicted price with a calibrated range, a SHAP explanation, and a London-wide map coloured by
price (or by price differences), with an optional per-postcode-district price layer on top.

## Honest framing

The estimate is a **range, not a single figure**. It is built on HM Land Registry Price Paid Data —
**actual sold prices**, which is the right target — but that data has **no floor area, bedrooms, or
bathrooms**. So the model predicts the **typical price of a flat in a given location, tenure, and
period**, not the price of one specific unit. That is exactly why the app leads with a map and a
conformal range rather than a lone number.

## How it works

- **Model:** XGBoost on log price, trained on pooled 2025 + 2026 Greater London flats with a
  random 60/20/20 train/calibrate/test split (so both years are balanced in training, which lets
  the model learn the year effect and satisfies the exchangeability split conformal assumes).
- **Range:** conformal prediction (split + adaptive/Mondrian), ~90% coverage.
- **Explanation:** Tree SHAP per prediction.
- **Map:** Folium choropleth of the 33 boroughs on a keyless neutral basemap, coloured from
  precomputed per-borough stats (price level, year-over-year change, difference from the London
  median), colourblind-safe, with a discrete swatch legend for price level, a diverging colourbar
  for the difference layers, and numeric hover tooltips. The selected location drops a pin showing
  the predicted price. A toggleable **per-postcode-district layer** draws a dot per district,
  coloured by predicted price — or by the 2025 → 2026 change when comparing years.
- **Geocoding:** crow-flies distance uses geopy; postcode-district centroids are precomputed with
  the keyless [postcodes.io](https://postcodes.io) API and committed, so the Space makes no live
  geocoding calls (only map basemap tiles are fetched by the browser, as any web map does).

## Data & attribution

- **Sale prices:** HM Land Registry Price Paid Data, Open Government Licence.
- **Borough boundaries:** ONS Open Geography Portal (generalised Local Authority Districts),
  Open Government Licence.
- **Postcode-district centroids:** [postcodes.io](https://postcodes.io), derived from ONS/OS open
  data (Open Government Licence).
- **Basemap:** Esri Light Gray Canvas (keyless).

## Develop and retrain

Python 3.11+ (the repo pins 3.12 via `.python-version` for the scientific wheels).

```bash
make setup        # install uv + sync dependencies
make data         # download Land Registry 2025 + 2026, filter to London flats
make boundaries   # fetch generalised borough GeoJSON from ONS Open Geography Portal
make centroids    # build postcode-district centroids via postcodes.io (keyless)
make train        # XGBoost + conformal + region stats + valid categories -> models/model.pkl
make run          # streamlit run app.py
make ci           # lint + typecheck + tests, offline on the frozen sample
```

The committed map assets are **real** (ONS borough boundaries, postcodes.io centroids). The
frozen `data/sample/` uses **real postcode districts with synthetic prices**, so `make ci` and
`make run` work fully offline and self-consistently; run `make data` then `make train` to build a
real-price model from the full Land Registry download. See [CLAUDE.md](CLAUDE.md) for the full
architecture.

### Manual upload checklist (Hugging Face Space)

`app.py`, `src/`, `ui/`, `models/model.pkl`, `config/` (GeoJSON, region stats, centroids, YAMLs),
`requirements.txt`, `Dockerfile`, and this `README.md`.
