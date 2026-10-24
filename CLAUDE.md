# CLAUDE.md — greater_london_apartment_price

Guidance for Claude Code (and humans) working in this repo. Read this before writing code.
The authoritative product spec is [.llm/PRD_greater_london_apartment_price.md](.llm/PRD_greater_london_apartment_price.md); this file is the working summary. When they disagree, the PRD wins — update this file to match.

---

## 1. What this is

A **map-first** Streamlit app, deployed as a free **Hugging Face Space**, that predicts the price of a Greater London flat and shows it in context on an interactive **choropleth** of London. Pick a location (click a borough or use searchable selectors), set a few attributes (tenure, new-build, year), and get a **predicted price with a calibrated range**, a SHAP explanation, and a London-wide map colored by price (or by price differences).

The backend deliberately mirrors the **`developer_salary_prediction`** Space: a thin UI over `src/infer.py` + `src/schema.py`, a `src/train.py` that writes `models/model.pkl` + `config/valid_categories.yaml`, and the same **uv / Makefile / pydantic / Apache-2.0** conventions. The new work is the **usability of the map UI**.

Reused patterns from sibling projects:
- **Conformal range** (split + adaptive/Mondrian) from `prediction_intervals_arena` — implemented from scratch, **no `mapie`**.
- **Keyless geocoding** (crow-flies geopy for distance) from `geo_distance_arena`; postcode-district centroids are precomputed via the keyless **postcodes.io** API and committed.
- **OOF target encoding** idea from the encoding project — no self-leak.

### The one thing that matters: honesty

Land Registry records **actual sold prices** (the right target) but has **no floor area, bedrooms, or bathrooms**. So the model predicts the **typical price of a flat in a given location, tenure, and period** — *not* a specific unit. The app leads with a **map + a range**, never a lone confident number, precisely because a point estimate would overclaim. Keep this framing in every surface you touch (copy, tooltips, the "about & caveats" panel).

---

## 2. Golden rules

- **Map FIRST, not a form.** Left sidebar = controls; main area = map + result. Detail lives in **expanders** (progressive disclosure).
- **No live data/geocoding calls on the Space.** Training data, borough GeoJSON, region stats, and postcode centroids are **precomputed and committed**; the served app loads artifacts only. (The browser still fetches map **basemap tiles** from a keyless CDN, as any web map does.)
- **Color never carries meaning alone.** Every map layer ships with a legend/colorbar and numeric hover tooltips; palettes are **colorblind-safe**. Price level uses a **discrete swatch legend** (quantile bins would crowd a colorbar); the difference layers use a diverging colorbar.
- **Predict on LOG price**, exponentiate back — predictions are never negative and errors are multiplicative.
- **The range is the message**, not fine print. Headline reads like "typically £520,000, with a 90% range of £430,000–£620,000 for comparable flats".
- **Session state is sacred.** `st_folium` reruns the script on every interaction; all selection state lives in `ui/state.py` so clicks and selectors never silently reset. This is the #1 known pitfall.
- **Offline-first.** `make test` and CI run on the **frozen sample** in `data/sample/` with **no network**.
- **Reproducible.** Seed training; assert the London-flat filter, OOF non-leak, conformal coverage, region-stats completeness, and a valid `lo <= point <= hi` range.

---

## 3. Architecture & data flow

```
download_data.py ──► raw PPD CSVs (pp-2025, pp-2026)  [keyless, cached + frozen sample]
        │
check_schema.py ──► assert 16 PPD columns + London-flat filter (type F, GREATER LONDON, PPD A)
        │
preprocessing.py ─► filter → features → one-hot (borough/tenure/new-build/year-month)
        │                                + OOF target encoding (postcode district)
        ▼
train.py ─────────► XGBoost on log price  ──► models/model.pkl {model, feature_columns,
        │                                                        target_encoding, conformal}
        ├─ intervals.py  → split + adaptive (Mondrian) conformal calibration
        ├─ explain.py    → Tree SHAP
        └─ regions.py    → config/region_stats.csv  (per-borough median/predicted/YoY/vs-median)

SERVE (app.py, no training):
  ui/controls.py ─┐
  ui/map_view.py ─┼─► ui/state.py (two-way sync) ─► src/infer.predict_price()  → point + range + SHAP
  ui/map_view.py ─┘                              └─► src/infer.predict_districts() → per-district dots
  regions.load_stats() feeds the borough choropleth; geocode.centroid/nearest_district drive click-to-select
```

### Module responsibilities (`src/`)
| File | Role |
|---|---|
| `config.py` | Paths, constants, load YAML configs |
| `download_data.py` | Fetch `pp-2025.csv` / `pp-2026.csv` (keyless), cache |
| `check_schema.py` | Confirm 16 PPD columns + London-flat filter |
| `preprocessing.py` | Filter to Greater London flats; features; one-hot + **OOF** target encoding |
| `schema.py` | `PropertyInput` (pydantic): borough, postcode district, tenure, new-build, year, month |
| `train.py` | XGBoost on log price; calibrate conformal; write model + region stats + valid_categories |
| `intervals.py` | Split + adaptive (Mondrian) conformal (from `prediction_intervals_arena`, from scratch) |
| `infer.py` | `predict_price` → point + range + SHAP; `predict_districts` (batched, all districts); validates categories; `get_fx` |
| `explain.py` | Tree SHAP values for a single prediction |
| `geocode.py` | Postcode district → lat/lon; `nearest_district`; geodesic `distance_to_centre` (geopy); `--build` refreshes centroids via **postcodes.io** |
| `regions.py` | Load borough GeoJSON; build per-region stats; choropleth bins; `load_stats`; `--build-boundaries` fetches from the **ONS Open Geography Portal** |

### UI responsibilities (`ui/`)
| File | Role |
|---|---|
| `map_view.py` | Folium choropleth: borders, sequential + diverging fills, discrete legend / colorbar / tooltips, click, priced marker, toggleable per-district dot layer (keyless Esri basemap) |
| `controls.py` | Sidebar selectors: searchable **borough-scoped** district select, toggles, radios, year, color-by (disabled while comparing), reset |
| `state.py` | Session-state sync between map clicks and selectors — never let them disagree |

---

## 4. The map (`ui/map_view.py`) — Folium via `streamlit-folium`

Render with `st_folium` on the **keyless Esri Light Gray Canvas** basemap (`_BASEMAP_TILES`) so colored regions stand out. CartoDB tiles now require an API key — don't use them.

- **Region borders** — 33 boroughs, from `config/london_boroughs.geojson` (ONS generalised, OGL, committed, ~260 KB).
- **Color by price (sequential)** — perceptually uniform, colorblind-safe, darker = dearer; **quantile bins** (`regions.compute_bins`). Shown with a **discrete swatch legend** (`_discrete_legend`, one equal-width band per class) — not a colorbar, whose clustered tick labels crowd.
- **Color by difference (diverging)** — colorblind-safe, **true neutral midpoint**, with a branca colorbar. Two modes: (a) YoY 2025→2026 % change (neutral = 0%), (b) vs London median (neutral = median).
- **Per-district dots** (`_add_district_layer`, toggleable via `LayerControl`) — one dot per postcode-district centroid from `infer.predict_districts` (one batched model call). Colored by **predicted price** (sequential), or by **2025→2026 change** (diverging) when comparing years; hover shows the district, borough, and value.
- **Click to select** — clicking a borough updates the selectors, drops a marker, and re-predicts. `st_folium` returns the click; `ui/state.py` reconciles it, then `st.rerun()`.
- **Chosen point** — marker at the selected centroid showing **only the predicted point price** (the range stays in the text beside the map, to keep the map uncluttered); faint geodesic line to central London (the crow-flies nod).
- **Borough colors come from precomputed `config/region_stats.csv`** — **never per-polygon model calls**. The per-district dots are the only serve-time model calls for the map (batched, cheap).

---

## 5. Selectors (`ui/controls.py`) & result

- **Location** — click the map, or a searchable **borough** select → searchable **postcode-district** select **scoped to that borough** (`state.districts_for`, which keys off `catalog["district_to_borough"]`). Two-way synced via `ui/state.py`.
- **Tenure** — radio, **Leasehold** default (most flats are leasehold).
- **New build** — toggle, **off** default.
- **Year** — radio (2025 / 2026) + a **"Compare 2025 and 2026"** checkbox.
- **Color by** — price level / YoY change / vs-median. **Never mutate `color_by` based on compare** — that was a bug (the YoY layer got stuck on). The checkbox *disables* this select; the map's effective mode is computed in `app.py` as `"yoy" if compare else color_by`, so unchecking reverts cleanly.
- **Reset** — returns every control and the map to sensible defaults.
- **First load shows a real colored map + a real estimate with no input.**

**Result:** prominent price metric with the conformal range beside it (price **and** range in the text left of the map). Below the map, expanders hold: SHAP "why this price", 2025-vs-2026 compare (both-year prediction + % change), GBP→EUR/USD conversion, and "about & caveats" (sold-price basis, no-floor-area limit, conformal-exchangeability note).

---

## 6. Model & conformal range

- XGBoost on **log price**; exponentiate back to GBP. Artifact: `{model, feature_columns, target_encoding, target_global_mean, median_distance, conformal}`.
- **Random 60/20/20 train/calibrate/test split** (`train_test_split`, seeded). This is deliberate: a chronological split starved training of 2026 (all 2025 + only early 2026), so the model learned a **zero year effect** and 2025 == 2026 predictions. A random split balances both years (year becomes learnable) **and** satisfies the exchangeability split conformal assumes. Don't revert to a chronological split without fixing the year-learnability problem.
- **Split conformal** = baseline (quantile of absolute log-residuals on the calibration slice; wraps the model, no retraining). **Adaptive / Mondrian conformal** = upgrade (quantile per borough → interval widens in volatile areas, keeps the guarantee).
- Coverage lands **near 90%** on the held-out test slice (printed by `train.py`). Caveat to keep in the app: conformal assumes exchangeability, so a sharp market move can drift it.
- Baseline to beat: a **borough-median** predictor.

---

## 7. Data & features

- **Sale prices:** HM Land Registry Price Paid Data (keyless CSV, no header, 16 columns), Open Government Licence.
  - 2025: `http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2025.csv`
  - 2026: `http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2026.csv`
  - Filter: property type **`F`** (flats), county **`GREATER LONDON`**, PPD category **`A`**, sensible price bounds. Date → year + month.
- **Borough boundaries:** **ONS Open Geography Portal** (generalised Local Authority Districts, GeoJSON, OGL), filtered to the 33 London boroughs by `regions._build_boundaries` — no shapefile reader / geopandas needed.
- **Postcode-district centroids:** **postcodes.io** (keyless), built by `geocode._build` from the outward codes seen in the data, with a borough-centroid fallback for any that don't resolve.
- **Borough name alias:** HM Land Registry writes `"CITY OF WESTMINSTER"`; ONS/the GeoJSON use `"Westminster"`. `config.normalize_borough` maps it via `_BOROUGH_ALIASES` — without this, Westminster's districts don't match the selector. It's the only such mismatch.
- **Features:** borough **one-hot**; postcode district (outward code) **OOF target encoding** (no self-leak); tenure **one-hot**; new-build; year + month; geodesic **distance-to-centre** from the centroid.
- **Offline:** committed GeoJSON / region stats / centroids are **real**; `data/sample/` uses **real outward codes with synthetic prices**. So CI is offline and self-consistent (sample districts ⊆ committed centroids). Run `make data` then `make train` for a real-price model.

---

## 8. Commands (Makefile)

```
make setup        # first-time: install uv, uv sync --all-extras
make data         # download PPD 2025+2026, filter to London flats, cache
make check-schema # verify 16 PPD columns + London-flat filter
make boundaries   # fetch generalised borough GeoJSON from ONS Open Geography Portal
make centroids    # build postcode-district → lat/lon lookup via postcodes.io (keyless), commit it
make train        # XGBoost + conformal + region_stats + valid_categories → models/model.pkl
make lint         # ruff format + ruff check --fix + ty typecheck
make test         # pytest, offline on the frozen sample
make run          # streamlit run app.py --server.port 8501
make ci           # sync + lint + test (offline)
```

Pipeline order for a fresh build: `setup → data → check-schema → boundaries → centroids → train → run`.

---

## 9. Conventions

- **Python 3.11+** (repo pins **3.12** via `.python-version` — scientific wheels lag on 3.14), line length 99, ruff (`E,F,W,I,UP,N,B,A,SIM,PTH`; `E501` ignored), `ty` for typing, pytest (`pythonpath = ["."]` so `from src import …` resolves).
- **Prefer `pathlib`** over `os.path` (ruff `PTH`).
- **`ty` is strict about third-party/dynamic access** — e.g. don't index nested JSON dicts it can't narrow, and keep `pandas`/`folium` access simple. When unsure, `make typecheck` locally.
- **uv** for dev; keep **`requirements.txt` minimal** for the Space (it mirrors `pyproject.toml` runtime deps, no dev extras, includes `streamlit-folium`, `folium`, `branca`, `geopy`).
- Validate all categoricals in `infer.py` against `config/valid_categories.yaml`; reindex feature frames to `feature_columns`, filling missing with 0.
- Keep `app.py` **thin** — orchestration only; logic lives in `src/` and `ui/`.
- Apache-2.0 headers / license already in place.

## 10. Tests (must stay green, offline)

`tests/conftest.py` trains once per session on the frozen sample into a temp dir, so the suite is self-contained and needs no committed binary. 23 tests across:

| Test | Asserts |
|---|---|
| `test_download.py` | London-flat filter keeps only type `F` in Greater London |
| `test_preprocessing.py` | OOF target encoding does not see the row's own price; `encode_matrix` reindexes |
| `test_intervals.py` | Split conformal covers near nominal; Mondrian widens for volatile groups |
| `test_regions.py` | Every borough polygon has stats; choropleth bins valid |
| `test_geocode.py` | Every valid postcode district has a centroid; clicks snap to the nearest district |
| `test_infer.py` | `predict_price` returns positive GBP with `lo <= point <= hi`; `predict_districts` prices every district |
| `test_schema.py` | `PropertyInput` validates + normalizes categories against the catalog |

## 11. Deployment (Hugging Face)

Space mirrors the salary project. `README.md` carries HF front-matter (`sdk: docker` + committed `Dockerfile`, or `sdk: streamlit`), `requirements.txt` lists runtime deps. The Space serves `models/model.pkl` + `config/` assets + `app.py` + `src/` + `ui/`. **Training is offline in the Codespace; the Space serves only** and fits a free CPU Space. Keep a manual-upload checklist of exactly which files to push.

## 12. Constraints

2-CPU / 8 GB Codespace for dev + training; free CPU Space for serving. No GPU, no paid API, no registration/keys. Read PPD CSVs with the London-flat filter applied **early** to stay within 8 GB.
