.PHONY: help setup sync data check-schema boundaries centroids train lint format check typecheck test run clean ci

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-extras
	@echo "\n✅ Ready. Run 'make data' 'make boundaries' 'make centroids' 'make train' 'make run'."

sync: ## Sync deps
	uv sync --all-extras

data: ## Download Land Registry 2025 + 2026, filter to Greater London flats, cache
	uv run python -m src.download_data

check-schema: ## Verify the 16 PPD columns and the London-flat filter
	uv run python -m src.check_schema

boundaries: ## Download + simplify London borough GeoJSON into config/london_boroughs.geojson
	uv run python -m src.regions --build-boundaries

centroids: ## Build postcode-district -> lat/lon lookup (crow-flies geopy/Nominatim), commit it
	uv run python -m src.geocode --build

train: ## Train XGBoost, calibrate conformal, write model + region_stats + valid_categories
	uv run python -m src.train
	@echo "✅ Model, conformal calibration, region stats, and configs written."

lint: format check typecheck ## All linters
format: ; uv run ruff format src/ ui/ tests/ app.py
check: ; uv run ruff check --fix src/ ui/ tests/ app.py
typecheck: ; uv run ty check src/ ui/

test: ## pytest (offline on the frozen sample)
	uv run pytest

run: ## Streamlit app locally
	uv run streamlit run app.py --server.port 8501

clean: ## Clean caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache .ruff_cache

ci: sync lint test ## CI (offline on the frozen sample)
