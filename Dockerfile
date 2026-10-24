# Hugging Face Space (Docker SDK). Serves only — the model is trained offline
# and the artifact plus config assets are committed, so the Space makes no
# external calls. Mirrors the developer_salary_prediction Space.
FROM python:3.11-slim

WORKDIR /app

# System deps for the scientific wheels (xgboost/shap) and matplotlib fonts.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces route to port 7860.
EXPOSE 7860
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
