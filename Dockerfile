FROM python:3.11-slim

WORKDIR /app

# libgomp1 required by LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY src/ ./src/
COPY config/ ./config/

# Champion model artefacts (built by the training pipeline)
COPY models/ ./models/
COPY data/processed/feature_cols.json ./data/processed/feature_cols.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
