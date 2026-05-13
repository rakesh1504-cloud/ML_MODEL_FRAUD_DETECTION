FROM python:3.11-slim

WORKDIR /app

# Prevent matplotlib font cache
ENV MPLBACKEND=Agg
ENV MPLCONFIGDIR=/tmp
ENV MATPLOTLIBRC=/tmp

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/feature_cols.json ./data/processed/feature_cols.json

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
