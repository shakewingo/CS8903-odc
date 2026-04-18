# Build context: repo root (CS8903-odc/)
# docker build -f web-app/backend/Dockerfile -t cs8903-backend .

FROM python:3.11-slim

# Install GDAL and build dependencies for rasterio, then clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (separate layer for caching)
COPY web-app/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip

# Remove build-only deps to shrink image
RUN apt-get purge -y gcc g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Mirror repo layout so PROJECT_ROOT (.parent.parent.parent) resolves to /app/
COPY src/ ./src/
COPY web-app/backend/ ./web-app/backend/

# Data files
COPY data/processed/malawi_masked_2024.tif ./data/processed/malawi_masked_2024.tif
COPY data/processed/lake_malawi_expanded_25km.json ./data/processed/lake_malawi_expanded_25km.json
COPY data/processed/rl_dataset.npz ./data/processed/rl_dataset.npz

# Pre-trained models
COPY models/6f0ta58i/model.zip ./models/6f0ta58i/model.zip
COPY models/2sk0pnp3/model.zip ./models/2sk0pnp3/model.zip
COPY models/pzy2mxod/model.zip ./models/pzy2mxod/model.zip

ENV PYTHONPATH="/app:/app/src"

EXPOSE 8000

CMD ["python", "-c", "import sys; sys.path.insert(0, 'web-app'); from backend.main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"]
