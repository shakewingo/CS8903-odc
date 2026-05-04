# Build context: repo root (CS8903-odc/)
# docker build -f web-app/backend/Dockerfile -t cs8903-backend .

FROM python:3.13-slim

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

# Data files (GeoTIFF, dataset, boundary polygon, ET values)
COPY data/processed/malawi_masked_2024.tif ./data/processed/malawi_masked_2024.tif
COPY data/processed/lake_malawi_expanded_25km.json ./data/processed/lake_malawi_expanded_25km.json
COPY data/processed/rl_dataset.npz ./data/processed/rl_dataset.npz
COPY data/processed/et_per_landcover_2024.json ./data/processed/et_per_landcover_2024.json

# Pre-trained models
COPY models/n2947wpo/model.zip ./models/n2947wpo/model.zip
COPY models/umgje44z/model.zip ./models/umgje44z/model.zip
COPY models/7ybfz89t/model.zip ./models/7ybfz89t/model.zip

ENV PYTHONPATH="/app:/app/src"

# Flush prints/logs immediately so we don't lose the last few lines if
# the worker dies (faulthandler writes to stderr but pipe buffering can
# eat it).
ENV PYTHONUNBUFFERED=1

# Memory hygiene on small Linux instances. Must be set before the Python
# process starts so glibc reads them at init.
ENV MALLOC_ARENA_MAX=2
ENV MALLOC_TRIM_THRESHOLD_=131072
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

EXPOSE 8000

CMD ["python", "-c", "import sys; sys.path.insert(0, 'web-app'); from backend.main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"]
