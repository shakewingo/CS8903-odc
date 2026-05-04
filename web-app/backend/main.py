"""FastAPI backend for dynamic grid generation and model inference."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Resolve paths so the server can be started from any directory:
#   cd web-app/backend && uvicorn main:app --port 8000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # web-app/backend → web-app → CS8903-odc
WEBAPP_DIR = str(PROJECT_ROOT / "web-app")
SRC_DIR = str(PROJECT_ROOT / "src")
for p in (str(PROJECT_ROOT), WEBAPP_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.grid_service import router as grid_router
from backend.infer_service import router as infer_router, load_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ML models on startup."""
    load_models()
    yield


app = FastAPI(
    title="CS8903 ODC Backend",
    description="Dynamic grid generation and RL model inference for Lake Malawi land-use optimization",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Vercel frontend and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://cs-8903-odc.vercel.app",
        "https://*.onrender.com",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(grid_router, prefix="/api")
app.include_router(infer_router, prefix="/api")


_proc = psutil.Process()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rss_mb": round(_proc.memory_info().rss / 1e6, 1),
    }
