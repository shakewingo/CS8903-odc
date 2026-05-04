"""Model inference endpoint.

Thin wrapper around ``src.eval``: ``make_env()``, ``run_inference()``,
``reconstruct_map()``, ``compute_diff_cells()``.

Memory-optimized: models are loaded lazily on first request and only
one model is kept in memory at a time (fits in Render Starter 512MB).
"""

import gc
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import src.train as train_module
from src.config import ECO_VALUES, SEED
from src.eval import (
    build_value_vecs, make_env, run_inference,
    reconstruct_map, compute_diff_cells,
)
from src.train import augment_data
from src.utils import get_logger
from backend.formatting import build_results_json

logger = get_logger("infer_service", level="INFO")
router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# run_id per experiment — must match training configs.
# All three checkpoints were trained with riparian_mask=True and the
# regen-crop default ECO_VALUES from src/config.py.
EXPERIMENTS = {
    "exp1": "n2947wpo",
    "exp2": "umgje44z",
    "exp3": "7ybfz89t",
}

REGEN_CROP_MULTIPLIER = 1.35

_proc = psutil.Process()


def _log_rss(label: str):
    rss_mb = _proc.memory_info().rss / 1e6
    logger.info(f"RSS {rss_mb:7.1f} MB — {label}")


# Only one model in memory at a time to fit in 512MB
_current_model = None       # (exp_id, model)
_initialized = False


def _init_once():
    """Initialize logger + value vectors on first use (not at startup)."""
    global _initialized
    if _initialized:
        return
    train_module.logger = get_logger("train", level="WARNING")
    build_value_vecs(use_et=False, assign_globals=True)
    _initialized = True
    logger.info("Value vectors initialized")


def load_models():
    """Lightweight startup — just validate model files exist.
    Actual model loading happens lazily on first request.
    """
    _init_once()
    for exp_id, run_id in EXPERIMENTS.items():
        path = PROJECT_ROOT / "models" / run_id / "model.zip"
        status = "OK" if path.exists() else "MISSING"
        logger.info(f"Model {exp_id} (run {run_id}): {status}")


def _get_model(experiment: str):
    """Load model for the requested experiment.
    Evicts the previous model to save memory.
    """
    global _current_model
    _log_rss(f"_get_model({experiment}) start")
    from sb3_contrib import MaskablePPO
    _log_rss("_get_model after sb3 import")

    _init_once()

    # Already loaded
    if _current_model and _current_model[0] == experiment:
        return _current_model[1]

    # Evict previous model
    if _current_model:
        logger.info(f"Evicting model {_current_model[0]} from memory")
        _current_model = None
        gc.collect()
        _log_rss("_get_model after eviction+gc")

    run_id = EXPERIMENTS[experiment]
    model_path = PROJECT_ROOT / "models" / run_id / "model.zip"
    if not model_path.exists():
        raise HTTPException(status_code=500, detail=f"Model file not found for {experiment}")

    dummy_env = make_env(SimpleNamespace(riparian_mask=True), "test_indices")
    _log_rss("_get_model after dummy_env")
    model = MaskablePPO.load(str(model_path), env=dummy_env)
    _log_rss("_get_model after MaskablePPO.load")
    _current_model = (experiment, model)
    logger.info(f"Loaded model for {experiment} (run {run_id})")
    return model


def _make_ppo_act_fn(model, deterministic=True):
    """Wrap a MaskablePPO model in the (env, ep) -> act_fn factory shape
    expected by src.eval.run_inference (see src/eval_zoom.py)."""
    def ppo_act(env):
        action, _ = model.predict(
            env.state,
            action_masks=env.action_masks(),
            deterministic=deterministic,
        )
        return int(action)
    return lambda env, ep: ppo_act


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class InferRequest(BaseModel):
    lat: float
    lng: float
    experiment: str


# ---------------------------------------------------------------------------
# Inference (cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _run_cached(lat_r: float, lng_r: float, experiment: str):
    """Run model inference following the ``eval.py:main()`` pattern."""
    from backend.grid_service import get_cached_dataset

    _log_rss(f"enter _run_cached ({experiment})")
    model = _get_model(experiment)
    _log_rss(f"after model load ({experiment})")
    dataset = get_cached_dataset(lat_r, lng_r)
    # All current checkpoints were trained with --riparian-mask; must match
    # at eval time so the policy sees the same action mask.
    env_args = SimpleNamespace(riparian_mask=True)
    make_act_fn = _make_ppo_act_fn(model, deterministic=True)

    # eval.py pattern: run both splits, merge results
    combined = {}
    for split in ("test_indices", "train_indices"):
        env = make_env(env_args, split)
        _log_rss(f"after make_env [{split}]")
        # make_env already called env.reset(), which built env.samples from
        # the static rl_dataset.npz that LandUseEnv loaded at __init__. For
        # the web app's per-request dataset, override env.samples so the
        # agent actually sees the user's clicked area instead of the
        # training-time area. (env.data is also re-pointed for completeness
        # in case anything downstream re-augments.)
        env.data = dataset
        env.samples = augment_data(
            dataset, size=env.size, seed=SEED, split=split, n_augment=0,
        )
        combined.update(
            run_inference(make_act_fn, env, dataset, split)
        )
        _log_rss(f"after run_inference [{split}]")

    # eval.py: reconstruct_map + compute_diff_cells
    initial_obs, final_obs = reconstruct_map(dataset, combined)
    changed_cells = compute_diff_cells(initial_obs, final_obs, atol=1e-4)
    _log_rss("after reconstruct_map")

    # ESV map (exp1 and exp2 uses non-regenerative crop multiplier)
    esv_map = dict(ECO_VALUES)
    if experiment != "exp3":
        esv_map[5] = round(esv_map[5] / REGEN_CROP_MULTIPLIER)

    return build_results_json(experiment, initial_obs, final_obs, changed_cells, esv_map)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post("/infer")
def infer(req: InferRequest):
    if req.experiment not in EXPERIMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown experiment '{req.experiment}'. "
                   f"Choose from: {list(EXPERIMENTS.keys())}",
        )

    lat_r = round(req.lat, 6)
    lng_r = round(req.lng, 6)
    logger.info(f"Running inference for {req.experiment} at ({lat_r}, {lng_r})")
    try:
        return _run_cached(lat_r, lng_r, req.experiment)
    except HTTPException:
        raise
    except BaseException as e:
        # Surface non-HTTP errors with a traceback so prod logs show what
        # raised (vs. the worker just dying with no signal).
        import traceback
        logger.error(
            f"/api/infer raised {type(e).__name__}: {e}\n"
            + traceback.format_exc()
        )
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
