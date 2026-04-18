# RL-Driven Sustainable Land-Use Allocation for the Lake Malawi Basin

Optimizing ecosystem service values in the Lake Malawi Basin through deep reinforcement learning.

**Demo:** https://cs-8903-odc.vercel.app/

**Paper:** [arXiv:2604.03768](https://arxiv.org/abs/2604.03768)

## Project Structure

```
CS8903-odc/
  src/                    # Core Python modules
    config.py             # Constants: center coords, grid params, ESV/ET values
    dataset.py            # Build 50x50 grid from GeoTIFF, split train/test
    train.py              # LandUseEnv (Gymnasium) + MaskablePPO training
    eval.py               # Model inference, reconstruct_map, diff analysis
    eval_zoom.py          # Zoom-in comparison figure for paper
    post_eda.py           # GeoTIFF processing, value grids, heatmap plotting
    utils.py              # Logger, MinMax normalization
  data/
    processed/            # GeoTIFFs, rl_dataset.npz, ET/ESV JSON
    raw/                  # Lake Malawi GeoJSON boundary
  models/                 # Pre-trained MaskablePPO checkpoints (per W&B run)
  notebook/               # Jupyter notebooks for EDA and land cover analysis
  web-app/
    frontend/             # Next.js 16 PWA (Vercel)
    backend/              # FastAPI backend (Render)
```

## Environment Setup

```bash
# 1. Create conda environment
conda create -n cs8903 python=3.13 -y
conda activate cs8903

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install sb3-contrib (for MaskablePPO, not in root requirements.txt)
pip install sb3-contrib torch
```

## Running the ML Pipeline

All scripts are run from the project root with the conda env activated.

```bash
conda activate cs8903
```

### Generate Dataset

Build the 50x50 land-cover grid from Sentinel-2 GeoTIFF:

```bash
python src/dataset.py
```

Outputs `data/processed/rl_dataset.npz` with pixel counts, ESV, ET, train/test splits.

### Train Model

Train a MaskablePPO agent with spatial rewards:

```bash
python src/train.py --spatial-scale 1.0 --total-timesteps 500000
```

Key flags: `--spatial-scale` (0 = eco-only), `--w-tree`, `--w-crop`, `--w-built`, `--w-buf`, `--reward-scale`. Models save to `models/<wandb_run_id>/model.zip`.

### Evaluate Model

Run inference and generate before/after comparison plots:

```bash
python src/eval.py --model-path models/<run_id>/model.zip --deterministic
```

Flags: `--split train|test|both`, `--spatial-scale` (must match training), `--no-plot`.

### Zoom-In Visualization

Generate the zoom-in comparison figure for the paper:

```bash
python src/eval_zoom.py
```

## Running the Web App Locally

### Backend (FastAPI)

```bash
conda activate cs8903
pip install fastapi "uvicorn[standard]"   # one-time

cd web-app/backend
uvicorn main:app --port 8000 --reload
```

Backend serves at http://localhost:8000. API docs at http://localhost:8000/docs.

### Frontend (Next.js)

```bash
cd web-app/frontend
npm install   # one-time
npm run dev -- --webpack
```

Frontend serves at http://localhost:3000. Loads static fallback data on start, calls backend API when you click the map.

## Production

| Service  | URL                                    | Platform       |
|----------|----------------------------------------|----------------|
| Frontend | https://cs-8903-odc.vercel.app         | Vercel |
| Backend  | https://cs8903-odc.onrender.com        | Render |

## Experiments

| Experiment | Config | Run ID |
|------------|--------|--------|
| Exp I      | Pure eco-value (`spatial_scale=0`) | `6f0ta58i` |
| Exp II     | Eco + spatial rewards (`spatial_scale=1.0`) | `2sk0pnp3` |
| Exp III    | Spatial + regenerative agriculture (1.35x crops) | `pzy2mxod` |
