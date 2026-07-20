#!/bin/bash
#SBATCH --job-name=fe_forecast        # FlashEdges forecast / inference
#SBATCH --partition=gpu_h200            # CRIANN Austral H200 nodes
                                        #   alt (A100): gpu | hpda | gpu_all | gpu_debug
#SBATCH --gpus=1                        # 1 GPU is enough for tiled inference
#SBATCH --cpus-per-task=24              # 24 for gpu_h200 | 16 for gpu/hpda/gpu_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00                 # tiled inference on 1800×3600 is fast (< 1 h)
#SBATCH --output=logs/%x-%j.out         # %x=job-name  %j=jobid  (relative to submit dir)
#SBATCH --error=logs/%x-%j.err

# ===== CRIANN Austral environment ===========================================
module purge
module load aidl/pytorch/2.6.0-cuda12.6   # python 3.13 — the ONLY module that works on H200
                                           # alt for A100: aidl/pytorch/2.5.1-cuda12.4 (py3.12)

# ===== Paths ================================================================
# /dlocal/home/<projet>  = PERSISTENT, no hard quota (Lustre). Use for code+data+checkpoints.
# /home/<projet>/<login> = only 50 Go — never train or write checkpoints here.
PROJ=/dlocal/home/$(id -gn)               # your projet_id group  (verify: echo $HOME ; cri_quota)
CODE_DIR=/home/1997048/abufor01/FlashEdges            # repo cloned via init.sh
export HF_HOME=$PROJ/hf_cache              # keep HF cache on persistent Lustre

mkdir -p logs
cd "$CODE_DIR"                            # so `python -m backend.main` resolves

# ===== Install deps once per module =========================================
# The inference engine (backend/inference_engine.py) needs h5py to read the
# input H5 and rasterio to write GeoTIFF forecasts. rio-cogeo is optional
# (Cloud Optimized GeoTIFF conversion); if it fails the engine falls back to
# plain GeoTIFF silently.
pip install --user --no-cache-dir \
    suncalc safetensors h5py rasterio tqdm pyyaml >/dev/null
pip install --user --no-cache-dir rio-cogeo 2>/dev/null || \
    echo "[warn] rio-cogeo not installed; COG conversion will be skipped"

# ===== Configuration (override via env vars) ================================
# Model checkpoint produced by scripts/train_rf_satellite_metar.py
MODEL_PATH="${MODEL_PATH:-models/checkpoint.safetensors}"

# H5 file(s) to forecast — accepts a single path or a glob pattern.
# Default: the sample file shipped in data/.
#   Override:  H5_PATTERN="data/2026-07-*_flashedges_global.h5" sbatch ...
H5_PATTERN="${H5_PATTERN:-data/2026-06-28_10-00_flashedges_global.h5}"

# How many hours ahead to forecast (each AR step generates nb_forecast=3 frames).
FORECAST_STEPS="${FORECAST_STEPS:-18}"
DENOISING_STEPS="${DENOISING_STEPS:-32}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Where to save the GeoTIFF forecasts.  Each timestep produces:
#   forecast_YYYYMMDDHHMM_sat.tif   (2 bands: LWIR + VIS)
#   forecast_YYYYMMDDHHMM_metar.tif (7 bands: tmpc, dwpc, mslp, cloud, p01m, u, v)
# ~230 MB per timestep on the full 1800×3600 grid — if you forecast many hours
# consider pointing OUTPUT_DIR at $PROJ (persistent Lustre, no quota).
OUTPUT_DIR="${OUTPUT_DIR:-forecasts}"

mkdir -p "$OUTPUT_DIR"

# ===== Run ==================================================================
# backend.main is the CLI wrapper around FlashEdgesInferenceEngine:
#   - loads the DualJiT3D model from the safetensors checkpoint
#   - reads the H5 (sat_data, metar_data, elevation_data + attrs)
#   - builds 4-frame context, normalizes, runs tiled diffusion with AR rollout
#   - writes GeoTIFFs per forecast hour to OUTPUT_DIR
#
# The defaults below match the training config (residual=true, linear
# interpolation, context_frames=4, SDE sampler). Override via env vars or edit.

FAIL=0
for h5_file in $H5_PATTERN; do
    # Skip if the glob didn't expand to a real file
    [ -f "$h5_file" ] || { echo "[skip] not a file: $h5_file"; continue; }

    echo "============================================================"
    echo "  Forecasting : $h5_file"
    echo "  Model       : $MODEL_PATH"
    echo "  Steps ahead : $FORECAST_STEPS  (denoising=$DENOISING_STEPS)"
    echo "  Output dir  : $OUTPUT_DIR"
    echo "============================================================"

    python -m backend.main \
        --model_path "$MODEL_PATH" \
        --data_path "$h5_file" \
        --output_dir "$OUTPUT_DIR" \
        --forecast_steps "$FORECAST_STEPS" \
        --denoising_steps "$DENOISING_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --context_frames 4 \
        --interpolation linear \
        --use_residual

    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[error] inference failed (exit $rc) for $h5_file"
        FAIL=1
    fi
done

if [ $FAIL -ne 0 ]; then
    echo "One or more forecasts failed. See logs/%x-%j.err for details."
    exit 1
fi

echo "Forecast complete. Results in $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"/
