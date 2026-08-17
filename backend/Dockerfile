# FlashEdges Backend Docker Image
# Single-stage build on the CUDA runtime base.

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# System dependencies (build-essential for some pip wheels; libgl/libsm/libxext
# for rasterio/opencv image libs; libhdf5 for h5py).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libhdf5-serial-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition files first to leverage Docker cache.
COPY pyproject.toml meteolibre_model/config/configs.yml ./
COPY backend/requirements.txt ./backend/

# Install Python dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# Copy and install the local meteolibre-model package.
COPY meteolibre_model/ ./meteolibre_model/
RUN pip install --no-cache-dir -e .

# Copy the rest of the backend application.
COPY backend/ ./backend/

# Non-root user.
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# ---------------------------------------------------------------------------
# Environment configuration (override at deploy time with --set-env-vars)
# ---------------------------------------------------------------------------
ENV MODE=cloud
ENV LOG_LEVEL=INFO

# Local scratch / cache dir (H5 input + model weights + output TIFFs).
ENV CACHE_DIR=/tmp/flashedges_cache
ENV MODEL_PATH=/tmp/flashedges_cache/model.safetensors

# Model weights are downloaded from GCS on first run if not present locally.
# Upload your chosen checkpoint to this path (or override MODEL_GCS_PATH).
ENV MODEL_GCS_PATH=gs://eumetsat_mtg_preprocess/assets_models/checkpoint_17augpeft.safetensors

# GCP buckets / prefixes.
ENV GCP_SOURCE_BUCKET=eumetsat_mtg_preprocess
ENV GCP_SOURCE_PREFIX=inference_h5_global
ENV GCP_DEST_BUCKET=inference_result_flashedges_forecast
ENV GCP_DEST_PREFIX=forecasts

# Inference hyperparameters (match the training config of the checkpoint).
ENV CONFIG_NAME=model_v4_global_satellite_metar
ENV PATCH_SIZE=128
ENV DENOISING_STEPS=16
ENV BATCH_SIZE=64
ENV FORECAST_STEPS=24
ENV NB_FORECAST=3
ENV CONTEXT_FRAMES=4
ENV INTERPOLATION=linear
ENV SAMPLER=sde
ENV SDE_EPS=0.1
ENV SDE_EPS_SCHEDULE=t2
ENV INFERENCE_SEED=128

# Cloud Run injects service-account credentials via the metadata server, so
# google-cloud-storage picks them up automatically. For local/CI use, set
# GOOGLE_APPLICATION_CREDENTIALS to a key file.

# Default command: run the full cloud pipeline.
CMD ["python3", "-m", "backend.main", "--mode=cloud"]
