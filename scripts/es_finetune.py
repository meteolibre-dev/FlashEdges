"""
Evolution Strategies (ES) fine-tuning for the FlashEdges global satellite +
METAR rectified-flow model.

Motivation
----------
The flow-matching SFT loss is teacher-forced: the model only ever sees *clean*
context.  During autoregressive (AR) rollout the context is the model's own
(slightly-off) predictions, and errors compound — the classic exposure-bias /
covariate-shift failure.  "SFT Memorizes, RL Generalizes" (Chu et al. 2025,
arXiv 2501.17161) shows that SFT memorizes the training distribution while RL
with *outcome-based* rewards generalizes to out-of-distribution inputs — which
is exactly the rollout-distribution shift we face.

This script implements a lightweight RL-style post-training using Evolution
Strategies (ES) instead of backprop-through-rollout:

  1. SFT-warmup checkpoint is loaded (the paper's prescription: SFT first to
     fix the output "format", then RL/ES to generalize).
  2. For each ES iteration:
       a. Sample N parameter perturbations ε_i ~ N(0, σ²I).
       b. For each perturbation, run a truncated AR rollout on a sub-grid
          extracted from an H5 rollout file and compute an outcome-based
          reward (ground-truth match + temporal/spatial gradient stability).
       c. Update θ ← θ + (α / Nσ²) Σ_i reward_i · ε_i  (zeroth-order gradient).
  3. Optionally restrict perturbations to a LoRA / PEFT subset (like the
     METAR PEFT fine-tune) so the base representation is preserved and only
     a small adapter is nudged — the same strategy as
     ``train_rf_satellite_metar_peft.py``.

The rollout mirrors ``backend/inference_engine.py``'s tiled inference but runs
on a small sub-grid (default 256×256) for speed: the ES inner loop must run
hundreds of rollouts, so we trade spatial coverage for iteration speed.

H5 rollout file format
----------------------
The H5 files are produced by the same backend pipeline but contain *more
temporal frames* than the 4-context + 1-forecast training patches — enough to
evaluate a multi-step rollout (e.g. 4 context + 8 forecast = 12 frames):

    sat_data        : (T, 4, H, W)   float16  — GMGSI
    metar_data      : (T, 7, H, W)   float32  — METAR, NaN where no station
    elevation_data  : (H, W)         float32  — DEM
    attrs:
        num_frames, target_height, target_width, transform, epsg,
        frame_timestamps (list of ISO strings, T entries)

Usage
-----
    # Full-model ES (all params perturbed)
    uv run python scripts/es_finetune.py \
        --base_model_path models/checkpoint.safetensors \
        --rollout_data_dir data/rollout/ \
        --output_model_path models/es_finetuned.safetensors \
        --T 200 --N 16 --sigma 0.001 --alpha 0.005

    # LoRA-only ES (base frozen, only adapters perturbed — much cheaper)
    uv run python scripts/es_finetune.py \
        --base_model_path models/checkpoint.safetensors \
        --rollout_data_dir data/rollout/ \
        --output_model_path models/es_lora.safetensors \
        --use_lora --lora_rank 8 --sigma 0.01 --alpha 0.01
"""

import argparse
import os
import sys
import copy
import math
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import safetensors.torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from suncalc import get_position

# Add project root to sys.path so ``meteolibre_model`` resolves
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from meteolibre_model.models.jit3d_dual_v2 import DualJiT3D
from meteolibre_model.diffusion.rectified_flow_satellite_metar_v1 import (
    normalize,
    denormalize,
    CLIP_MIN,
    METAR_CLIP_MAX,
    structured_gaussian_noise,
    reconstruct_residual,
    build_residual_target,
)
from meteolibre_model.diffusion.utils import (
    SAT_MEAN, SAT_STD, METAR_MEAN, METAR_STD,
)

logger = logging.getLogger("es_finetune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Constants (mirror backend/inference_engine.py) ─────────────────────────
METAR_NAN_SENTINEL = -10000.0
ELEVATION_FLOOR = -100.0
NUM_SAT_CHANNELS = 5    # GMGSI 4 + elevation 1
NUM_METAR_CHANNELS = 7  # tmpc, dwpc, mslp, cloud_cover, p01m_dBZ, wind_u, wind_v

# Channel names for logging
SAT_CHANNEL_NAMES = ["gmgsi_lwir", "gmgsi_vis", "gmgsi_wv", "gmgsi_sw", "elevation"]
METAR_CHANNEL_NAMES = ["tmpc", "dwpc", "mslp", "cloud_cover", "p01m", "wind_u", "wind_v"]

# LoRA target modules (mirror train_rf_satellite_metar_peft.py)
LORA_TARGET_MODULES = r".*blocks\.\d+\.(attn\.(qkv|proj)|mlp\.w[123])$"
MODULES_TO_SAVE = ["final_layer_kpi", "persist_proj", "gate_proj"]


def load_config(config_name: str) -> dict:
    config_path = os.path.join(project_root, "meteolibre_model", "config", "configs.yml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if config_name not in config:
        raise KeyError(f"Config '{config_name}' not found in {config_path}")
    return config[config_name]


# ─────────────────────────────────────────────────────────────────────────────
# H5 rollout data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_rollout_h5(
    h5_path: str,
    context_frames: int,
    forecast_frames: int,
    device: str,
    subgrid_size: int = 256,
    seed: Optional[int] = None,
) -> Optional[dict]:
    """Load one H5 rollout file, crop a sub-grid, and return normalized tensors.

    Returns a dict with:
        initial_context : (1, C, T_ctx, H, W) normalized  [sat(5) + metar(7)]
        gts_sat         : (T_fore, c_sat, H, W) normalized ground truth sat
        gts_metar       : (T_fore, c_metar, H, W) normalized ground truth metar
        metar_mask      : (T_fore, c_metar, H, W) float  1.0 where station reported
        sat_nodata_mask : (1, c_sat, 1, H, W) bool  True where sat is no-data
        frame_timestamps: list of datetime for sun-position computation
        transform       : geo transform for the crop
        epsg            : CRS
    Returns None if the file doesn't have enough frames.
    """
    with h5py.File(h5_path, "r") as hf:
        sat_data_full = np.array(hf["sat_data"], dtype=np.float32)       # (T, 4, H, W)
        metar_data_full = np.array(hf["metar_data"], dtype=np.float32)   # (T, 7, H, W)
        elevation_full = np.array(hf["elevation_data"], dtype=np.float32) # (H, W)
        num_frames = int(hf.attrs["num_frames"])
        H_full = int(hf.attrs["target_height"])
        W_full = int(hf.attrs["target_width"])
        transform_full = list(hf.attrs["transform"])
        epsg = int(hf.attrs["epsg"])
        ts_list = [s.decode() if isinstance(s, bytes) else str(s)
                   for s in hf.attrs.get("frame_timestamps", [])]

    total_needed = context_frames + forecast_frames
    if num_frames < total_needed:
        logger.warning(f"{h5_path}: only {num_frames} frames, need {total_needed}")
        return None

    # --- Random sub-grid crop ---
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    if H_full > subgrid_size:
        crop_y = rng.randint(0, H_full - subgrid_size + 1)
    else:
        crop_y = 0
        subgrid_size = H_full
    if W_full > subgrid_size:
        crop_x = rng.randint(0, W_full - subgrid_size + 1)
    else:
        crop_x = 0
        subgrid_size = min(subgrid_size, W_full)

    H, W = subgrid_size, subgrid_size

    sat_crop = sat_data_full[:, :, crop_y:crop_y + H, crop_x:crop_x + W]       # (T,4,H,W)
    metar_crop = metar_data_full[:, :, crop_y:crop_y + H, crop_x:crop_x + W]   # (T,7,H,W)
    elev_crop = elevation_full[crop_y:crop_y + H, crop_x:crop_x + W]            # (H,W)

    # Adjust geo transform for crop
    a, b, c, d, e, f = transform_full
    new_c = a * crop_x + b * crop_y + c
    new_f = d * crop_x + e * crop_y + f
    transform = [a, b, new_c, d, e, new_f]

    # --- Parse timestamps ---
    timestamps = []
    for ts_str in ts_list[:total_needed]:
        try:
            timestamps.append(datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            try:
                timestamps.append(datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                timestamps.append(datetime.utcnow())
    if len(timestamps) < total_needed:
        # Fallback: synthesize hourly timestamps from the last known
        base = timestamps[-1] if timestamps else datetime.utcnow()
        timestamps = [base - timedelta(hours=total_needed - 1 - i) for i in range(total_needed)]

    c_sat = NUM_SAT_CHANNELS  # 5
    c_metar = NUM_METAR_CHANNELS  # 7

    # --- Build context (first context_frames) ---
    ctx_frames = []
    sat_nodata_frames = []
    for i in range(context_frames):
        sat_frame = sat_crop[i]                              # (4, H, W)
        metar_frame = metar_crop[i]                          # (7, H, W)
        elev_frame = elev_crop[None, :, :]                   # (1, H, W)
        elev_frame = np.where(elev_frame < 0, ELEVATION_FLOOR, elev_frame)

        sat_valid = ~np.isnan(sat_frame)
        sat_frame = np.where(np.isnan(sat_frame), 0.0, sat_frame)

        metar_valid = ~np.isnan(metar_frame)
        metar_frame = np.where(np.isnan(metar_frame), METAR_NAN_SENTINEL, metar_frame)

        sat_elev = np.concatenate([sat_frame, elev_frame], axis=0)  # (5, H, W)
        frame = np.concatenate([sat_elev, metar_frame], axis=0)     # (12, H, W)
        ctx_frames.append(frame[None, ...])                          # (1, 12, H, W)
        sat_nodata_frames.append((~sat_valid)[None, ...])            # (1, 4, H, W)

    context_np = np.stack(ctx_frames, axis=2)  # (1, 12, T_ctx, H, W)
    context = torch.from_numpy(context_np).float().to(device)

    sat_nodata = np.stack(sat_nodata_frames, axis=2)  # (1, 4, T_ctx, H, W)
    sat_nodata = torch.from_numpy(sat_nodata).to(device)
    elev_valid = torch.zeros_like(sat_nodata[:, :1])  # elevation always valid
    sat_nodata = torch.cat([sat_nodata, elev_valid], dim=1)  # (1, 5, T_ctx, H, W)

    # --- Normalize context ---
    sat_ctx = context[:, :c_sat]
    metar_ctx = context[:, c_sat:]
    metar_valid_ctx = (metar_ctx != METAR_NAN_SENTINEL)
    metar_ctx = torch.where(metar_valid_ctx, metar_ctx, torch.zeros_like(metar_ctx))

    sat_ctx, metar_ctx = normalize(sat_ctx, metar_ctx, device)
    sat_ctx = torch.where(~sat_nodata, sat_ctx, torch.zeros_like(sat_ctx))
    metar_ctx = torch.where(metar_valid_ctx, metar_ctx, torch.zeros_like(metar_ctx))
    initial_context = torch.cat([sat_ctx, metar_ctx], dim=1)  # (1, 12, T_ctx, H, W)

    # --- Build ground truth (next forecast_frames) ---
    gt_sat_list = []
    gt_metar_list = []
    gt_metar_mask_list = []
    for i in range(context_frames, context_frames + forecast_frames):
        sat_gt = sat_crop[i]                    # (4, H, W)
        metar_gt = metar_crop[i]                # (7, H, W)
        elev_gt = elev_crop[None, :, :]
        elev_gt = np.where(elev_gt < 0, ELEVATION_FLOOR, elev_gt)

        sat_valid_gt = ~np.isnan(sat_gt)
        sat_gt = np.where(np.isnan(sat_gt), 0.0, sat_gt)
        metar_valid_gt = ~np.isnan(metar_gt)
        metar_gt = np.where(np.isnan(metar_gt), METAR_NAN_SENTINEL, metar_gt)

        sat_elev_gt = np.concatenate([sat_gt, elev_gt], axis=0)  # (5, H, W)
        sat_gt_t = torch.from_numpy(sat_elev_gt).float().to(device).unsqueeze(0).unsqueeze(2)  # (1,5,1,H,W)
        metar_gt_t = torch.from_numpy(metar_gt).float().to(device).unsqueeze(0).unsqueeze(2)   # (1,7,1,H,W)

        sat_gt_norm, metar_gt_norm = normalize(sat_gt_t, metar_gt_t, device)
        sat_gt_norm = torch.where(
            ~sat_valid_gt[None, None, ...], sat_gt_norm, torch.zeros_like(sat_gt_norm)
        )
        metar_valid_gt_t = (metar_gt_t != METAR_NAN_SENTINEL)
        metar_gt_norm = torch.where(
            metar_valid_gt_t, metar_gt_norm, torch.zeros_like(metar_gt_norm)
        )

        gt_sat_list.append(sat_gt_norm.squeeze(0).squeeze(1))        # (5, H, W)
        gt_metar_list.append(metar_gt_norm.squeeze(0).squeeze(1))    # (7, H, W)
        gt_metar_mask_list.append(metar_valid_gt_t.squeeze(0).squeeze(1).float())  # (7, H, W)

    gts_sat = torch.stack(gt_sat_list, dim=0)          # (T_fore, 5, H, W)
    gts_metar = torch.stack(gt_metar_list, dim=0)      # (T_fore, 7, H, W)
    metar_mask = torch.stack(gt_metar_mask_list, dim=0) # (T_fore, 7, H, W)

    return {
        "initial_context": initial_context,
        "gts_sat": gts_sat,
        "gts_metar": gts_metar,
        "metar_mask": metar_mask,
        "sat_nodata_mask": sat_nodata,
        "timestamps": timestamps,
        "transform": transform,
        "epsg": epsg,
        "crop_y": crop_y,
        "crop_x": crop_x,
        "H": H,
        "W": W,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Patch-level AR rollout (no tiling — sub-grid is small enough)
# ─────────────────────────────────────────────────────────────────────────────

def compute_spatial_position(
    date: datetime,
    crop_x: int,
    crop_y: int,
    H: int,
    W: int,
    transform: list,
    device: str,
) -> torch.Tensor:
    """Compute sun-position spatial features for the sub-grid center.

    Returns (4,): [sun_azimuth, sun_altitude, noon_sun_altitude, lat/25]
    """
    a, b, c, d, e, f = transform
    cx = crop_x + W // 2
    cy = crop_y + H // 2
    lon = a * cx + c
    lat = e * cy + f

    pos = get_position(date, lon, lat)
    date_noon = date.replace(hour=12, minute=0, second=0, microsecond=0)
    pos_noon = get_position(date_noon, lon, lat)

    spatial = torch.tensor(
        [pos["azimuth"], pos["altitude"], pos_noon["altitude"], lat / 25.0],
        device=device, dtype=torch.float32,
    )
    return spatial


@torch.no_grad()
def run_rollout(
    model: torch.nn.Module,
    rollout_data: dict,
    context_frames: int,
    forecast_frames: int,
    denoising_steps: int,
    interpolation: str,
    use_residual: bool,
    noise_rho: float,
    device: str,
    patch_size: int = 128,
    batch_size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run autoregressive rollout on the sub-grid.

    If the sub-grid is larger than patch_size, runs a simple tiled inference
    with Gaussian-weighted blending (mirrors inference_engine but lighter).
    If it fits in one patch, runs a single forward pass per denoising step.

    Returns:
        sat_pred  : (T_fore, c_sat, H, W) normalized predictions
        metar_pred: (T_fore, c_metar, H, W) normalized predictions
    """
    model.eval()
    model.to(device)

    initial_context = rollout_data["initial_context"]  # (1, 12, T_ctx, H, W)
    c_sat = NUM_SAT_CHANNELS
    c_metar = NUM_METAR_CHANNELS
    C = c_sat + c_metar
    _, _, T_ctx, H, W = initial_context.shape
    timestamps = rollout_data["timestamps"]
    transform = rollout_data["transform"]
    crop_x = rollout_data["crop_x"]
    crop_y = rollout_data["crop_y"]

    current_context = initial_context.clone()

    all_sat = []
    all_metar = []

    nb_forecast_per_step = forecast_frames  # generate all at once (sub-grid is small)

    step = 0
    while step < forecast_frames:
        remaining = forecast_frames - step
        this_nb = min(nb_forecast_per_step, remaining)
        pred_date = timestamps[context_frames + step] if context_frames + step < len(timestamps) else datetime.utcnow()

        # --- Structured Gaussian prior ---
        x_t = structured_gaussian_noise(
            (1, C, this_nb, H, W),
            device=device,
            rho=noise_rho,
        ).clone()

        # --- Spatial position for this forecast frame ---
        spatial_pos = compute_spatial_position(
            pred_date, crop_x, crop_y, H, W, transform, device
        )

        # --- Tiled or single-patch denoising loop ---
        if H <= patch_size and W <= patch_size:
            # Single patch — no tiling needed
            for i in range(denoising_steps):
                t_val = 1.0 - i / denoising_steps
                dt = 1.0 / denoising_steps
                t_batch = torch.full((1,), t_val, device=device)
                d_batch = torch.full((1,), 0.0, device=device)

                model_input = torch.cat([current_context, x_t], dim=2)
                context_global = torch.cat([
                    spatial_pos.unsqueeze(0),
                    d_batch.unsqueeze(-1),
                    t_batch.unsqueeze(-1),
                ], dim=1)

                sat_pred, metar_pred = model(
                    model_input[:, :c_sat].float(),
                    model_input[:, c_metar:].float(),
                    context_global.float(),
                    metar_ref=model_input[:, c_metar:].float(),
                )

                x_pred = torch.cat([sat_pred, metar_pred], dim=1)[:, :, context_frames:]

                if interpolation == "polynomial":
                    s_theta = (x_t - x_pred) / (2 * t_val + 1e-8)
                else:
                    s_theta = (x_t - x_pred) / t_val
                x_t = x_t - s_theta * dt
                x_t = x_t.clamp(-7, 8)
        else:
            # Tiled inference with Gaussian blending
            patch_weights = _get_gaussian_weights(patch_size, device)
            patch_weights = patch_weights.view(1, 1, 1, patch_size, patch_size)
            patch_coords = _build_patch_coords(H, W, patch_size)

            for i in range(denoising_steps):
                t_val = 1.0 - i / denoising_steps
                dt = 1.0 / denoising_steps
                t_batch = torch.full((1,), t_val, device=device)
                d_batch = torch.full((1,), 0.0, device=device)

                aggregated = torch.zeros(1, C, this_nb, H, W, device=device, dtype=torch.float32)
                wsum = torch.zeros(1, 1, this_nb, H, W, device=device, dtype=torch.float32)

                for ib in range(0, len(patch_coords), batch_size):
                    coords_batch = patch_coords[ib:ib + batch_size]
                    ctx_batch, xt_batch, cg_batch = [], [], []

                    for (x_s, y_s) in coords_batch:
                        p_ctx = current_context[..., y_s:y_s + patch_size, x_s:x_s + patch_size]
                        p_xt = x_t[..., y_s:y_s + patch_size, x_s:x_s + patch_size]
                        cg_batch.append(torch.cat([
                            spatial_pos.unsqueeze(0),
                            d_batch.unsqueeze(-1), t_batch.unsqueeze(-1)
                        ], dim=1))
                        ctx_batch.append(p_ctx)
                        xt_batch.append(p_xt)

                    mi = torch.cat([
                        torch.cat(ctx_batch, dim=0),
                        torch.cat(xt_batch, dim=0),
                    ], dim=2)

                    sp, mp = model(
                        mi[:, :c_sat].float(), mi[:, c_metar:].float(),
                        torch.cat(cg_batch, dim=0).float(),
                        metar_ref=mi[:, c_metar:].float(),
                    )
                    x_pred_batch = torch.cat([sp, mp], dim=1)[:, :, context_frames:]

                    pw = patch_weights
                    for j, (x_s, y_s) in enumerate(coords_batch):
                        x_t_patch = x_t[..., y_s:y_s + patch_size, x_s:x_s + patch_size]
                        if interpolation == "polynomial":
                            v = (x_t_patch - x_pred_batch[j:j+1]) / (2 * t_val + 1e-8)
                        else:
                            v = (x_t_patch - x_pred_batch[j:j+1]) / t_val
                        aggregated[..., y_s:y_s+patch_size, x_s:x_s+patch_size] += v * pw
                        wsum[..., y_s:y_s+patch_size, x_s:x_s+patch_size] += pw

                wsum[wsum == 0] = 1.0
                averaged = aggregated / wsum
                x_t = x_t - averaged * dt
                x_t = x_t.clamp(-7, 8)

        # --- Residual reconstruction ---
        if use_residual:
            last_ctx = current_context[:, :, -1:, :, :]
            x_t = reconstruct_residual(x_t, last_ctx, c_sat, c_metar, device)

        sat_frame = x_t[:, :c_sat, :this_nb, :, :]    # (1, c_sat, this_nb, H, W)
        metar_frame = x_t[:, c_metar:, :this_nb, :, :]

        for k in range(this_nb):
            all_sat.append(sat_frame[0, :, k])       # (c_sat, H, W)
            all_metar.append(metar_frame[0, :, k])   # (c_metar, H, W)

        # --- Slide context window ---
        if this_nb >= T_ctx:
            new_context = x_t[:, :, -T_ctx:, :, :].clone()
        else:
            tail = current_context[:, :, this_nb:, :, :]
            new_context = torch.cat([tail, x_t[:, :, :this_nb, :, :]], dim=2)
        current_context = new_context
        step += this_nb

    sat_pred = torch.stack(all_sat, dim=0)      # (T_fore, c_sat, H, W)
    metar_pred = torch.stack(all_metar, dim=0)  # (T_fore, c_metar, H, W)
    return sat_pred, metar_pred


def _get_gaussian_weights(patch_size: int, device: str, sigma_scale: float = 0.3) -> torch.Tensor:
    x = torch.linspace(-(patch_size - 1) / 2, (patch_size - 1) / 2, patch_size, device=device)
    sigma = sigma_scale * patch_size
    w_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    w_2d = w_1d.unsqueeze(1) * w_1d.unsqueeze(0)
    return w_2d / w_2d.max()


def _build_patch_coords(H: int, W: int, ps: int) -> List[Tuple[int, int]]:
    shift = ps // 2

    def get_starts(total, size, offset):
        starts = list(range(offset, total - size + 1, size))
        if not starts or starts[-1] != total - size:
            if total >= size:
                starts.append(total - size)
        return starts

    y0 = get_starts(H, ps, 0)
    x0 = get_starts(W, ps, 0)
    yS = get_starts(H, ps, shift)
    xS = get_starts(W, ps, shift)

    c1 = [(x, y) for y in y0 for x in x0]
    c2 = [(x, y) for y in yS for x in xS]
    extra = [(x, y0[0]) for x in xS] + [(x0[0], y) for y in yS]
    extra += [(x, y0[-1]) for x in xS] + [(x0[-1], y) for y in yS]
    return list(set(c1 + c2 + extra))


# ─────────────────────────────────────────────────────────────────────────────
# Reward computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    sat_pred: torch.Tensor,       # (T_fore, c_sat, H, W) normalized
    metar_pred: torch.Tensor,     # (T_fore, c_metar, H, W) normalized
    gts_sat: torch.Tensor,        # (T_fore, c_sat, H, W) normalized
    gts_metar: torch.Tensor,      # (T_fore, c_metar, H, W) normalized
    metar_mask: torch.Tensor,     # (T_fore, c_metar, H, W) float
    sat_nodata_mask: torch.Tensor, # (1, c_sat, 1, H, W) bool
    # Reward weights
    gt_weight: float = 1.0,
    temporal_grad_weight: float = 0.1,
    spatial_grad_weight: float = 0.1,
    metar_gt_weight: float = 0.5,
    # Temporal horizon weighting: later frames count more (compounding errors)
    temporal_horizon_scale: float = 1.0,
) -> Tuple[float, dict]:
    """Outcome-based reward for an AR rollout.

    The reward is negative-cost (higher = better). Components:

    1. **Ground-truth match** (gt_weight): masked MSE between rollout predictions
       and ground truth. Satellite is masked by valid-data; METAR is masked by
       the station mask. Later forecast frames are upweighted (temporal_horizon_scale)
       because that's where compounding errors manifest — directly targeting
       rollout stability.

    2. **Temporal gradient stability** (temporal_grad_weight): penalizes
       frame-to-frame jitter in the satellite prediction. The *ground-truth*
       temporal gradient is the reference; we penalize deviation from it.
       This is the FastNet temporal-gradient regularizer, but evaluated on the
       *rolled-out* trajectory instead of a teacher-forced single step.

    3. **Spatial gradient stability** (spatial_grad_weight): penalizes
       nonphysical spatial artifacts that compound during AR rollout.
       FastNet spatial-gradient regularizer, evaluated on rollout output.

    METAR is excluded from the gradient regularizers (sparse point data has
    legitimately sharp gradients at station pixels).
    """
    T_fore, c_sat, H, W = sat_pred.shape
    device = sat_pred.device

    # --- Temporal horizon weighting ---
    if temporal_horizon_scale > 0 and T_fore > 1:
        ramp = torch.arange(1, T_fore + 1, device=device, dtype=torch.float32)
        ramp = ramp / ramp.mean()
        hw = (1.0 - temporal_horizon_scale) + temporal_horizon_scale * ramp
        hw = hw.view(T_fore, 1, 1, 1)
    else:
        hw = torch.ones(T_fore, 1, 1, 1, device=device)

    # --- 1. Ground-truth match ---
    # Satellite: mask out no-data pixels
    sat_valid = ~sat_nodata_mask[:, :, 0, :, :]  # (1, c_sat, H, W) -> use first frame layout
    sat_valid = sat_valid.expand(T_fore, c_sat, H, W)

    sat_diff = hw * (sat_pred - gts_sat) ** 2   # (T_fore, c_sat, H, W)
    sat_cnt = sat_valid.float().sum().clamp(min=1.0)
    sat_mse = (sat_diff * sat_valid.float()).sum() / sat_cnt

    # METAR: mask by station mask
    metar_diff = hw * (metar_pred - gts_metar) ** 2
    metar_cnt = metar_mask.sum().clamp(min=1.0)
    metar_mse = (metar_diff * metar_mask).sum() / metar_cnt

    gt_cost = gt_weight * sat_mse + metar_gt_weight * metar_mse

    # --- 2. Temporal gradient stability (satellite only) ---
    if temporal_grad_weight > 0 and T_fore > 1:
        dT_pred = sat_pred[1:] - sat_pred[:-1]   # (T_fore-1, c_sat, H, W)
        dT_gt = gts_sat[1:] - gts_sat[:-1]
        pair_valid = sat_valid[1:] & sat_valid[:-1]
        tg_err = (hw[1:] * (dT_pred - dT_gt) ** 2)
        tg_cost = (tg_err * pair_valid.float()).sum() / pair_valid.float().sum().clamp(min=1.0)
    else:
        tg_cost = torch.tensor(0.0, device=device)

    # --- 3. Spatial gradient stability (satellite only) ---
    if spatial_grad_weight > 0:
        gy_p, gx_p = torch.gradient(sat_pred, dim=(-2, -1))
        gy_g, gx_g = torch.gradient(gts_sat, dim=(-2, -1))
        sg_err = hw * ((gy_p - gy_g) ** 2 + (gx_p - gx_g) ** 2)
        sg_cost = (sg_err * sat_valid.float()).sum() / sat_cnt
    else:
        sg_cost = torch.tensor(0.0, device=device)

    total_cost = gt_cost + temporal_grad_weight * tg_cost + spatial_grad_weight * sg_cost
    reward = -total_cost.item()

    components = {
        "sat_mse": sat_mse.item(),
        "metar_mse": metar_mse.item(),
        "temporal_grad": tg_cost.item(),
        "spatial_grad": sg_cost.item(),
        "total_cost": total_cost.item(),
    }
    return reward, components


# ─────────────────────────────────────────────────────────────────────────────
# ES loop
# ─────────────────────────────────────────────────────────────────────────────

def get_trainable_params(model: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
    """Return list of (name, param) for parameters that require grad."""
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def perturb_model(model: torch.nn.Module, seeds: List[int], sigma: float, device: str):
    """Apply Gaussian perturbation to all trainable params. Seeds must match
    for restore. Modifies params in-place."""
    for (name, param), seed in zip(get_trainable_params(model), seeds):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        noise = torch.randn(param.shape, generator=gen, device=device, dtype=param.dtype)
        param.data.add_(noise * sigma)


def restore_model(model: torch.nn.Module, seeds: List[int], sigma: float, device: str):
    """Undo the perturbation (subtract the same noise)."""
    for (name, param), seed in zip(get_trainable_params(model), seeds):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        noise = torch.randn(param.shape, generator=gen, device=device, dtype=param.dtype)
        param.data.sub_(noise * sigma)


def es_update(model: torch.nn.Module, all_seeds: List[List[int]], rewards: torch.Tensor,
              sigma: float, alpha: float, device: str, antithetic: bool = False):
    """Zeroth-order gradient update: θ ← θ + (α / Nσ) Σ_i z_i * ε_i.

    With antithetic sampling, rewards has 2N entries (first N are +ε, last N
    are -ε), and the gradient uses the paired difference (r_i+ - r_i-) which
    cancels the constant baseline and reduces variance.
    """
    if antithetic:
        N = len(all_seeds)
        rewards_pos = rewards[:N]
        rewards_neg = rewards[N:]
        if rewards_pos.std() == 0 and rewards_neg.std() == 0:
            return
        # Antithetic gradient: (r_i+ - r_i-) * ε_i  (baseline-free)
        paired = rewards_pos - rewards_neg
        # Still z-score the paired advantages for stable step sizes
        if paired.std() > 1e-8:
            advantages = (paired - paired.mean()) / (paired.std() + 1e-8)
        else:
            advantages = paired

        for (name, param), seed_idx in zip(get_trainable_params(model), range(len(get_trainable_params(model)))):
            grad = torch.zeros_like(param.data)
            for n in range(N):
                seeds = all_seeds[n]
                seed = seeds[seed_idx]
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                noise = torch.randn(param.shape, generator=gen, device=device, dtype=param.dtype)
                grad += advantages[n].item() * noise
            param.data.add_((alpha / (N * sigma)) * grad)
    else:
        N = len(all_seeds)
        if rewards.std() == 0:
            return

        z_scores = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for (name, param), seed_idx in zip(get_trainable_params(model), range(len(get_trainable_params(model)))):
            grad = torch.zeros_like(param.data)
            for n in range(N):
                seeds = all_seeds[n]
                seed = seeds[seed_idx]
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                noise = torch.randn(param.shape, generator=gen, device=device, dtype=param.dtype)
                grad += z_scores[n].item() * noise
            param.data.add_((alpha / (N * sigma)) * grad)


def es_finetune(
    model: torch.nn.Module,
    rollout_files: List[str],
    context_frames: int,
    forecast_frames: int,
    T: int = 200,
    N: int = 16,
    sigma: float = 0.001,
    alpha: float = 0.005,
    device: str = "cuda",
    denoising_steps: int = 16,
    interpolation: str = "linear",
    use_residual: bool = True,
    noise_rho: float = 0.0,
    subgrid_size: int = 256,
    patch_size: int = 128,
    batch_size: int = 8,
    # Reward weights
    gt_weight: float = 1.0,
    temporal_grad_weight: float = 0.1,
    spatial_grad_weight: float = 0.1,
    metar_gt_weight: float = 0.5,
    temporal_horizon_scale: float = 1.0,
    # Logging
    save_path: Optional[str] = None,
    save_every: int = 20,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 10,
    eval_files: Optional[List[str]] = None,
    antithetic: bool = True,
):
    """Main ES fine-tuning loop.

    Args:
        model: The model to fine-tune (already on device, optionally PEFT-wrapped).
        rollout_files: List of H5 file paths for rollout evaluation.
        context_frames: Number of context frames (must match model).
        forecast_frames: Number of forecast frames per rollout.
        T: Number of ES iterations.
        N: Number of perturbation samples per iteration.
        sigma: Perturbation std.
        alpha: Learning rate (update step size).
        device: torch device.
        denoising_steps: Euler steps for the RF ODE per rollout.
        interpolation: 'linear' or 'polynomial'.
        use_residual: Whether model uses residual targets.
        noise_rho: Structured noise sharing for the RF prior.
        subgrid_size: Spatial crop size for rollout (smaller = faster).
        patch_size: Patch size for tiled inference (if subgrid > patch).
        batch_size: Batch size for tiled inference.
        gt_weight, temporal_grad_weight, spatial_grad_weight, metar_gt_weight:
            Reward component weights.
        temporal_horizon_scale: Upweight later forecast frames in reward.
        save_path: Where to save the fine-tuned model.
        save_every: Save checkpoint every N iterations.
        writer: TensorBoard SummaryWriter.
        log_interval: Full-eval logging interval.
        eval_files: Separate files for periodic full eval (None = use rollout_files).
    """
    model.to(device)
    model.train()  # for param updates; eval during rollout

    trainable = get_trainable_params(model)
    n_trainable = sum(p.numel() for _, p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"ES fine-tuning: {n_trainable:,} trainable / {n_total:,} total params")
    logger.info(f"  T={T}, N={N}, sigma={sigma}, alpha={alpha}")
    logger.info(f"  rollout: {forecast_frames} forecast frames, {denoising_steps} denoising steps, "
                f"subgrid={subgrid_size}px")
    logger.info(f"  reward: gt={gt_weight}, temporal_grad={temporal_grad_weight}, "
                f"spatial_grad={spatial_grad_weight}, metar_gt={metar_gt_weight}")
    logger.info(f"  antithetic={antithetic} (effective N={N*2 if antithetic else N})")

    eval_files = eval_files or rollout_files

    best_reward = -float("inf")
    best_state = None

    for t in tqdm(range(T), desc="ES Iterations"):
        # --- Pick a random rollout file + crop ---
        h5_path = random.choice(rollout_files)
        data_seed = random.randint(0, 2**32 - 1)
        rollout_data = load_rollout_h5(
            h5_path, context_frames, forecast_frames, device,
            subgrid_size=subgrid_size, seed=data_seed,
        )
        if rollout_data is None:
            continue  # skip files with insufficient frames

        # --- Generate per-param seeds for each of N perturbations ---
        # With antithetic sampling, each seed pair (ε, -ε) gives two evaluations,
        # halving gradient variance for free.
        n_params = len(trainable)
        n_evals = N * 2 if antithetic else N
        all_seeds = [[random.randint(0, 2**32 - 1) for _ in range(n_params)] for _ in range(N)]
        all_signs = [1.0] * N + [-1.0] * N if antithetic else [1.0] * N
        # For antithetic: second half reuses the same seeds but with negative perturbation
        if antithetic:
            all_seeds = all_seeds + all_seeds  # duplicate for the -ε half

        rewards = []
        component_accum = {"sat_mse": 0, "metar_mse": 0, "temporal_grad": 0, "spatial_grad": 0}

        for n in range(n_evals):
            sign = all_signs[n]
            # Perturb (positive or negative)
            perturb_model(model, all_seeds[n], sign * sigma, device)

            # Evaluate
            sat_pred, metar_pred = run_rollout(
                model, rollout_data, context_frames, forecast_frames,
                denoising_steps, interpolation, use_residual, noise_rho, device,
                patch_size=patch_size, batch_size=batch_size,
            )

            reward, comps = compute_reward(
                sat_pred, metar_pred,
                rollout_data["gts_sat"], rollout_data["gts_metar"],
                rollout_data["metar_mask"], rollout_data["sat_nodata_mask"],
                gt_weight, temporal_grad_weight, spatial_grad_weight,
                metar_gt_weight, temporal_horizon_scale,
            )
            rewards.append(reward)
            for k in component_accum:
                component_accum[k] += comps[k]

            # Restore
            restore_model(model, all_seeds[n], sign * sigma, device)

        rewards_tensor = torch.tensor(rewards, device=device)

        # --- ES update ---
        es_update(model, all_seeds[:N] if antithetic else all_seeds,
                  rewards_tensor, sigma, alpha, device, antithetic=antithetic)

        # --- Logging ---
        avg_reward = rewards_tensor.mean().item()
        std_reward = rewards_tensor.std().item()
        for k in component_accum:
            component_accum[k] /= n_evals

        if writer is not None:
            writer.add_scalar("ES/avg_reward", avg_reward, t + 1)
            writer.add_scalar("ES/std_reward", std_reward, t + 1)
            writer.add_scalar("ES/max_reward", rewards_tensor.max().item(), t + 1)
            writer.add_scalar("ES/min_reward", rewards_tensor.min().item(), t + 1)
            writer.add_scalar("Reward/sat_mse", component_accum["sat_mse"], t + 1)
            writer.add_scalar("Reward/metar_mse", component_accum["metar_mse"], t + 1)
            writer.add_scalar("Reward/temporal_grad", component_accum["temporal_grad"], t + 1)
            writer.add_scalar("Reward/spatial_grad", component_accum["spatial_grad"], t + 1)

        if (t + 1) % 5 == 0:
            logger.info(
                f"Iter {t+1}/{T}: reward={avg_reward:.4f}±{std_reward:.4f} "
                f"[sat_mse={component_accum['sat_mse']:.4f} "
                f"metar_mse={component_accum['metar_mse']:.4f} "
                f"tgrad={component_accum['temporal_grad']:.4f} "
                f"sgrad={component_accum['spatial_grad']:.4f}]"
            )

        # --- Periodic full eval (on separate eval files) ---
        if writer is not None and (t + 1) % log_interval == 0:
            model.eval()
            eval_rewards = []
            for ef in eval_files[:3]:  # eval on up to 3 files
                ed = load_rollout_h5(ef, context_frames, forecast_frames, device,
                                     subgrid_size=subgrid_size, seed=42)
                if ed is None:
                    continue
                sp, mp = run_rollout(
                    model, ed, context_frames, forecast_frames,
                    denoising_steps, interpolation, use_residual, noise_rho, device,
                    patch_size=patch_size, batch_size=batch_size,
                )
                r, _ = compute_reward(
                    sp, mp, ed["gts_sat"], ed["gts_metar"],
                    ed["metar_mask"], ed["sat_nodata_mask"],
                    gt_weight, temporal_grad_weight, spatial_grad_weight,
                    metar_gt_weight, temporal_horizon_scale,
                )
                eval_rewards.append(r)

            if eval_rewards:
                avg_eval = np.mean(eval_rewards)
                writer.add_scalar("Reward/eval", avg_eval, t + 1)
                logger.info(f"  [eval] avg reward: {avg_eval:.4f}")

                if avg_eval > best_reward:
                    best_reward = avg_eval
                    best_state = copy.deepcopy({
                        k: v.clone() for k, v in model.state_dict().items()
                    })
                    logger.info(f"  [eval] new best: {best_reward:.4f}")

            model.train()

        # --- Checkpoint ---
        if save_path and (t + 1) % save_every == 0:
            _save_model(model, save_path)
            logger.info(f"  [save] checkpoint -> {save_path}")

    # --- Save final / best ---
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model (eval reward={best_reward:.4f})")

    if save_path:
        _save_model(model, save_path)
        logger.info(f"Final model saved to {save_path}")

    return model


def _save_model(model: torch.nn.Module, save_path: str):
    """Save model. If PEFT-wrapped, merge LoRA and save as plain state_dict."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Check if it's a PEFT model
    peft_model = getattr(model, "_orig_mod", model)
    if hasattr(peft_model, "merge_and_unload"):
        try:
            merged = peft_model.merge_and_unload()
            save_file(merged.state_dict(), save_path)
            return
        except Exception as e:
            logger.warning(f"merge_and_unload failed ({e}); saving raw state_dict")

    save_file(model.state_dict(), save_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ES fine-tuning for FlashEdges model (rollout stability)."
    )
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base .safetensors model (SFT warmup).")
    parser.add_argument("--rollout_data_dir", type=str, required=True,
                        help="Directory with H5 rollout files (many temporal frames).")
    parser.add_argument("--eval_data_dir", type=str, default=None,
                        help="Directory with H5 eval files (default: same as rollout_data_dir).")
    parser.add_argument("--config", type=str, default="model_v2_global_satellite_metar",
                        help="Config name in configs.yml.")
    # ES params
    parser.add_argument("--T", type=int, default=200, help="ES iterations.")
    parser.add_argument("--N", type=int, default=16, help="Perturbation samples per iteration.")
    parser.add_argument("--sigma", type=float, default=0.001, help="Perturbation std.")
    parser.add_argument("--alpha", type=float, default=0.005, help="Update rate.")
    # Rollout params
    parser.add_argument("--forecast_frames", type=int, default=8,
                        help="Number of forecast frames per rollout (AR steps).")
    parser.add_argument("--denoising_steps", type=int, default=16,
                        help="Euler steps for the RF ODE.")
    parser.add_argument("--subgrid_size", type=int, default=256,
                        help="Spatial crop size for rollout eval (smaller = faster).")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="Patch size for tiled inference within the subgrid.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for tiled patch inference.")
    parser.add_argument("--noise_rho", type=float, default=0.0,
                        help="Structured noise sharing for the RF prior.")
    # Reward weights
    parser.add_argument("--gt_weight", type=float, default=1.0,
                        help="Weight for ground-truth MSE in reward.")
    parser.add_argument("--temporal_grad_weight", type=float, default=0.1,
                        help="Weight for temporal-gradient stability in reward.")
    parser.add_argument("--spatial_grad_weight", type=float, default=0.1,
                        help="Weight for spatial-gradient stability in reward.")
    parser.add_argument("--metar_gt_weight", type=float, default=0.5,
                        help="Weight for METAR ground-truth MSE in reward.")
    parser.add_argument("--temporal_horizon_scale", type=float, default=1.0,
                        help="Upweight later forecast frames in reward (targets compounding).")
    # PEFT
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA adapters (base frozen, only adapters perturbed).")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    # Output
    parser.add_argument("--output_model_path", type=str, default="models/es_finetuned.safetensors")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--no_antithetic", action="store_true",
                        help="Disable antithetic sampling (default: enabled for variance reduction).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # --- Config ---
    params = load_config(args.config)
    context_frames = params["model"]["context_frames"]
    interpolation = params.get("interpolation", "linear")
    use_residual = bool(params.get("residual", False))

    # --- Load model ---
    logger.info(f"Loading base model from {args.base_model_path}")
    torch.set_float32_matmul_precision("medium")
    model = DualJiT3D(**params["model"])
    state_dict = safetensors.torch.load_file(args.base_model_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")
    model.to(args.device)

    # --- Optional LoRA ---
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGET_MODULES,
            modules_to_save=MODULES_TO_SAVE,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"[peft] trainable: {trainable:,} / {total:,} ({100.0*trainable/total:.3f}%)")

    # --- Collect rollout files ---
    rollout_files = sorted(str(p) for p in Path(args.rollout_data_dir).glob("*.h5"))
    if not rollout_files:
        raise ValueError(f"No H5 files found in {args.rollout_data_dir}")
    logger.info(f"Found {len(rollout_files)} rollout files in {args.rollout_data_dir}")

    eval_files = None
    if args.eval_data_dir:
        eval_files = sorted(str(p) for p in Path(args.eval_data_dir).glob("*.h5"))
        logger.info(f"Found {len(eval_files)} eval files in {args.eval_data_dir}")

    # --- TensorBoard ---
    if args.log_dir is None:
        log_dir = f"runs/es_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard logs -> {log_dir}")

    # --- Initial eval ---
    logger.info("Computing initial reward...")
    model.eval()
    init_rewards = []
    for f in rollout_files[:3]:
        rd = load_rollout_h5(f, context_frames, args.forecast_frames, args.device,
                             subgrid_size=args.subgrid_size, seed=42)
        if rd is None:
            continue
        sp, mp = run_rollout(
            model, rd, context_frames, args.forecast_frames,
            args.denoising_steps, interpolation, use_residual, args.noise_rho, args.device,
            patch_size=args.patch_size, batch_size=args.batch_size,
        )
        r, comps = compute_reward(
            sp, mp, rd["gts_sat"], rd["gts_metar"], rd["metar_mask"], rd["sat_nodata_mask"],
            args.gt_weight, args.temporal_grad_weight, args.spatial_grad_weight,
            args.metar_gt_weight, args.temporal_horizon_scale,
        )
        init_rewards.append(r)
        logger.info(f"  {os.path.basename(f)}: reward={r:.4f} {comps}")

    if init_rewards:
        avg_init = np.mean(init_rewards)
        writer.add_scalar("Reward/eval", avg_init, 0)
        logger.info(f"Initial avg reward: {avg_init:.4f}")

    # --- Run ES ---
    es_finetune(
        model=model,
        rollout_files=rollout_files,
        context_frames=context_frames,
        forecast_frames=args.forecast_frames,
        T=args.T,
        N=args.N,
        sigma=args.sigma,
        alpha=args.alpha,
        device=args.device,
        denoising_steps=args.denoising_steps,
        interpolation=interpolation,
        use_residual=use_residual,
        noise_rho=args.noise_rho,
        subgrid_size=args.subgrid_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        gt_weight=args.gt_weight,
        temporal_grad_weight=args.temporal_grad_weight,
        spatial_grad_weight=args.spatial_grad_weight,
        metar_gt_weight=args.metar_gt_weight,
        temporal_horizon_scale=args.temporal_horizon_scale,
        save_path=args.output_model_path,
        save_every=args.save_every,
        writer=writer,
        log_interval=args.log_interval,
        eval_files=eval_files,
        antithetic=not args.no_antithetic,
    )

    writer.close()
    logger.info("ES fine-tuning complete.")


if __name__ == "__main__":
    main()
