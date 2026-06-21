"""
Compute FastNet-style per-channel loss weights for the FlashEdges model.

For each output channel ``j`` we compute the inverse variance of its
*normalized* time-difference,

    s_j = 1 / Var[ x_norm[t+1] - x_norm[t] ]

then mean-normalize the weights within each branch so that the average
per-channel weight is 1. This keeps the total loss scale and the satellite /
METAR branch balance unchanged; only the *intra-branch* per-channel balance is
adjusted. Rationale (FastNet, arxiv 2509.17601, eq. 7): channels that change
little frame-to-frame (e.g. elevation, mean-sea-level pressure) would otherwise
contribute almost nothing to a plain masked-MSE gradient, while fast channels
(GMGSI LWIR, wind components) dominate. Equalizing the per-channel contribution
is the directly-applicable precedent for the satellite/METAR imbalance.

Masking (mirrors scripts/compute_mean_std.py):
  * sat: NaN (GMGSI off-disk) and elevation nodata (< -1) excluded. The
    time-difference at frame t is valid only where BOTH t and t+1 are valid
    (we intersect the two masks defensively; for geostationary GMGSI the
    off-disk mask is static in time, so this is ~equivalent to the single-frame
    mask).
  * metar: the dataset's valid-station mask, intersected across the two
    adjacent frames. Stations are spatially fixed so this is ~equivalent to the
    single-frame mask, but again intersected defensively.

Stats are computed in normalized space (same SAT_MEAN/STD, METAR_MEAN/STD the
model uses), because that is the space the loss lives in.

Usage (mirrors compute_mean_std.py):
    uv run python scripts/compute_loss_weights.py --num_samples 2000
    uv run python scripts/compute_loss_weights.py --num_samples -1   # all
    uv run python scripts/compute_loss_weights.py --data_dir data_2022_02
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from meteolibre_model.dataset.dataset_global_satellite_streaming import (
    FlashEdgesStreamingDataset,
)
from meteolibre_model.dataset.dataset_global_satellite_metar import METAR_FEATURES
from meteolibre_model.diffusion.utils import SAT_MEAN, SAT_STD, METAR_MEAN, METAR_STD

# sat_patch_data channel layout: [GMGSI_0, GMGSI_1, GMGSI_2, GMGSI_3, elevation]
SAT_CHANNEL_NAMES = ["gmgsi_lwir", "gmgsi_vis", "gmgsi_wv", "gmgsi_sw", "elevation"]
ELEVATION_CHANNEL_IDX = 4

# Floor on Var[Delta_x] (in normalized space, where field variance ~= 1). A
# value of 1e-3 corresponds to a per-frame change of ~3% of the field std.
# Below this a channel is treated as "essentially static" so its weight does not
# explode to infinity (e.g. elevation, which is constant in time).
VAR_FLOOR = 1e-3


def compute(
    hf_dataset_repo,
    data_files,
    data_dir,
    num_samples,
    shuffle_buffer,
    prefetch_rows,
    seed,
):
    dataset = FlashEdgesStreamingDataset(
        hf_dataset_repo=hf_dataset_repo,
        data_files=data_files,
        data_dir=data_dir,
        shuffle_buffer=shuffle_buffer,
        prefetch_rows=prefetch_rows,
        nb_temporal=7,
        precip_to_dbz=True,
        seed=seed,
    )

    n_iter = num_samples if num_samples is not None and num_samples >= 0 else None
    print(
        "Computing Var[Delta_x] in normalized space over "
        + (f"{n_iter} samples " if n_iter is not None else "the full stream ")
        + "(streaming dataset, length unknown), masking NaN/sentinel/nodata..."
    )

    n_sat = len(SAT_CHANNEL_NAMES)
    n_met = len(METAR_FEATURES)
    # streaming accumulators for the per-channel time-difference
    sat_d_sum = np.zeros(n_sat, dtype=np.float64)
    sat_d_sumsq = np.zeros(n_sat, dtype=np.float64)
    sat_d_cnt = np.zeros(n_sat, dtype=np.int64)
    met_d_sum = np.zeros(n_met, dtype=np.float64)
    met_d_sumsq = np.zeros(n_met, dtype=np.float64)
    met_d_cnt = np.zeros(n_met, dtype=np.int64)

    sat_mean = SAT_MEAN.numpy().reshape(1, -1, 1, 1)
    sat_std = SAT_STD.numpy().reshape(1, -1, 1, 1)
    met_mean = METAR_MEAN.numpy().reshape(1, -1, 1, 1)
    met_std = METAR_STD.numpy().reshape(1, -1, 1, 1)

    pbar = tqdm(dataset, total=n_iter)
    for i, sample in enumerate(pbar):
        if n_iter is not None and i >= n_iter:
            break

        # --- satellite -------------------------------------------------------
        sat = sample["sat_patch_data"].numpy().astype(np.float32)  # (T, C, H, W)
        smask = ~np.isnan(sat)
        # elevation nodata is floored to -100; real elevations are >= 0
        smask[:, ELEVATION_CHANNEL_IDX] &= sat[:, ELEVATION_CHANNEL_IDX] > -1.0
        sat = np.where(np.isnan(sat), 0.0, sat)
        sat_n = (sat - sat_mean) / sat_std  # normalized, (T, C, H, W)
        d = sat_n[1:] - sat_n[:-1]          # (T-1, C, H, W)
        dm = (smask[1:] & smask[:-1])       # valid at both endpoints
        d_valid = np.where(dm, d, 0.0)
        sat_d_sum += d_valid.sum(axis=(0, 2, 3))
        sat_d_sumsq += np.square(d_valid).sum(axis=(0, 2, 3))
        sat_d_cnt += dm.sum(axis=(0, 2, 3))

        # --- METAR -----------------------------------------------------------
        metar = sample["metar_patch_data"].numpy().astype(np.float32)
        mask = sample["metar_mask"].numpy().astype(bool)
        mask &= metar != -10000.0           # belt-and-suspenders: drop sentinel
        metar_n = (metar - met_mean) / met_std
        d = metar_n[1:] - metar_n[:-1]
        dm = (mask[1:] & mask[:-1])
        d_valid = np.where(dm, d, 0.0)
        met_d_sum += d_valid.sum(axis=(0, 2, 3))
        met_d_sumsq += np.square(d_valid).sum(axis=(0, 2, 3))
        met_d_cnt += dm.sum(axis=(0, 2, 3))

    def _var(sum_, sumsq_, cnt_):
        mean = np.divide(sum_, cnt_, out=np.zeros_like(sum_), where=cnt_ != 0)
        var = np.divide(
            sumsq_, cnt_, out=np.zeros_like(sumsq_), where=cnt_ != 0
        ) - np.square(mean)
        return mean, np.maximum(var, 0.0)

    sat_d_mean, sat_d_var = _var(sat_d_sum, sat_d_sumsq, sat_d_cnt)
    met_d_mean, met_d_var = _var(met_d_sum, met_d_sumsq, met_d_cnt)

    sat_d_var_floored = np.maximum(sat_d_var, VAR_FLOOR)
    met_d_var_floored = np.maximum(met_d_var, VAR_FLOOR)

    # raw inverse-variance weights, then mean-normalized to 1 per branch so the
    # total loss scale and the sat:metar branch balance are preserved.
    sat_raw_w = 1.0 / sat_d_var_floored
    met_raw_w = 1.0 / met_d_var_floored
    sat_w = sat_raw_w / np.mean(sat_raw_w)
    met_w = met_raw_w / np.mean(met_raw_w)

    def _print(title, names, mean, var, cnt, w):
        print(f"\n=== {title} (normalized time-difference stats) ===")
        print(f"  {'channel':14s} {'E[d]':>10s} {'Var[d]':>10s} {'count':>12s} {'weight':>10s}")
        for name, m, v, c, ww in zip(names, mean, var, cnt, w):
            print(f"  {name:14s} {m:10.4f} {v:10.4f} {c:12d} {ww:10.4f}")

    _print("sat_patch_data", SAT_CHANNEL_NAMES, sat_d_mean, sat_d_var, sat_d_cnt, sat_w)
    _print("metar_patch_data", METAR_FEATURES, met_d_mean, met_d_var, met_d_cnt, met_w)

    print("\n# paste-ready (into meteolibre_model/diffusion/utils.py):")
    print("# SAT_LOSS_WEIGHT  (s_j = 1/Var[Delta_x], mean-normalized)")
    print(
        "SAT_LOSS_WEIGHT = torch.tensor("
        + str([round(float(x), 4) for x in sat_w]) + ", dtype=torch.float32)"
    )
    print("# METAR_LOSS_WEIGHT (s_j = 1/Var[Delta_x], mean-normalized)")
    print(
        "METAR_LOSS_WEIGHT = torch.tensor("
        + str([round(float(x), 4) for x in met_w]) + ", dtype=torch.float32)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute FastNet-style per-channel loss weights "
        "(inverse normalized time-difference variance) for the FlashEdges "
        "dataset."
    )
    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        default="meteolibre-dev/global_sat_metar",
        help="Hugging Face dataset repo id to stream from. "
        "Default 'meteolibre-dev/global_sat_metar'. Pass '' to use local "
        "parquet via --data_files instead.",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default="**/*.parquet",
        help="Recursive glob of local parquet files (used when "
        "--hf_dataset_repo is not set). Default '**/*.parquet'.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="For Hub mode, restrict to one subfolder "
        "(e.g. data_2022_02). None = all subfolders.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of patches to sample (-1 = whole stream).",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=1,
        help="Shuffle buffer rows (1 disables shuffling; fine for stats).",
    )
    parser.add_argument("--prefetch_rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    compute(
        args.hf_dataset_repo or None,
        args.data_files,
        args.data_dir,
        args.num_samples,
        args.shuffle_buffer,
        args.prefetch_rows,
        args.seed,
    )
