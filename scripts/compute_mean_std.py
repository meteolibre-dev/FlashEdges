"""
Compute per-channel mean / std for the FlashEdges global satellite + METAR
dataset, for use as normalization constants in training.

Mirrors flashnet's ``compute_mean_std_lightning.py`` but targets the streaming
``FlashEdgesStreamingDataset`` (local parquet or Hugging Face Hub):

  * sat_patch_data   (T, 5, H, W)  — GMGSI(4) + elevation(1)
      - GMGSI channels (0..3): NaN where off-disk / no coverage  -> masked
      - elevation channel (4): floored to -100 for nodata        -> masked (< -1)

  * metar_patch_data (T, 7, H, W)  — [tmpc, dwpc, mslp, cloud_cover,
                                      p01m(dBZ), wind_u, wind_v]
      - valid only where a station reported (dataset returns ``metar_mask``);
        the -10000 sentinel and the sparse-station structure mean we MUST mask
        or the stats are dominated by the sentinel.

Precipitation is read in dBZ (Marshall-Palmer) because the dataset applies the
mm/h -> dBZ transform by default (``precip_to_dbz=True``); the stats therefore
reflect exactly what the model sees.

Usage:
    # stream from the Hugging Face Hub (default repo, no local clone needed):
    uv run python scripts/compute_mean_std.py --num_samples 2000
    uv run python scripts/compute_mean_std.py --num_samples -1            # all

    # restrict to one dated subfolder on the Hub:
    uv run python scripts/compute_mean_std.py --data_dir data_2022_02

    # local parquet instead (recursive glob from the dataset repo root):
    uv run python scripts/compute_mean_std.py --hf_dataset_repo "" \
        --data_files 'data/**/*.parquet' --num_samples 2000
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

# sat_patch_data channel layout: [GMGSI_0, GMGSI_1, GMGSI_2, GMGSI_3, elevation]
SAT_CHANNEL_NAMES = ["gmgsi_lwir", "gmgsi_vis", "gmgsi_wv", "gmgsi_sw", "elevation"]
ELEVATION_CHANNEL_IDX = 4


def _accumulate(masked, sum_acc, sumsq_acc, count_acc):
    """Fold a (T, C, H, W) masked-array slice into per-channel accumulators."""
    sum_acc += masked.sum(axis=(0, 2, 3)).filled(0)
    sumsq_acc += np.ma.power(masked, 2).sum(axis=(0, 2, 3)).filled(0)
    count_acc += masked.count(axis=(0, 2, 3))


def compute_mean_std(
    hf_dataset_repo,
    data_files,
    data_dir,
    num_samples: int,
    shuffle_buffer: int,
    prefetch_rows: int,
    seed: int,
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
        "Calculating mean/std over "
        + (f"{n_iter} samples " if n_iter is not None else "the full stream ")
        + "(streaming dataset, length unknown), "
        "masking NaN / sentinel / nodata..."
    )

    n_sat = len(SAT_CHANNEL_NAMES)
    n_met = len(METAR_FEATURES)
    sat_sum = np.zeros(n_sat, dtype=np.float64)
    sat_sumsq = np.zeros(n_sat, dtype=np.float64)
    sat_count = np.zeros(n_sat, dtype=np.int64)
    met_sum = np.zeros(n_met, dtype=np.float64)
    met_sumsq = np.zeros(n_met, dtype=np.float64)
    met_count = np.zeros(n_met, dtype=np.int64)

    pbar = tqdm(dataset, total=n_iter)
    for i, sample in enumerate(pbar):

        if n_iter is not None and i >= n_iter:
            break

        # --- satellite: mask NaN (GMGSI off-disk) + elevation nodata floor ---
        sat = sample["sat_patch_data"].numpy().astype(np.float32)
        sat_mask = ~np.isnan(sat)
        # elevation channel: nodata is floored to -100; real elevations are >= 0
        sat_mask[:, ELEVATION_CHANNEL_IDX] &= sat[:, ELEVATION_CHANNEL_IDX] > -1.0
        sat_masked = np.ma.array(sat, mask=~sat_mask)
        _accumulate(sat_masked, sat_sum, sat_sumsq, sat_count)

        # --- METAR: use the dataset's valid-station mask exactly ---
        metar = sample["metar_patch_data"].numpy().astype(np.float32)
        mask = sample["metar_mask"].numpy().astype(bool)
        # belt-and-suspenders: also drop any residual sentinel
        mask &= metar != -10000.0
        metar_masked = np.ma.array(metar, mask=~mask)
        _accumulate(metar_masked, met_sum, met_sumsq, met_count)

    sat_mean = np.divide(
        sat_sum, sat_count, out=np.zeros_like(sat_sum), where=sat_count != 0
    )
    sat_std = np.sqrt(
        np.divide(
            sat_sumsq, sat_count, out=np.zeros_like(sat_sumsq), where=sat_count != 0
        )
        - np.square(sat_mean)
    )
    met_mean = np.divide(
        met_sum, met_count, out=np.zeros_like(met_sum), where=met_count != 0
    )
    met_std = np.sqrt(
        np.divide(
            met_sumsq, met_count, out=np.zeros_like(met_sumsq), where=met_count != 0
        )
        - np.square(met_mean)
    )

    print("\n=== sat_patch_data (GMGSI + elevation) ===")
    for name, m, s, c in zip(SAT_CHANNEL_NAMES, sat_mean, sat_std, sat_count):
        print(f"  {name:14s} mean={m:12.4f}  std={s:12.4f}  count={c}")

    print("\n=== metar_patch_data (p01m in dBZ) ===")
    for name, m, s, c in zip(METAR_FEATURES, met_mean, met_std, met_count):
        print(f"  {name:14s} mean={m:12.4f}  std={s:12.4f}  count={c}")

    print("\n# paste-ready:")
    print("sat_mean  =", [round(float(x), 4) for x in sat_mean])
    print("sat_std   =", [round(float(x), 4) for x in sat_std])
    print("metar_mean=", [round(float(x), 4) for x in met_mean])
    print("metar_std =", [round(float(x), 4) for x in met_std])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std for the FlashEdges dataset "
        "(streaming; local parquet or Hugging Face Hub)."
    )
    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        default="meteolibre-dev/global_sat_metar",
        help="Hugging Face dataset repo id to stream from. "
        "Default 'meteolibre-dev/global_sat_metar' (multi-folder layout: "
        "data/, data_2022_02/, ...). Pass an empty string ('') to use local "
        "parquet via --data_files instead.",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default="**/*.parquet",
        help="Recursive glob of local parquet files (used when "
        "--hf_dataset_repo is not set). Default 'data/**/*.parquet'.",
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

    compute_mean_std(
        args.hf_dataset_repo or None,
        args.data_files,
        args.data_dir,
        args.num_samples,
        args.shuffle_buffer,
        args.prefetch_rows,
        args.seed,
    )
