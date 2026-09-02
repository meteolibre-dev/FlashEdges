"""
Normalization constants for the FlashEdges global satellite + METAR model.

Channel layout:
  sat_patch_data   (T, 5, H, W)  — [gmgsi_lwir, gmgsi_vis, gmgsi_wv,
                                    gmgsi_sw, elevation]
  metar_patch_data (T, 7, H, W)  — [tmpc, dwpc, mslp, cloud_cover,
                                     p01m(dBZ), wind_u, wind_v]

The METAR p01m channel is in dBZ (Marshall-Palmer) because the dataset applies
the mm/h -> dBZ transform by default.

NOTE: the values below were computed on a 300-sample subset of a single day
(2021-07-14) via ``scripts/compute_mean_std.py``.  They are structurally correct
but should be recomputed over the full 4-year HuggingFace dataset for production
training:

    uv run python scripts/compute_mean_std.py --localrepo . --num_samples -1
"""

import torch

# --- satellite: GMGSI(4) + elevation(1) ---
SAT_MEAN = torch.tensor(
    [123.2937, 46.5135, 169.5546, 125.0362, 677.6422], dtype=torch.float32
)
SAT_STD = torch.tensor(
    [43.3566, 53.4644, 26.8234, 42.3147, 874.6544], dtype=torch.float32
)

# --- v2 dataset: GMGSI(4) + radar(1) + elevation(1) ---
# Channel layout for ``dataset_global_satellite_metar_v2``:
#   sat_patch_data (T, 6, H, W) — [gmgsi_lwir, gmgsi_vis, gmgsi_wv, gmgsi_sw,
#                                   radar_dbz, elevation]
# The GMGSI/elevation stats are carried over from the v1 5-channel block
# (same channels, same position); the radar channel uses simple hand-picked
# stats (mean 15 dBZ, std 20 dBZ) covering the practical DBZH range
# (~[-5, 65] dBZ) — recompute with scripts/compute_mean_std.py for production.
SAT_MEAN_V2 = torch.tensor(
    [123.2937, 46.5135, 169.5546, 125.0362, 10.0, 677.6422], dtype=torch.float32
)
SAT_STD_V2 = torch.tensor(
    [43.3566, 53.4644, 26.8234, 42.3147, 20.0, 874.6544], dtype=torch.float32
)

# --- METAR: [tmpc, dwpc, mslp, cloud_cover, p01m_dBZ, wind_u, wind_v] ---
METAR_MEAN = torch.tensor(
    [24.0332, 15.0477, 1017.3679, 0.2766, -3.0849, 0.574, 0.664],
    dtype=torch.float32,
)
METAR_STD = torch.tensor(
    [10.7002, 7.9932, 8.3278, 0.7042, 9.2245, 6.7808, 6.1393],
    dtype=torch.float32,
)

# --- residual stats (future - last_context_frame) ---
# Placeholder zeros/ones; only used when use_residual=True. Recompute with
# scripts/compute_mean_std.py in residual mode if enabling residual training.
SAT_RESIDUAL_MEAN = torch.zeros(5, dtype=torch.float32)
SAT_RESIDUAL_STD = torch.ones(5, dtype=torch.float32)
METAR_RESIDUAL_MEAN = torch.zeros(7, dtype=torch.float32)
METAR_RESIDUAL_STD = torch.ones(7, dtype=torch.float32)

# v2 6-channel layout residuals (radar included)
SAT_RESIDUAL_MEAN_V2 = torch.zeros(6, dtype=torch.float32)
SAT_RESIDUAL_STD_V2 = torch.ones(6, dtype=torch.float32)

# --- FastNet-style per-channel loss weights (s_j = 1 / Var[Delta_x_j]) -------
# Per-channel inverse variance of the *normalized* time-difference, mean-
# normalized to 1 within each branch over the DYNAMIC channels only. This
# equalizes the per-channel gradient contribution: channels that barely move
# frame-to-frame would otherwise contribute almost nothing to a plain masked-
# MSE, while fast channels dominate. See FastNet (arxiv 2509.17601) eq. 7.
#
# STATIC channels (constant in time, e.g. elevation) get a NEUTRAL weight of
# 1.0 instead of the FastNet weight: their Var[Delta_x] ~ 0 so 1/Var explodes
# and would let them capture ~the entire branch loss even though they are
# trivial targets (FastNet itself keeps orography / land-sea-mask as input-
# only, never forecast targets). We keep elevation as a target for rollout
# integrity but at its old neutral contribution.
#
# Mean-normalization keeps the total loss scale AND the satellite:METAR branch
# balance identical to the previous unweighted masked-mean; only the intra-
# branch per-channel balance is adjusted.
#
# Recompute over the full dataset with:
#     uv run python scripts/compute_loss_weights.py --num_samples -1
# and paste the printed tensors here.
SAT_LOSS_WEIGHT = torch.tensor(
    [0.9208, 1.1572, 0.9409, 0.9811, 0.5], dtype=torch.float32  # elevation=1.0 (static)
)
METAR_LOSS_WEIGHT = torch.tensor(
    [1.1829, 0.9163, 3.9911, 0.2683, 0.1199, 0.3948, 0.3267], dtype=torch.float32
)

# --- v2 6-channel satellite branch: GMGSI(4) + radar(1) + elevation(1) ------
# The 4 GMGSI values carry over from v1 (same channels, same positions);
# elevation keeps its static-channel value. The radar channel uses a neutral
# placeholder (1.0): radar dBZ moves a lot frame-to-frame so its FastNet
# 1/Var[Delta] weight is likely BELOW 1, but the exact value must be measured
# with scripts/compute_loss_weights.py over the v2 dataset (which needs a
# 6-channel-aware update) before trusting the intra-branch balance.
SAT_LOSS_WEIGHT_V2 = torch.tensor(
    [0.9208, 1.1572, 0.9409, 0.9811, 1.0, 0.5], dtype=torch.float32
)
