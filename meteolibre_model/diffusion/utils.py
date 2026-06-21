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

# --- METAR: [tmpc, dwpc, mslp, cloud_cover, p01m_dBZ, wind_u, wind_v] ---
METAR_MEAN = torch.tensor(
    [24.0332, 15.0477, 1017.3679, 0.2766, 0.0849, 0.574, 0.664],
    dtype=torch.float32,
)
METAR_STD = torch.tensor(
    [10.7002, 7.9932, 8.3278, 0.7042, 6.2245, 6.7808, 6.1393],
    dtype=torch.float32,
)

# --- residual stats (future - last_context_frame) ---
# Placeholder zeros/ones; only used when use_residual=True. Recompute with
# scripts/compute_mean_std.py in residual mode if enabling residual training.
SAT_RESIDUAL_MEAN = torch.zeros(5, dtype=torch.float32)
SAT_RESIDUAL_STD = torch.ones(5, dtype=torch.float32)
METAR_RESIDUAL_MEAN = torch.zeros(7, dtype=torch.float32)
METAR_RESIDUAL_STD = torch.ones(7, dtype=torch.float32)

# --- FastNet-style per-channel loss weights (s_j = 1 / Var[Delta_x_j]) -------
# Per-channel inverse variance of the *normalized* time-difference, mean-
# normalized to 1 within each branch. This equalizes the per-channel gradient
# contribution: channels that barely move frame-to-frame (e.g. elevation, mslp)
# would otherwise contribute almost nothing to a plain masked-MSE, while fast
# channels (GMGSI LWIR, wind) dominate. See FastNet (arxiv 2509.17601) eq. 7.
#
# Mean-normalization keeps the total loss scale AND the satellite:METAR branch
# balance identical to the previous unweighted masked-mean; only the intra-
# branch per-channel balance is adjusted.
#
# These defaults are all-ones => identical to the previous unweighted loss.
# RECOMPUTE over the full dataset with:
#     uv run python scripts/compute_loss_weights.py --num_samples -1
# and paste the printed tensors here.
SAT_LOSS_WEIGHT = torch.ones(5, dtype=torch.float32)
METAR_LOSS_WEIGHT = torch.ones(7, dtype=torch.float32)
