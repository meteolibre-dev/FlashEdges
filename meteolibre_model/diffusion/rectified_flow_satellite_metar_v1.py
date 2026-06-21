"""
Shortcut Rectified Flow for the FlashEdges global satellite + METAR model.

Adapted from flashnet's ``rectified_flow_lightning_shortcut_xpred_blur_v2.py``
(https://arxiv.org/pdf/2410.12557) with two key differences:

  1. Dual branches are **satellite** (GMGSI 4ch + elevation 1ch = 5ch) and
     **METAR** (7ch, p01m in dBZ) instead of satellite + lightning.

  2. METAR loss is **masked by the valid-station mask**.  METAR observations
     are extremely sparse (~5e-5 pixel fill), so an unmasked loss would be
     dominated by the -10000 sentinel and teach the model nothing.  The
     ``metar_mask`` from the dataset selects only the pixels where a station
     actually reported.

Supports 'linear' and 'polynomial' interpolation schedules, x-prediction
parametrization, optional context blur augmentation, and residual targets.
"""

import math
import random

import torch
import torch.nn.functional as F

from meteolibre_model.diffusion.utils import (
    SAT_MEAN,
    SAT_STD,
    METAR_MEAN,
    METAR_STD,
    SAT_RESIDUAL_MEAN,
    SAT_RESIDUAL_STD,
    METAR_RESIDUAL_MEAN,
    METAR_RESIDUAL_STD,
)

# -- Parameters --
CLIP_MIN = -4
# METAR upper clamp: dBZ precipitation can reach ~60, winds/temps are bounded.
METAR_CLIP_MAX = 15.0
SHORTCUT_M = 128  # base steps (M=128 as in the paper)
SHORTCUT_K = 0.25  # fraction of batch for self-consistency


def normalize(sat_data, metar_data, device):
    """Normalize sat (5ch) and metar (7ch) batches using precomputed stats.

    GMGSI NaN (off-disk) should be filled with 0 *before* calling this; the
    caller is responsible for building the sat mask.  METAR -10000 sentinel
    pixels collapse to CLIP_MIN after normalize + clamp, which is the no-data
    convention the model learns to ignore.
    """
    sat_data = (
        sat_data
        - SAT_MEAN.to(device).view(1, -1, 1, 1, 1)
    ) / SAT_STD.to(device).view(1, -1, 1, 1, 1)
    sat_data = sat_data.clamp(CLIP_MIN, 4)

    metar_data = (
        metar_data
        - METAR_MEAN.to(device).view(1, -1, 1, 1, 1)
    ) / METAR_STD.to(device).view(1, -1, 1, 1, 1)
    metar_data = metar_data.clamp(CLIP_MIN, METAR_CLIP_MAX)

    return sat_data, metar_data


def denormalize(sat_data, metar_data, device):
    """Denormalize back to physical units."""
    sat_data = (
        sat_data * SAT_STD.to(device).view(1, -1, 1, 1, 1)
        + SAT_MEAN.to(device).view(1, -1, 1, 1, 1)
    )
    metar_data = (
        metar_data * METAR_STD.to(device).view(1, -1, 1, 1, 1)
        + METAR_MEAN.to(device).view(1, -1, 1, 1, 1)
    )
    return sat_data, metar_data


def normalize_residual(x0, c_sat, device):
    """Normalize residual target (sat + metar channels concatenated)."""
    mean = torch.cat([SAT_RESIDUAL_MEAN, METAR_RESIDUAL_MEAN]).to(device).view(1, -1, 1, 1, 1)
    std = torch.cat([SAT_RESIDUAL_STD, METAR_RESIDUAL_STD]).to(device).view(1, -1, 1, 1, 1)
    return (x0 - mean) / std


def denormalize_residual(x0, c_sat, device):
    """Denormalize residual back to normalized-data space."""
    mean = torch.cat([SAT_RESIDUAL_MEAN, METAR_RESIDUAL_MEAN]).to(device).view(1, -1, 1, 1, 1)
    std = torch.cat([SAT_RESIDUAL_STD, METAR_RESIDUAL_STD]).to(device).view(1, -1, 1, 1, 1)
    return x0 * std + mean


def get_x_t_rf(x0, x1, t, interpolation="linear"):
    """Interpolated point x_t.
    - 'linear':     x_t = (1 - t) * x0 + t * x1
    - 'polynomial': x_t = (1 - sqrt(t)) * x0 + sqrt(t) * x1
    """
    if interpolation == "linear":
        return (1 - t) * x0 + t * x1
    elif interpolation == "polynomial":
        alpha = 1 - t ** 0.5
        return alpha * x0 + (1 - alpha) * x1
    else:
        raise ValueError(f"Unknown interpolation schedule: {interpolation}")


def apply_blur_with_sigma_batched(x, blur_sigma, n_bins=8, min_kernel=0, sigma_factor=3):
    """Vectorized Gaussian blur via sigma binning.
    blur_sigma: (B,) tensor, sigma in pixels.
    """
    b, c, t, h, w = x.shape
    out = torch.zeros_like(x)

    sigma_max = blur_sigma.max().item()
    bin_edges = torch.linspace(0, sigma_max + 1e-6, n_bins + 1, device=x.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_ids = torch.bucketize(blur_sigma, bin_edges[1:])

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not mask.any():
            continue

        s = bin_centers[bin_idx].item()
        x_bin = x[mask]
        b_bin = x_bin.shape[0]

        if s < 0.1:
            out[mask] = x_bin
            continue

        k = max(min_kernel, 2 * int(sigma_factor * s) + 1)
        coords = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * s ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d.expand(c * t, 1, k, k)

        x_flat = x_bin.reshape(b_bin, c * t, h, w)
        pad = k // 2
        blurred = F.conv2d(x_flat, kernel, padding=pad, groups=c * t)
        out[mask] = blurred.reshape(b_bin, c, t, h, w)

    return out


def trainer_step(
    model,
    batch,
    device,
    sigma=0.0,
    parametrization="standard",
    interpolation="linear",
    use_residual=True,
    metar_loss_weight=1.0,
    metar_drop_frac=0.20,
):
    """One flow-matching training step with x-prediction.

    Returns (total_loss, loss_sat, loss_metar).
    """
    if parametrization != "standard":
        raise ValueError("Only 'standard' parametrization is supported for x-prediction.")

    # (B, C, T, H, W) after permute
    sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)       # (B, 5, T, H, W)
    metar_data = batch["metar_patch_data"].permute(0, 2, 1, 3, 4)   # (B, 7, T, H, W)
    metar_mask = batch["metar_mask"].permute(0, 2, 1, 3, 4)         # (B, 7, T, H, W)

    b, c_sat, t_dim, h, w = sat_data.shape
    _, c_metar, _, _, _ = metar_data.shape

    # --- sat mask: NaN where GMGSI off-disk (before normalize) ---
    sat_mask = ~torch.isnan(sat_data)
    sat_data = torch.where(torch.isnan(sat_data), torch.zeros_like(sat_data), sat_data)

    sat_data, metar_data = normalize(sat_data, metar_data, device)
    batch_data = torch.cat([sat_data, metar_data], dim=1)  # (B, 12, T, H, W)

    x_context = batch_data[:, :, : model.context_frames]

    if use_residual:
        x0 = (
            batch_data[:, :, model.context_frames:]
            - batch_data[:, :, model.context_frames - 1 : model.context_frames]
        )
        x0 = normalize_residual(x0, c_sat, device)
    else:
        x0 = batch_data[:, :, model.context_frames:]

    context_info = batch["spatial_position"]
    x1 = torch.randn_like(x0)

    # ====================== EMPIRICAL (flow-matching) PART ======================
    num_emp = b
    x0_emp = x0
    x1_emp = x1
    context_info_emp = context_info

    # masks restricted to target frames
    sat_mask_emp = sat_mask[:num_emp, :, model.context_frames:]
    metar_mask_emp = metar_mask[:num_emp, :, model.context_frames:].bool()

    # Stratified t sampling with 32 bins
    n_bins = 32
    bin_size = 1.0 / n_bins
    bin_indices = torch.randperm(n_bins, device=device).repeat_interleave(
        (num_emp + n_bins - 1) // n_bins
    )[:num_emp]
    t_emp = (bin_indices.float() + torch.rand(num_emp, device=device)) * bin_size
    t_emp = t_emp[torch.randperm(num_emp, device=device)]

    # progressive context blur augmentation
    if sigma > 0:
        # eps = torch.randn(num_emp, device=device)
        # t_emp_blur = torch.sigmoid(1.4 + 1.8 * eps).clamp(1e-4, 1 - 1e-4)

        t_emp_blur = torch.rand(num_emp, device=device)

        blur_sigma = t_emp_blur * sigma
        x_context_t = apply_blur_with_sigma_batched(x_context, blur_sigma)

        frame_noise_rand = torch.rand(b, model.context_frames, device=device)
        noise_sigma = (blur_sigma.unsqueeze(1) / sigma * 0.05 * frame_noise_rand)
        noise_sigma = noise_sigma.view(b, 1, model.context_frames, 1, 1)
        x_context_t = x_context_t + noise_sigma * torch.randn_like(x_context)
    else:
        x_context_t = x_context

    # --- METAR context dropout (self-supervised spatial fill) ----------
    # Randomly hide a fraction of the valid-station METAR pixels in the
    # conditioning context, replacing them with the no-data sentinel (CLIP_MIN,
    # the same value the model already sees at non-station pixels). The model
    # must then reconstruct those pixels in the forecast from satellite + the
    # remaining ~80% of stations, instead of just echoing the sparse input.
    # Over training this teaches spatial generalization, so at inference the
    # model can emit a plausible *full* METAR image even where no station
    # reported. The drop mask is per-sample and spatial (B, H, W): stations are
    # fixed in space, so a hidden station is dropped across all context frames
    # and all 7 METAR channels at once. The loss is unaffected (it still covers
    # every valid target-frame station, including the hidden ones -- which is
    # exactly the reconstruction signal we want). METAR is ~8e-5 fill, so we
    # drop among *valid* positions ("drop 20% of the HxW grid" would be a no-op
    # on the already-empty pixels).
    if metar_drop_frac > 0:
        metar_ctx_valid = metar_mask[:, :, : model.context_frames].bool()  # (B,7,ctx,H,W)
        present = metar_ctx_valid.any(dim=1).any(dim=1)                    # (B,H,W)
        drop = (
            torch.rand(b, h, w, device=device) < metar_drop_frac
        ) & present                                                         # (B,H,W)
        if drop.any():
            drop_e = drop.view(b, 1, 1, h, w)  # broadcast over (c_metar, ctx)
            # clone first: x_context_t may be a view into batch_data, and the
            # last context frame feeds the residual target, so we must not
            # mutate it in place.
            x_context_t = x_context_t.clone()
            x_context_t[:, c_sat:] = torch.where(
                drop_e,
                torch.full_like(x_context_t[:, c_sat:], CLIP_MIN),
                x_context_t[:, c_sat:],
            )

    xt_emp = get_x_t_rf(x0_emp, x1_emp, t_emp.view(num_emp, 1, 1, 1, 1), interpolation)

    # da/dt for v-loss weighting
    if interpolation == "linear":
        da_dt = torch.full_like(t_emp, -1.0)
    else:  # polynomial: alpha(t) = 1 - t^(1/2)
        da_dt = -0.5 / (t_emp ** 0.5 + 1e-8)
    da_dt = da_dt.view(num_emp, 1, 1, 1, 1)

    # model predicts clean target (x-prediction)
    model_input_emp = torch.cat([x_context_t, xt_emp], dim=2)
    context_global_emp = torch.cat(
        [context_info_emp, t_emp.unsqueeze(1), torch.zeros_like(t_emp).unsqueeze(1)],
        dim=1,
    )

    sat_x_pred_emp, metar_x_pred_emp = model(
        model_input_emp[:, :c_sat].float(),
        model_input_emp[:, c_sat:].float(),
        context_global_emp.float(),
    )

    x_sat_pred_emp = sat_x_pred_emp[:, :, model.context_frames:]
    x_metar_pred_emp = metar_x_pred_emp[:, :, model.context_frames:]

    # loss weighting (1/t^2 upweighting of small t, clamped)
    weight = 1.0 / (t_emp.view(b, 1, 1, 1, 1) + 1e-2) ** 2
    weight = weight.clamp(0.9, 10.0)

    # --- sat loss: masked by GMGSI validity ---
    sat_diff = weight * (x_sat_pred_emp - x0_emp[:, :c_sat]) ** 2
    if sat_mask_emp.any():
        loss_sat = sat_diff[sat_mask_emp].mean()
    else:
        loss_sat = sat_diff.mean()

    # --- METAR loss: masked by valid-station mask (sparse!) ---
    metar_diff = weight * (x_metar_pred_emp - x0_emp[:, c_sat:]) ** 2
    if metar_mask_emp.any():
        loss_metar = metar_diff[metar_mask_emp].mean()
    else:
        # No station in this batch — don't penalize metar, don't NaN the step.
        loss_metar = torch.zeros((), device=device)

    total = loss_sat + metar_loss_weight * loss_metar
    return total, loss_sat, loss_metar


def full_image_generation(
    model,
    batch,
    steps=256,
    device="cuda",
    parametrization="standard",
    interpolation="linear",
    nb_element=1,
    normalize_input=True,
    use_residual=True,
):
    """Generate forecast frames via Euler integration of the RF ODE."""
    model.eval()
    with torch.no_grad():
        sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
        metar_data = batch["metar_patch_data"].permute(0, 2, 1, 3, 4)

        b, c_sat, t, h, w = sat_data.shape
        b, c_metar, t, h, w = metar_data.shape

        nb_forecasted_frame = t - model.context_frames

        # fill sat NaN before normalize
        sat_data = torch.where(
            torch.isnan(sat_data), torch.zeros_like(sat_data), sat_data
        )

        if normalize_input:
            sat_data, metar_data = normalize(sat_data, metar_data, device=device)

        batch_data = torch.cat([sat_data, metar_data], dim=1)[0:nb_element]

        x_context = batch_data[:, :, : model.context_frames]
        last_context = x_context[:, :, model.context_frames - 1 : model.context_frames]

        context_info = batch["spatial_position"].to(device)[0:nb_element]

        batch_size, nb_channel, nb_context, h, w = x_context.shape
        x_t = torch.randn(
            batch_size, nb_channel, nb_forecasted_frame, h, w, device=device
        )

        d_const = 1.0 / steps
        t_val = 1.0

        for _ in range(steps):
            t_batch = torch.full((batch_size,), t_val, device=device)
            d_batch = torch.full((batch_size,), 0.0, device=device)

            x_context_t = x_context
            model_input = torch.cat([x_context_t, x_t], dim=2)
            context_global = torch.cat(
                [context_info, t_batch.unsqueeze(1), d_batch.unsqueeze(1)], dim=1
            )

            sat_x_pred, metar_x_pred = model(
                model_input[:, :c_sat].float(),
                model_input[:, c_sat:].float(),
                context_global.float(),
            )

            x_pred = torch.cat([sat_x_pred, metar_x_pred], dim=1)[
                :, :, model.context_frames :
            ]

            # Euler step: x_{t-dt} = x_t - v(x_t, t) * dt
            if interpolation == "polynomial":
                s_theta = (x_t - x_pred) / (2 * t_val + 1e-8)
            else:
                s_theta = (x_t - x_pred) / t_val
            x_t = x_t - s_theta * d_const
            x_t = x_t.clamp(-7, 8)

            t_val -= d_const

        if use_residual:
            x_t = denormalize_residual(x_t, c_sat, device)
            x_t = x_t + last_context.expand_as(x_t)

        # keep no-data values from last context frame (sat NaN / metar sentinel)
        x_t = torch.where(last_context == CLIP_MIN, last_context, x_t)

        generated = x_t.cpu()
        target = batch_data[:, :, model.context_frames :].cpu()

    model.train()
    return generated, target
