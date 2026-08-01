"""
Rectified Flow for satellite + METAR — **v2 with Forward Explorative Modeling (XM-K)**.

This is a drop-in replacement for
``rectified_flow_satellite_metar_v1.trainer_step`` that adds **Forward XM**
(Explorative Modeling, arXiv 2607.27372) on top of the existing flow-matching
training loop.

What changes vs v1
------------------
* **K noise draws per datapoint.**  Instead of a single noise endpoint ``x1``
  per sample, we draw ``K`` independent structured-Gaussian noises, build ``K``
  interpolated points ``x_t`` (all sharing the *same* target ``x0``, timestep
  ``t``, and conditioning context — only the noise varies), and run the model
  once with a ``K×B`` batch.
* **Best-of-K loss (Forward XM).**  A per-sample reconstruction loss is
  computed for every candidate.  For each batch element only the
  **lowest-loss candidate** receives gradient (``torch.min`` on a
  ``(K, B)`` tensor does this automatically — the min's backward passes
  gradient 1 to the argmin, 0 to the rest).
* **Inference is unchanged.**  ``full_image_generation`` is re-exported from
  v1 verbatim; the model architecture and sampling loop are identical.

Why (from the paper)
--------------------
* Generation factorization (many small denoising steps) *reduces* but never
  *eliminates* residual multimodality at each step.  Forward XM attacks this
  residue directly by finding which noise draw best couples to the current
  datapoint, instead of pairing noise→data at random.
* The paper shows gains even on converged / SOTA recipes (Table 1, Figures 7-8),
  so fine-tuning a pretrained rectified-flow model with Forward XM should
  improve the noise→data coupling without changing the architecture.
* For continuous conditioning (our case: spatial position + satellite context),
  Forward XM is the correct choice — it fixes the condition and only varies the
  noise, so the condition-mismatch problem that affects Reverse XM never arises.

Cost
----
Each training step costs ``K×`` forward-pass FLOPs (K=2 → 2×).  The paper
folds K into the batch dimension so accelerators process it efficiently; we do
the same.  Memory scales with ``K×`` the activation buffer; use gradient
checkpointing if OOM.

Usage in training scripts
-------------------------
Replace::

    from meteolibre_model.diffusion.rectified_flow_satellite_metar_v1 import trainer_step

with::

    from meteolibre_model.diffusion.rectified_flow_satellite_metar_v2 import trainer_step

and add ``xm_k=2`` to the call (default).  ``full_image_generation`` is
re-exported from v1 so inference code needs no changes.

Warm-up suggestion
~~~~~~~~~~~~~~~~~~
When fine-tuning a model pretrained *without* XM, the objective shifts from
``E_z[loss]`` to ``min_z loss``, which can cause a transient loss bump.  A
simple warm-up: start with ``xm_k=1`` (standard training) for the first N
steps, then switch to ``xm_k=2`` for the rest.  Alternatively, use
``xm_mix < 1.0`` to blend the standard (K=1) and XM (K=2) losses.
"""

# Re-export everything unchanged from v1 — only trainer_step is overridden.
from meteolibre_model.diffusion.rectified_flow_satellite_metar_v1 import (  # noqa: F401
    CLIP_MIN,
    METAR_CLIP_MAX,
    SHORTCUT_M,
    SHORTCUT_K,
    METAR_RESIDUAL_FEATURES,
    METAR_RESIDUAL_IDX,
    metar_residual_channel_mask,
    build_residual_target,
    reconstruct_residual,
    normalize,
    denormalize,
    normalize_residual,
    denormalize_residual,
    structured_gaussian_noise,
    get_x_t_rf,
    apply_blur_with_sigma_batched,
    full_image_generation,
)

import torch
import torch.nn.functional as F

from meteolibre_model.diffusion.utils import (
    SAT_LOSS_WEIGHT,
    METAR_LOSS_WEIGHT,
)


def _per_sample_sat_loss(x_sat_pred, x0_sat, sat_mask, weight, sat_lw):
    """Per-sample, channel-weighted satellite reconstruction loss.

    Returns ``(B,)`` per-sample scalar loss **and** ``(B, c_sat)`` per-channel
    per-sample loss (for diagnostics).
    """
    sat_diff = weight * (x_sat_pred - x0_sat) ** 2          # (B, c, T, H, W)
    sat_m = sat_mask.float()
    sat_cnt = sat_m.sum(dim=(2, 3, 4)).clamp(min=1.0)        # (B, c)
    per_chan = (sat_diff * sat_m).sum(dim=(2, 3, 4)) / sat_cnt  # (B, c)
    per_sample = (per_chan * sat_lw).mean(dim=1)             # (B,)
    return per_sample, per_chan


def _per_sample_metar_loss(x_metar_pred, x0_metar, metar_mask, weight, metar_lw):
    """Per-sample, channel-weighted METAR reconstruction loss (masked)."""
    metar_diff = weight * (x_metar_pred - x0_metar) ** 2
    met_m = metar_mask.float()
    met_cnt = met_m.sum(dim=(2, 3, 4)).clamp(min=1.0)        # (B, c)
    per_chan = (metar_diff * met_m).sum(dim=(2, 3, 4)) / met_cnt  # (B, c)
    per_sample = (per_chan * metar_lw).mean(dim=1)          # (B,)
    return per_sample, per_chan


def _per_sample_grad_loss(x_sat_pred, x0_sat, sat_mask, weight):
    """Per-sample horizontal-gradient regularization loss (satellite only)."""
    gy_p, gx_p = torch.gradient(x_sat_pred, dim=(-2, -1))
    gy_t, gx_t = torch.gradient(x0_sat, dim=(-2, -1))
    grad_err = (gy_p - gy_t) ** 2 + (gx_p - gx_t) ** 2
    sat_m = sat_mask.float()
    sat_cnt = sat_m.sum(dim=(1, 2, 3, 4)).clamp(min=1.0)
    per_sample = (weight * grad_err * sat_m).sum(dim=(1, 2, 3, 4)) / sat_cnt  # (B,)
    return per_sample


def _per_sample_tgrad_loss(x_sat_pred, x0_sat, sat_mask, weight):
    """Per-sample temporal-gradient regularization loss (satellite only).

    Forward difference over forecast frames; both adjacent frames must be valid.
    """
    dT_p = x_sat_pred[:, :, 1:] - x_sat_pred[:, :, :-1]
    dT_t = x0_sat[:, :, 1:] - x0_sat[:, :, :-1]
    pair = sat_mask[:, :, 1:] & sat_mask[:, :, :-1]
    pair_m = pair.float()
    pair_cnt = pair_m.sum(dim=(1, 2, 3, 4)).clamp(min=1.0)
    # weight spans all frames; take the later-frame half
    wt = weight if weight.shape[2] == 1 else weight[:, :, 1:]
    per_sample = (wt * (dT_p - dT_t) ** 2 * pair_m).sum(dim=(1, 2, 3, 4)) / pair_cnt
    return per_sample


def trainer_step(
    model,
    batch,
    device,
    sigma=0.0,
    parametrization="standard",
    interpolation="linear",
    use_residual=True,
    metar_loss_weight=0.05,
    metar_drop_frac=0.05,
    noise_rho=0.0,
    temporal_weight_scale=1.0,
    grad_weight=0.,
    temporal_grad_weight=0.,
    # ── Forward XM parameters ────────────────────────────────────────────
    xm_k=2,
    xm_mix=1.0,
):
    """One flow-matching training step with **Forward XM** (best-of-K noise).

    For each batch element we draw ``xm_k`` independent noise endpoints, build
    ``xm_k`` interpolated points ``x_t`` (same ``x0``, same ``t``, same
    context — only noise differs), run the model once on the concatenated
    ``xm_k * B`` batch, compute a per-sample loss for each candidate, and
    backpropagate only through the **minimum-loss candidate** per sample
    (``torch.min`` over the K axis).

    Parameters
    ----------
    xm_k : int
        Number of exploration candidates (noise draws) per datapoint.
        ``xm_k=1`` recovers the standard v1 training loss exactly.
        ``xm_k=2`` is the recommended starting point for fine-tuning.
    xm_mix : float
        Blend between standard loss (K=1, random noise) and XM loss (K=xm_k,
        best-of-K).  ``1.0`` = pure Forward XM.  ``0.0`` = standard training.
        Values in (0, 1) mix both, useful for a warm-up when fine-tuning a
        model pretrained without XM (avoids the objective-shift transient).

    **Selection criterion: satellite loss only.**
    The winner is chosen by the satellite reconstruction loss (dense, full-field),
    not the total loss.  METAR is extremely sparse (~5e-5 fill) so its per-sample
    loss is unreliable for selection, and the gradient regularizers are
    auxiliary.  However, the **full** loss (sat + metar + regs) is backpropagated
    through the winner, so all branches still receive gradient.

    Returns
    -------
    (total_loss, loss_sat, loss_metar, components)

    ``components`` adds two XM-specific diagnostics:
    * ``xm_winner_rate``: fraction of batch elements won by candidate 0
      (for K=2, ~0.5 means exploration is balanced; deviation suggests one
      noise draw is systematically better, which can indicate the noise
      structure is too similar across draws).
    * ``xm_loss_reduction``: mean relative loss reduction of the winner vs
      the average candidate (0 = no benefit from exploration, higher = more).
    """
    if parametrization != "standard":
        raise ValueError("Only 'standard' parametrization is supported for x-prediction.")

    K = xm_k
    assert K >= 1, "xm_k must be >= 1"

    # ── Data preparation (identical to v1) ───────────────────────────────
    sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)       # (B, 5, T, H, W)
    metar_data = batch["metar_patch_data"].permute(0, 2, 1, 3, 4)   # (B, 7, T, H, W)
    metar_mask = batch["metar_mask"].permute(0, 2, 1, 3, 4)         # (B, 7, T, H, W)

    b, c_sat, t_dim, h, w = sat_data.shape
    _, c_metar, _, _, _ = metar_data.shape

    # --- sat mask: NaN where GMGSI off-disk (before normalize) ---
    sat_mask = ~torch.isnan(sat_data)
    sat_data = torch.where(torch.isnan(sat_data), torch.zeros_like(sat_data), sat_data)

    sat_data, metar_data = normalize(sat_data, metar_data, device)
    sat_data = torch.where(sat_mask, sat_data, torch.zeros_like(sat_data))
    metar_data = torch.where(
        metar_mask.bool(), metar_data, torch.zeros_like(metar_data)
    )
    batch_data = torch.cat([sat_data, metar_data], dim=1)  # (B, 12, T, H, W)

    x_context = batch_data[:, :, : model.context_frames]

    if use_residual:
        x0 = build_residual_target(
            batch_data, model.context_frames, c_sat, c_metar, device
        )
    else:
        x0 = batch_data[:, :, model.context_frames:]

    context_info = batch["spatial_position"]

    # ── Context augmentation (identical to v1, applied ONCE before tiling) ──
    # The same augmented context is shared across all K candidates — only the
    # noise varies (the paper: "same data sample, timestep, and condition, with
    # only the noise varying").
    if sigma > 0:
        eps = torch.randn(b, device=device)
        t_emp_blur = torch.sigmoid(1.4 + 1.8 * eps).clamp(1e-4, 1 - 1e-4)
        blur_sigma = t_emp_blur * sigma
        sat_ctx_t = apply_blur_with_sigma_batched(x_context[:, :c_sat], blur_sigma)
        x_context_t = torch.cat([sat_ctx_t, x_context[:, c_sat:]], dim=1)
    else:
        x_context_t = x_context

    # --- METAR context dropout (same as v1, before tiling) ---
    if metar_drop_frac > 0:
        metar_ctx_valid = metar_mask[:, :, : model.context_frames].bool()
        present = metar_ctx_valid.any(dim=1).any(dim=1)                    # (B,H,W)
        drop = (
            torch.rand(b, h, w, device=device) < metar_drop_frac
        ) & present
        if drop.any():
            drop_e = drop.view(b, 1, 1, h, w)
            x_context_t = x_context_t.clone()
            x_context_t[:, c_sat:] = torch.where(
                drop_e,
                torch.zeros_like(x_context_t[:, c_sat:]),
                x_context_t[:, c_sat:],
            )

    # ── Timestep sampling (stratified, same as v1, sampled ONCE per batch element) ──
    # All K candidates for a given batch element share the same t (only noise
    # varies, per the paper).
    n_bins = 32
    bin_size = 1.0 / n_bins
    bin_indices = torch.randperm(n_bins, device=device).repeat_interleave(
        (b + n_bins - 1) // n_bins
    )[:b]
    t_emp = (bin_indices.float() + torch.rand(b, device=device)) * bin_size
    t_emp = t_emp[torch.randperm(b, device=device)]

    # da/dt for v-loss weighting
    if interpolation == "linear":
        da_dt = torch.full_like(t_emp, -1.0)
    else:
        da_dt = -0.5 / (t_emp ** 0.5 + 1e-8)

    # ── Tile to (K*B) — repeat each sample K times ──────────────────────
    # repeat_interleave([a,b], K) → [a,a,...,b,b,...]  so the K candidates
    # for sample i occupy rows [i*K : (i+1)*K].
    def tile(x):
        return x.repeat_interleave(K, dim=0)

    x0_tiled = tile(x0)                                              # (K*B, C, T, H, W)
    x_context_tiled = tile(x_context_t)                              # (K*B, C, ctx, H, W)
    context_info_tiled = tile(context_info)                           # (K*B, ctx_dim)
    sat_mask_tiled = tile(sat_mask[:, :, model.context_frames:])     # (K*B, c_sat, T, H, W)
    metar_mask_tiled = tile(metar_mask[:, :, model.context_frames:].bool())  # (K*B, c_met, T, H, W)
    t_tiled = tile(t_emp)                                            # (K*B,)
    da_dt_tiled = tile(da_dt)                                        # (K*B,)

    # ── K independent noise draws ────────────────────────────────────────
    # Each candidate gets its own structured-Gaussian noise endpoint x1.
    # The paper: "each explored candidate uses a different input noise draw."
    if K == 1:
        x1 = structured_gaussian_noise(x0.shape, x0.device, x0.dtype, rho=noise_rho)
    else:
        noise_list = [
            structured_gaussian_noise(x0.shape, x0.device, x0.dtype, rho=noise_rho)
            for _ in range(K)
        ]
        x1 = torch.stack(noise_list, dim=0).reshape(K * b, *x0.shape[1:])  # (K*B, C, T, H, W)

    # ── Build x_t for all K candidates ───────────────────────────────────
    xt = get_x_t_rf(
        x0_tiled, x1, t_tiled.view(K * b, 1, 1, 1, 1), interpolation
    )

    # ── Loss weight (1/t² upweighting + temporal ramp) ──────────────────
    weight = 1.0 / (t_tiled.view(K * b, 1, 1, 1, 1) + 1e-2) ** 2
    weight = weight.clamp(0.9, 10.0)

    n_forecast = x0_tiled.shape[2]
    if temporal_weight_scale > 0 and n_forecast > 1:
        ramp = torch.arange(1, n_forecast + 1, device=device, dtype=weight.dtype)
        ramp = ramp / ramp.mean()
        blend = (1.0 - temporal_weight_scale) + temporal_weight_scale * ramp
        weight = weight * blend.view(1, 1, n_forecast, 1, 1)

    # ── Channel loss weights ────────────────────────────────────────────
    sat_lw = SAT_LOSS_WEIGHT.to(device)[:c_sat]        # (c_sat,)
    metar_lw = METAR_LOSS_WEIGHT.to(device)[:c_metar]   # (c_metar,)

    # ── Forward pass (single call with K*B batch) ───────────────────────
    model_input = torch.cat([x_context_tiled, xt], dim=2)  # (K*B, C, ctx+T, H, W)
    context_global = torch.cat(
        [context_info_tiled, torch.zeros_like(t_tiled).unsqueeze(1), t_tiled.unsqueeze(1)],
        dim=1,
    )

    sat_x_pred, metar_x_pred = model(
        model_input[:, :c_sat].float(),
        model_input[:, c_sat:].float(),
        context_global.float(),
        metar_ref=model_input[:, c_sat:].float(),
    )

    x_sat_pred = sat_x_pred[:, :, model.context_frames:]      # (K*B, c_sat, T, H, W)
    x_metar_pred = metar_x_pred[:, :, model.context_frames:]  # (K*B, c_met, T, H, W)

    x0_sat = x0_tiled[:, :c_sat]
    x0_metar = x0_tiled[:, c_sat:]

    # ── Per-sample, per-candidate losses ────────────────────────────────
    sat_loss_ps, sat_per_chan_ps = _per_sample_sat_loss(
        x_sat_pred, x0_sat, sat_mask_tiled, weight, sat_lw
    )  # (K*B,), (K*B, c_sat)
    metar_loss_ps, metar_per_chan_ps = _per_sample_metar_loss(
        x_metar_pred, x0_metar, metar_mask_tiled, weight, metar_lw
    )  # (K*B,), (K*B, c_met)

    if grad_weight > 0:
        grad_loss_ps = _per_sample_grad_loss(
            x_sat_pred, x0_sat, sat_mask_tiled, weight
        )
    else:
        grad_loss_ps = torch.zeros(K * b, device=device)

    if temporal_grad_weight > 0:
        tgrad_loss_ps = _per_sample_tgrad_loss(
            x_sat_pred, x0_sat, sat_mask_tiled, weight
        )
    else:
        tgrad_loss_ps = torch.zeros(K * b, device=device)

    # Total per-sample loss: (K*B,)
    total_ps = (
        sat_loss_ps
        + metar_loss_weight * metar_loss_ps
        + grad_weight * grad_loss_ps
        + temporal_grad_weight * tgrad_loss_ps
    )

    # ── Forward XM selection: min over K candidates ──────────────────────
    # Reshape to (K, B) so the min is taken per batch element across the K
    # noise draws.
    #
    # **Selection criterion: satellite reconstruction loss only.**
    # The satellite branch is the dense, full-field signal (every pixel is
    # valid), so sat_loss is the reliable measure of how well a noise draw
    # couples to the datapoint.  METAR is extremely sparse (~5e-5 pixel fill),
    # so its per-sample loss is noisy and can be zero for batches with no
    # stations — using it for selection would be unreliable.  The gradient
    # regularizers are auxiliary and should not drive candidate selection.
    #
    # We select the winner by sat_loss, but backpropagate the **full** loss
    # (sat + metar + regs) through the winner — so all branches still receive
    # gradient for the selected candidate.
    sat_loss_kb = sat_loss_ps.view(K, b)                   # (K, B)
    total_kb = total_ps.view(K, b)                         # (K, B)
    arange_b = torch.arange(b, device=device)

    if K == 1:
        winner_idx = torch.zeros(b, device=device, dtype=torch.long)
    else:
        winner_idx = sat_loss_kb.argmin(dim=0)              # (B,) — SAT-only selection

    winner_total = total_kb[winner_idx, arange_b]           # (B,) full loss of winner
    xm_loss = winner_total.mean()                           # scalar

    # ── Optional warm-up blend with standard (K=1) loss ──────────────────
    # When xm_mix < 1, also compute the standard loss (mean over all K
    # candidates, equivalent to K=1 expectation) and blend.  This smooths the
    # objective shift when fine-tuning a model pretrained without XM.
    if xm_mix < 1.0:
        standard_loss = total_ps.mean()  # mean over all K*B → E_z[loss]
        total = (1.0 - xm_mix) * standard_loss + xm_mix * xm_loss
    else:
        total = xm_loss

    # ── Diagnostics ──────────────────────────────────────────────────────
    # Per-channel losses for the *winning* candidates (gather along K axis).
    sat_per_chan_kb = sat_per_chan_ps.view(K, b, c_sat)       # (K, B, c_sat)
    metar_per_chan_kb = metar_per_chan_ps.view(K, b, c_met)   # (K, B, c_met)
    sat_per_chan_diag = sat_per_chan_kb[winner_idx, arange_b].mean(dim=0)
    metar_per_chan_diag = metar_per_chan_kb[winner_idx, arange_b].mean(dim=0)

    # Winner rate: fraction of batch won by candidate 0 (for K=2, ~0.5 is healthy)
    xm_winner_rate = (winner_idx == 0).float().mean().detach()

    # Loss reduction: how much better the winner's SAT loss is vs the mean
    # candidate's SAT loss (measures how much exploration helps)
    mean_sat_candidate = sat_loss_kb.mean(dim=0).detach()    # (B,)
    winner_sat = sat_loss_kb[winner_idx, arange_b].detach()  # (B,)
    xm_loss_reduction = ((mean_sat_candidate - winner_sat) / (mean_sat_candidate + 1e-8)).mean()

    # Report loss_sat / loss_metar as the winning candidates' values (what
    # the model is actually trained on), averaged over the batch.
    metar_loss_kb = metar_loss_ps.view(K, b)
    loss_sat = sat_loss_kb[winner_idx, arange_b].mean()
    loss_metar = metar_loss_kb[winner_idx, arange_b].mean()

    components = {
        "sat_per_chan": sat_per_chan_diag.detach(),
        "metar_per_chan": metar_per_chan_diag.detach(),
        "loss_grad_sat": grad_loss_ps.view(K, b)[winner_idx, arange_b].mean().detach(),
        "loss_tgrad_sat": tgrad_loss_ps.view(K, b)[winner_idx, arange_b].mean().detach(),
        "xm_winner_rate": xm_winner_rate,
        "xm_loss_reduction": xm_loss_reduction.detach(),
    }
    return total, loss_sat, loss_metar, components
