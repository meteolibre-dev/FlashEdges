"""Convolutional GRU head for the KPI / METAR branch.

Replaces the previous gated-persistence skip path in ``JiT3D_Modern`` with a
small spatiotemporal recurrent decoder that:

  - Unpatchifies the shared trunk tokens into a per-frame trunk estimate of
    the KPI field at full resolution (same role as the old ``final_layer_kpi``).
  - Concatenates the raw METAR (``metar_ref``) channel-by-channel so the
    recurrence sees the sparse station anchors directly (bypassing the trunk's
    patchify + attention dilution).
  - Unrolls a ConvGRU over the temporal axis. The conv gates give each step a
    spatial receptive field, so station values propagate into their
    neighbourhood guided by the local trunk texture -- this is the spatial
    propagation the old 1x1 persistence path could not do. The recurrent hidden
    state carries temporal continuity, which (when the state is persisted
    across AR batches at inference + trained with short rollouts) directly
    attacks the autoregressive batch-boundary discontinuity.
  - Injects the per-sample scalar context (sun position + lat + diffusion
    timestep, the same (B, context_dim) tensor the trunk already uses) as an
    additive bias on the hidden state at every frame, giving the head a
    time-of-day / latitude signal to drive the diurnal curve even when the
    raw-METAR context is all zeros at non-station pixels.

The module exposes an optional ``init_states`` input and returns the final
``states`` so callers can carry the hidden state across autoregressive batches
(or across denoising steps). For now the inference engine does not persist
state, so the head degrades gracefully to a within-batch refinement (still
useful: spatial propagation + diurnal scalar). Cross-batch carry + rollout
training is the follow-on change that unlocks the full benefit.

References:
  ConvGRU dynamics follow Ballas et al., "Delving Deeper into Convolutional
  Networks for Learning Video Representations" (ICLR 2016) and the
  jacobkimmel/pytorch_convgru implementation. The cell uses the standard
  3-gate GRU (reset r, update z, candidate c) with the candidate conditioned
  on the reset-gated hidden state.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from meteolibre_model.models.jit3d import FinalLayer


class ConvGRUCell(nn.Module):
    """Single convolutional GRU cell (3 gates, spatial conv).

    Input  : x (B, input_dim, H, W), h (B, hidden_dim, H, W)
    Output : h_next (B, hidden_dim, H, W)

    Dynamics (Ballas et al. 2016):
        r = sigmoid(W_rz [x, h])      (reset gate)
        z = sigmoid(W_rz [x, h])      (update gate)
        c = tanh   (W_c  [x, r * h])  (candidate, gated by reset)
        h_next = (1 - z) * h + z * c
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        # reset + update gates share the [x, h] concat (one conv, 2*hidden out)
        self.conv_rz = nn.Conv2d(
            input_dim + hidden_dim, 2 * hidden_dim,
            kernel_size, padding=pad, bias=bias,
        )
        # candidate sees [x, r*h] (also a concat of input_dim + hidden_dim)
        self.conv_c = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size, padding=pad, bias=bias,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        rz = self.conv_rz(torch.cat([x, h], dim=1))
        r, z = torch.split(rz, self.hidden_dim, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        c = torch.tanh(self.conv_c(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * c


class ConvGRUHead(nn.Module):
    """ConvGRU refinement head for the KPI / METAR branch.

    Replaces ``final_layer_kpi`` + the gated-persistence path. Operates at
    full spatial resolution so sparse station dots in ``raw_kpi`` survive
    (avg-pooling to patch resolution would erase isolated stations).

    Args:
        patch_size: trunk patch size (pt, ph, pw) -- used by the internal
            FinalLayer to unpatchify trunk tokens.
        kpi_out_channels: number of output KPI channels (7 for METAR).
        kpi_in_channels: number of raw METAR channels fed back to the head
            (the ``metar_ref`` tensor). Set 0 / None to disable the raw path
            (head then refines the trunk estimate alone).
        embed_dim: trunk token dimension D.
        hidden_dim: ConvGRU hidden width. Cost scales with H*W*hidden_dim;
            32-64 is a good range for 128x128 patches.
        scalar_dim: dimensionality of the per-sample scalar context (sun +
            lat + diffusion timestep) injected as a hidden-state bias. Matches
            the trunk's ``context_dim`` (6 for FlashEdges: spatial 4 + d 1 + t 1).
        kernel_size: conv kernel in the ConvGRU cells (3 is a good default;
            larger kernels propagate stations further per step).
        num_layers: stacked ConvGRU layers (1 is usually enough for a
            refinement head; 2 adds capacity at ~2x cost).
    """

    def __init__(
        self,
        patch_size,
        kpi_out_channels: int,
        kpi_in_channels: int,
        embed_dim: int,
        hidden_dim: int = 64,
        scalar_dim: int = 6,
        kernel_size: int = 3,
        num_layers: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.kpi_out_channels = kpi_out_channels
        self.kpi_in_channels = kpi_in_channels or 0
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Trunk-token unpatchifier -> per-frame trunk KPI estimate at full res.
        # Same role as the old final_layer_kpi: maps (B, N, D) -> (B, C_out, T, H, W).
        self.final_layer = FinalLayer(patch_size, kpi_out_channels, embed_dim)

        # Scalar context (sun position + lat + diffusion t) -> hidden-dim bias,
        # injected additively on the hidden state at every frame so the
        # recurrence has a time-of-day / latitude signal to drive the diurnal
        # curve even when raw METAR context is all zeros at non-station pixels.
        if scalar_dim and scalar_dim > 0:
            self.scalar_mlp = nn.Sequential(
                nn.Linear(scalar_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.scalar_mlp = None

        # Per-frame input channels: trunk estimate (+ raw METAR if present).
        in_dim = kpi_out_channels + self.kpi_in_channels
        cells = []
        for i in range(num_layers):
            cells.append(
                ConvGRUCell(
                    input_dim=in_dim if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                )
            )
        self.cells = nn.ModuleList(cells)

        # hidden -> KPI output channels (1x1 conv; per-pixel linear over ch)
        self.out_proj = nn.Conv2d(hidden_dim, kpi_out_channels, kernel_size=1)

    def forward(
        self,
        trunk_tokens: torch.Tensor,
        T: int,
        H: int,
        W: int,
        raw_kpi: torch.Tensor | None = None,
        scalar: torch.Tensor | None = None,
        init_states: list | None = None,
    ):
        """Run the ConvGRU refinement head.

        Args:
            trunk_tokens: (B, N, D) shared trunk output (possibly detached
                when ``isolate_metar_grad`` is on).
            T, H, W: full-frame temporal + spatial dims (used by FinalLayer).
            raw_kpi: (B, kpi_in_channels, T, H, W) raw METAR (``metar_ref``).
                May be None -> the head refines the trunk estimate alone.
            scalar: (B, scalar_dim) per-sample scalar context (the same tensor
                the trunk uses for its global token bias). Broadcast over T.
            init_states: optional list of (h, ) initial hidden states per
                layer, for cross-batch / cross-step carry. None -> zero init.

        Returns:
            kpi_out: (B, kpi_out_channels, T, H, W) refined KPI forecast.
            states: list of final hidden states per layer, to carry forward.
        """
        # 1. Trunk estimate at full resolution (B, C_out, T, H, W)
        trunk_kpi = self.final_layer(trunk_tokens, T, H, W)
        B, C_out, T_, H_, W_ = trunk_kpi.shape

        # 2. Per-frame input: trunk estimate (+ raw METAR channels)
        if raw_kpi is not None and self.kpi_in_channels > 0:
            # raw_kpi is (B, C_in, T, H, W) at full res; concat along channels.
            inp = torch.cat([trunk_kpi, raw_kpi], dim=1)
        else:
            inp = trunk_kpi

        # 3. Scalar bias (B, hidden_dim) -> broadcast spatially when applied
        s_emb = None
        if self.scalar_mlp is not None and scalar is not None:
            s_emb = self.scalar_mlp(scalar)  # (B, hidden_dim)

        # 4. Init hidden states (zero if not provided)
        device = trunk_kpi.device
        states = []
        for i in range(self.num_layers):
            if init_states is not None and i < len(init_states) and init_states[i] is not None:
                states.append(init_states[i])
            else:
                states.append(torch.zeros(B, self.hidden_dim, H_, W_, device=device))

        # 5. Unroll over T (context frames first, then forecast -- the hidden
        #    state builds up from the real sparse context and carries into the
        #    forecast frames).
        outs = []
        for t in range(T_):
            x = inp[:, :, t]  # (B, Cin, H, W)
            for i, cell in enumerate(self.cells):
                h = states[i]
                # inject scalar as an additive bias on the hidden state for
                # this frame (broadcast over H, W). Applied fresh every frame
                # so the recurrence is continuously conditioned on time-of-day.
                if s_emb is not None:
                    h = h + s_emb.view(B, self.hidden_dim, 1, 1)
                h = cell(x, h)
                states[i] = h
                x = h  # feed hidden to the next layer
            outs.append(self.out_proj(states[-1]))  # (B, C_out, H, W)

        kpi_out = torch.stack(outs, dim=2)  # (B, C_out, T, H, W)
        return kpi_out, states


if __name__ == "__main__":
    # Smoke test on CPU.
    B, C_out, C_in, T, H, W = 2, 7, 7, 7, 128, 128
    patch_size = (1, 8, 8)
    embed_dim = 384
    tokens = T * (H // 8) * (W // 8)
    trunk_tokens = torch.randn(B, tokens, embed_dim)
    raw_kpi = torch.randn(B, C_in, T, H, W)
    scalar = torch.randn(B, 6)

    head = ConvGRUHead(
        patch_size, C_out, C_in, embed_dim,
        hidden_dim=32, scalar_dim=6, kernel_size=3, num_layers=1,
    )
    out, states = head(trunk_tokens, T, H, W, raw_kpi=raw_kpi, scalar=scalar)
    print("kpi_out:", out.shape)  # (2, 7, 7, 128, 128)
    print("states:", [s.shape for s in states])  # [(2, 32, 128, 128)]

    # cross-step carry: feed previous states back in
    out2, states2 = head(trunk_tokens, T, H, W, raw_kpi=raw_kpi, scalar=scalar, init_states=states)
    print("carry ok:", out2.shape)
    loss = out.sum() + out2.sum()
    loss.backward()
    print("backward ok")
