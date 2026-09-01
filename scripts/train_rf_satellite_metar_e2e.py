"""Full end-to-end fine-tune of the FlashEdges global DiT (satellite + METAR).

Strategy
--------
Start from a checkpoint whose trunk was trained (mostly) by the satellite loss
and whose ConvGRU METAR head is already warmed up -- e.g. a
``train_rf_satellite_metar.py`` run with ``--isolate_metar_grad``. Then
fine-tune EVERYTHING jointly with ``isolate_metar_grad = False`` so the METAR
loss finally reaches the shared trunk (patch embed + DiT blocks), letting
spatial attention propagate the ~5500 station anchors across each tile -- the
one mechanism with a global receptive field that the isolated ConvGRU head
(2 layers of 3x3 convs) does not have.

Interference between the dense sat loss and the sparse metar loss is managed
with three mechanisms instead of the old hard detach:

  * **Differential learning rates**: the trunk (patch_embed, blocks, context
    MLP, sat head) fine-tunes at a low LR (``--trunk_lr``, default 1e-6, i.e.
    0.1x the from-scratch LR) so the converged satellite branch barely
    drifts, while the METAR head (``kpi_head``) trains fast
    (``--metar_head_lr``, default 1e-4). Trunk 2D params get the full trunk
    LR, trunk 1D (norms/biases) get trunk_lr/3, mirroring the base script's
    Muon-style/AdamW split.
  * **METAR weight ramp**: ``metar_loss_weight`` ramps linearly 0 -> target
    over ``--metar_warmup_steps`` so the trunk is not hit at full metar
    gradient strength while the fresh optimizer moments settle. The sat loss
    is always on at weight 1.0.
  * **Gradient-norm probing** (the diagnostic the loss-balancing research
    note calls for): every ``--grad_probe_every`` steps we run
    ``torch.autograd.grad`` from ``loss_sat`` and from ``w * loss_metar``
    down to a few reference TRUNK parameters (patch_embed, a mid block, the
    last block) and log the per-loss norms + their ratio. Balance with
    ``--metar_loss_weight`` using that ratio, not the loss magnitudes: keep
    ``grads/metar_over_sat`` roughly in the 0.3-1 band; if sat per-channel
    losses climb while metar falls, lower ``--trunk_lr`` further.

Also logs the per-batch valid-station pixel count in the target frames
(``metar/valid_target_station_px``): with ~5500 fixed stations on a 3600x1800
grid many tiles (oceans) carry zero stations and produce no metar gradient at
all -- this metric tells you the *effective* number of metar-training steps.

Checkpoints are written to ``--out_dir`` (default ``models_e2e/``) and never
touch the base checkpoint's directory. A relaunch (e.g. submit_chain.sh)
resumes from ``{out_dir}/checkpoint.safetensors`` + optimizer state +
``global_step`` so the warmup/ramp schedules stay continuous.

Usage
-----
    uv run python scripts/train_rf_satellite_metar_e2e.py \
        --base_checkpoint models/checkpoint.safetensors \
        --trunk_lr 1e-6 --metar_head_lr 1e-4 --metar_loss_weight 1.0
    # multi-GPU:
    accelerate launch scripts/train_rf_satellite_metar_e2e.py --trunk_lr 1e-6
"""

import argparse
import math
import os
import random
import sys
from datetime import datetime, timezone

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from safetensors.torch import save_file, load_file
from tqdm.auto import tqdm

# torch.compile's donated-buffer optimization reuses activation buffers inside
# the compiled backward, which forbids retain_graph=True -- and TrunkGradProbe
# needs retain_graph (two autograd.grad traversals before the real backward).
# Without this the probe dies on its first call with "backward function was
# compiled with non-empty donated buffers". Costs a bit of activation memory;
# must be set before the first compiled backward is traced.
try:
    import torch._functorch.config as _functorch_config

    _functorch_config.donated_buffer = False
except (ImportError, AttributeError):
    pass

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset_global_satellite_metar_v2 import (
    FlashEdgesGlobalDatasetV2,
    METAR_FEATURES,
)
from meteolibre_model.diffusion.rectified_flow_satellite_metar_v1 import (
    trainer_step,
    full_image_generation,
)
from meteolibre_model.models.jit3d_dual_v2 import DualJiT3D

# sat channel names mirror scripts/compute_mean_std.py
SAT_CHANNEL_NAMES = ["gmgsi_lwir", "gmgsi_vis", "gmgsi_wv", "gmgsi_sw", "elevation"]

# Substrings identifying METAR-head parameters (the modules that get the fast
# LR). Mirrors MODULES_TO_SAVE / METAR_HEAD_KEYWORDS in the PEFT script so the
# two runs are comparable: kpi_head is the current ConvGRU head; the others
# are legacy module names kept so older checkpoints/configs still group.
METAR_HEAD_KEYWORDS = ("kpi_head", "final_layer_kpi", "persist_proj", "gate_proj")


def load_config(config_name: str):
    config_path = os.path.join(project_root, "meteolibre_model", "config", "configs.yml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if config_name not in config:
        raise KeyError(f"Config '{config_name}' not found in {config_path}")
    return config[config_name]


def get_grouped_params(model):
    """Split trainable params into three fine-tune groups.

    1. ``trunk_2d``   -- trunk 2D weights (patch embed conv is 5D but lives in
       the trunk; see group 3 note) at ``--trunk_lr``.
    2. ``trunk_other``-- trunk 1D (norms/biases) at ``trunk_lr / 3``.
    3. ``metar_head`` -- every parameter of the METAR head modules, any rank,
       at ``--metar_head_lr`` (flat: conv kernels, linears and biases all
       learn fast; the head carries a sparse masked signal and is the module
       we most want to move).

    A param belongs to the METAR head if any keyword in METAR_HEAD_KEYWORDS
    appears in its dotted name; everything else is trunk.
    """
    trunk_2d = []
    trunk_other = []
    metar_head = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in METAR_HEAD_KEYWORDS):
            metar_head.append(p)
        elif p.ndim == 2:
            trunk_2d.append(p)
        else:
            trunk_other.append(p)
    return trunk_2d, trunk_other, metar_head


class CombinedOptimizer:
    """Wrapper to make a list of optimizers behave like a single one."""

    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, state in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state)


class TrunkGradProbe:
    """Per-loss gradient-norm probe on a few reference trunk parameters.

    Every ``every`` optimizer steps, computes
    ``||grad_{ref} loss_sat||`` and ``||grad_{ref} loss_metar||`` (unweighted)
    with ``torch.autograd.grad`` in ONE traversal per loss (it accepts the
    full parameter list), with ``retain_graph=True`` so the real training
    backward that immediately follows still has its graph. The effective
    ratio ``grads/metar_over_sat`` additionally scales the metar norm by the
    CURRENT weight ramp so it reflects each loss's actual pull on the trunk
    in the combined update; the raw norms stay readable during the ramp.

    Notes:
      * ``autograd.grad`` does not go through AccumulateGrad nodes, so under
        DDP it fires no reducer hooks: no allreduce, no interference with the
        training backward. Values are rank-local and only computed/logged on
        the main process (other ranks just wait at the next collective).
      * torch.compile: the probe's autograd.grad(retain_graph=True) is
        incompatible with inductor's donated-buffer reuse in the compiled
        backward, so donated_buffer is disabled at module import (see the
        comment near the imports). If some other build still rejects the
        probe, it disables itself after the first error and training
        continues -- a diagnostic must never crash a run.

    Reference params (chosen to tell the interesting story):
      * ``patch_embed`` -- the input bottleneck: the only place the raw METAR
        station dots enter the trunk. If metar grads here stay ~0 forever,
        the trunk is not learning to read stations.
      * ``block_mid``   -- mid-depth block qkv.
      * ``block_last``  -- last block attention output projection.
    """

    def __init__(self, model, every: int):
        self.every = int(every)
        self.failed = False
        jit = model.jit
        depth = len(jit.blocks)
        self.params = {
            "patch_embed": jit.patch_embed.proj.weight,
            "block_mid": jit.blocks[depth // 2].attn.qkv.weight,
            "block_last": jit.blocks[-1].attn.proj.weight,
        }

    def due(self, global_step: int) -> bool:
        return (
            self.every > 0
            and not self.failed
            and global_step % self.every == 0
        )

    def probe(self, loss_sat, loss_metar, metar_w: float):
        try:
            ps = list(self.params.values())
            g_sat = torch.autograd.grad(loss_sat, ps, retain_graph=True)
            g_met = torch.autograd.grad(loss_metar, ps, retain_graph=True)
        except Exception as e:
            self.failed = True
            print(f"[grad-probe] disabled after error: {e}", flush=True)
            return {}

        out = {}
        for (name, _), gs, gm in zip(self.params.items(), g_sat, g_met):
            out[f"grads/{name}/sat"] = gs.norm().item()
            out[f"grads/{name}/metar"] = gm.norm().item()
        # Aggregate (L2 over per-param norms) + the ratios to tune with.
        ns = math.sqrt(sum(v**2 for k, v in out.items() if k.endswith("/sat")))
        nm = math.sqrt(sum(v**2 for k, v in out.items() if k.endswith("/metar")))
        out["grads/trunk_sat_norm"] = ns
        out["grads/trunk_metar_norm"] = nm
        out["grads/metar_over_sat_raw"] = nm / max(ns, 1e-12)
        out["grads/metar_over_sat"] = (metar_w * nm) / max(ns, 1e-12)
        return out


def count_station_px(batch, context_frames: int) -> int:
    """Number of (frame, pixel) positions with >=1 valid METAR channel in the
    TARGET frames -- the pixels the metar loss actually supports this batch."""
    m = batch["metar_mask"]                      # (B, T, C, H, W) float
    per_frame_any = (m > 0).any(dim=2)           # (B, T, H, W)
    return int(per_frame_any[:, context_frames:].sum().item())


def set_lr_factor(raw_optimizers, factor: float):
    """Scale every param group's LR by ``factor`` (linear warmup).

    Operates on the RAW optimizer objects created before ``accelerate.prepare``
    -- the wrapper shares the same param_group dicts, so mutating ``lr`` here
    is seen by the wrapped step. Base LRs are stashed as ``lr0`` at creation.
    """
    for opt in raw_optimizers:
        for g in opt.param_groups:
            g["lr"] = g["lr0"] * factor


def main():
    parser = argparse.ArgumentParser(
        description="FlashEdges full e2e METAR fine-tune (differential LRs + grad probes)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="model_v6_global_satellite_metar",
        help="Config name in meteolibre_model/config/configs.yml",
    )
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default=None,
        help="Sat-trained checkpoint to start from (strict=False). If omitted, "
        "defaults to the config model_dir checkpoint.safetensors. Ignored when "
        "{out_dir}/checkpoint.safetensors exists (relaunch resume).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="models_e2e/",
        help="Directory for this run's checkpoints + optimizer state. Keep it "
        "separate from the base model_dir so the fine-tune never clobbers the "
        "base checkpoint.",
    )
    # --- the three balance knobs (see module docstring) ---
    parser.add_argument(
        "--trunk_lr",
        type=float,
        default=1e-6,
        help="LR for the shared trunk (patch embed, blocks, context MLP, sat "
        "head). 0.1x the from-scratch LR is a gentle fine-tune: the sat branch "
        "barely drifts while the trunk can still learn to read station dots.",
    )
    parser.add_argument(
        "--metar_head_lr",
        type=float,
        default=1e-4,
        help="LR for the METAR head modules (kpi_head + legacy names), any rank.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Linear LR warmup (all groups) in optimizer steps.",
    )
    parser.add_argument(
        "--metar_loss_weight",
        type=float,
        default=1.0,
        help="Target METAR branch weight (sat stays 1.0). The per-channel "
        "FastNet weights in diffusion.utils are already mean-normalized, so "
        "1.0 is a sane start. Re-tune with grads/metar_over_sat, not the loss "
        "ratio: keep the probe ratio roughly in 0.3-1.",
    )
    parser.add_argument(
        "--metar_warmup_steps",
        type=int,
        default=1000,
        help="Linear ramp 0 -> metar_loss_weight in optimizer steps, so the "
        "trunk is not hit at full metar gradient strength while the fresh "
        "optimizer moments settle.",
    )
    parser.add_argument(
        "--grad_probe_every",
        type=int,
        default=50,
        help="Log per-loss trunk gradient norms every N optimizer steps "
        "(main process only). 0 disables.",
    )
    parser.add_argument(
        "--noise_rho",
        type=float,
        default=0.0,
        help="Structured-noise correlation for the flow endpoint. NOTE: "
        "full_image_generation samples with rho=0.5 while the historical "
        "training scripts used 0.0 -- pass 0.5 here to close that "
        "train/inference mismatch.",
    )
    parser.add_argument(
        "--exit_after_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit after each epoch (matches the base script's chain-mode "
        "behaviour: submit_chain.sh relaunches and resumes from out_dir).",
    )
    # --- dataset args (mirror train_rf_satellite_metar.py) ---
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--hf_dataset_repo", type=str, default="meteolibre-dev/global_sat_metar")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--prefetch_rows", type=int, default=8)
    parser.add_argument("--shuffle_buffer", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--cache_size", type=int, default=2)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument(
        "--metar_drop_frac",
        type=float,
        default=0.05,
        help="Fraction of valid-station METAR pixels hidden in the context.",
    )
    parser.add_argument("--temporal_weight_scale", type=float, default=None)
    args = parser.parse_args()

    params = load_config(args.config)

    # --- Accelerator ---
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=4,
        log_with="tensorboard",
        project_dir=".",
        kwargs_handlers=[kwargs],
    )
    device = accelerator.device

    if torch.cuda.is_available():
        print(
            f"[device] CUDA available: {torch.cuda.device_count()} GPU(s); "
            f"this process -> {device} ({torch.cuda.get_device_name(device)})",
            flush=True,
        )
    else:
        print(f"[device] running on CPU (device={device})", flush=True)

    LOG_EVERY_N_STEPS = params["log_every_n_steps"]
    SAVE_EVERY_N_EPOCHS = params["save_every_n_epochs"]
    PARAMETRIZATION = params["parametrization"]
    INTERPOLATION = params.get("interpolation", "linear")
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    seed = params["seed"] + int(random.random() * 1000)
    residual = bool(params.get("residual", False))
    sigma_noise_input = params.get("sigma_noise_input", 0.0)
    gradient_clip_value = params["gradient_clip_value"]
    temporal_weight_scale = (
        args.temporal_weight_scale
        if args.temporal_weight_scale is not None
        else params.get("temporal_weight_scale", 1.0)
    )
    dataset_path = args.dataset_path or params["dataset_path"]
    base_checkpoint = args.base_checkpoint or os.path.join(
        params["model_dir"], "checkpoint.safetensors"
    )

    id_run = str(datetime.now(timezone.utc))[:19]
    accelerator.init_trackers(f"flashedges_e2e_metar_{id_run}")
    set_seed(seed)

    # --- Dataset: local map-style or HF streaming ---
    if args.streaming:
        from meteolibre_model.dataset.dataset_global_satellite_streaming import (
            FlashEdgesStreamingDataset,
        )
        dataset = FlashEdgesStreamingDataset(
            hf_dataset_repo=args.hf_dataset_repo,
            split="train",
            data_dir=args.data_dir,
            shuffle_buffer=args.shuffle_buffer,
            prefetch_rows=args.prefetch_rows,
            precip_to_dbz=True,
            nb_temporal=7,
            seed=seed,
        )
        streaming = True
        scope = f"data_dir={args.data_dir}" if args.data_dir else "all subfolders"
        print(
            f"  streaming: {args.hf_dataset_repo} ({scope}, "
            f"buffer={args.shuffle_buffer}, prefetch={args.prefetch_rows}, "
            f"steps/epoch={args.steps_per_epoch})"
        )
    else:
        dataset = FlashEdgesGlobalDatasetV2(
            localrepo=dataset_path,
            cache_size=args.cache_size,
            seed=seed,
            nb_temporal=7,
            precip_to_dbz=True,
        )
        streaming = False

    # persistent_workers is REQUIRED for streaming so the dataset's file
    # cursor survives across epochs (see base script).
    use_persistent = args.streaming and args.num_workers > 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )

    # --- Model ---
    assert params["model_type"] == "jit", "Only 'jit' model_type is supported"
    model = DualJiT3D(**params["model"])
    # THE point of this script: metar gradients reach the shared trunk.
    model.jit.isolate_metar_grad = False
    accelerator.print(
        "[model] isolate_metar_grad=False -- metar loss reaches the shared "
        "trunk (patch embed + DiT blocks) alongside the satellite loss"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    resume_path = os.path.join(args.out_dir, "checkpoint.safetensors")
    resume_opt_path = os.path.join(args.out_dir, "checkpoint_optimizer.pt")

    global_step = 0
    start_epoch = 0
    if os.path.exists(resume_path):
        state_dict = load_file(resume_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(
            f"[resume] loaded weights from {resume_path}\n"
            f"         missing: {len(missing)} / unexpected: {len(unexpected)}"
        )
        if os.path.exists(resume_opt_path):
            try:
                saved = torch.load(resume_opt_path, map_location=device)
                # this script saves {"optimizer": [...], "global_step": int,
                # "epoch": int}; the base script saved a bare list.
                if isinstance(saved, dict) and "optimizer" in saved:
                    optimizer_state = saved["optimizer"]
                    global_step = int(saved.get("global_step", 0))
                    start_epoch = int(saved.get("epoch", 0)) + 1
                else:
                    optimizer_state = saved
                pending_optimizer_state = optimizer_state
                accelerator.print(
                    f"[resume] optimizer state found (global_step={global_step}, "
                    f"start_epoch={start_epoch})"
                )
            except Exception as e:
                accelerator.print(f"[resume] optimizer state unreadable ({e}); fresh")
                pending_optimizer_state = None
        else:
            pending_optimizer_state = None
    else:
        if os.path.exists(base_checkpoint):
            state_dict = load_file(base_checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            accelerator.print(
                f"[load] base checkpoint: {base_checkpoint}\n"
                f"       missing keys (fresh-init): {len(missing)}\n"
                f"       unexpected keys (ignored): {len(unexpected)}"
            )
        else:
            accelerator.print(
                f"[load] WARNING: no checkpoint at {base_checkpoint}; random init."
            )
        pending_optimizer_state = None

    probe = TrunkGradProbe(model, args.grad_probe_every)

    print("start compiling")
    model = torch.compile(model)
    print("end compiling")

    # --- Optimizer: three AdamW groups with differential LRs ---
    trunk_2d, trunk_other, metar_head = get_grouped_params(model)
    raw_optimizers = []
    if trunk_2d:
        opt = torch.optim.AdamW(trunk_2d, lr=args.trunk_lr, weight_decay=0.01)
        raw_optimizers.append(opt)
    if trunk_other:
        opt = torch.optim.AdamW(
            trunk_other, lr=args.trunk_lr / 3, weight_decay=0.01
        )
        raw_optimizers.append(opt)
    if metar_head:
        opt = torch.optim.AdamW(metar_head, lr=args.metar_head_lr, weight_decay=0.01)
        raw_optimizers.append(opt)
    for opt in raw_optimizers:
        for g in opt.param_groups:
            g["lr0"] = g["lr"]  # stash base LR for the warmup factor

    if accelerator.is_main_process:
        n_t2d = sum(p.numel() for p in trunk_2d)
        n_tot = sum(p.numel() for p in trunk_other)
        n_head = sum(p.numel() for p in metar_head)
        accelerator.print(
            f"[opt] param groups -- trunk_2d: {n_t2d:,} @ lr={args.trunk_lr:.2e} | "
            f"trunk_other(1D): {n_tot:,} @ lr={args.trunk_lr/3:.2e} | "
            f"metar_head: {n_head:,} @ lr={args.metar_head_lr:.2e}"
        )

    optimizer = list(raw_optimizers)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if isinstance(optimizer, list):
        optimizer = CombinedOptimizer(optimizer)

    if pending_optimizer_state is not None:
        try:
            optimizer.load_state_dict(pending_optimizer_state)
            accelerator.print("[resume] optimizer state restored")
        except Exception as e:
            accelerator.print(f"[resume] optimizer state rejected ({e}); fresh")
    # LR warmup factor is re-applied from the raw base LRs every step, so a
    # resumed run lands on the right LR even mid-warmup.

    c_sat = params["model"]["sat_out_channels"]
    context_frames = params["model"]["context_frames"]

    # --- Training loop ---
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        total_loss_sat = 0.0
        total_loss_metar = 0.0
        n_steps_epoch = 0

        epoch_step_limit = args.steps_per_epoch if streaming else None

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=epoch_step_limit,
            disable=not accelerator.is_main_process,
        )
        for batch in progress_bar:
            # LR warmup + METAR weight ramp for this step (continuous across
            # relaunches thanks to the persisted global_step).
            lr_factor = min(1.0, global_step / max(1, args.warmup_steps))
            set_lr_factor(raw_optimizers, max(lr_factor, 1e-3))
            metar_w = args.metar_loss_weight * min(
                1.0, global_step / max(1, args.metar_warmup_steps)
            )

            with accelerator.accumulate(model):
                loss, loss_sat, loss_metar, components = trainer_step(
                    model,
                    batch,
                    device,
                    parametrization=PARAMETRIZATION,
                    interpolation=INTERPOLATION,
                    sigma=sigma_noise_input,
                    use_residual=residual,
                    metar_drop_frac=args.metar_drop_frac,
                    metar_loss_weight=metar_w,
                    temporal_weight_scale=temporal_weight_scale,
                    noise_rho=args.noise_rho,
                )

                # Per-loss trunk gradient probe BEFORE the real backward
                # (retain_graph=True keeps the graph alive for it). Main
                # process only: autograd.grad fires no DDP hooks, other ranks
                # just wait at the next collective.
                probe_metrics = {}
                if (
                    probe.due(global_step)
                    and accelerator.is_main_process
                ):
                    probe_metrics = probe.probe(loss_sat, loss_metar, metar_w)
                    for k, v in probe_metrics.items():
                        accelerator.log({k: v}, step=global_step)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), gradient_clip_value)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                n_steps_epoch += 1

                if global_step % LOG_EVERY_N_STEPS == 0 and accelerator.is_main_process:
                    accelerator.log({"Loss/train": loss.item()}, step=global_step)
                    accelerator.log({"Loss_sat/train": loss_sat.item()}, step=global_step)
                    accelerator.log({"Loss_metar/train": loss_metar.item()}, step=global_step)
                    accelerator.log({"schedule/metar_loss_weight": metar_w}, step=global_step)
                    accelerator.log(
                        {
                            "metar/valid_target_station_px": count_station_px(
                                batch, context_frames
                            )
                        },
                        step=global_step,
                    )
                    sat_pc = components["sat_per_chan"].tolist()
                    metar_pc = components["metar_per_chan"].tolist()
                    for name, v in zip(SAT_CHANNEL_NAMES, sat_pc):
                        accelerator.log({f"Loss_sat_chan/{name}": v}, step=global_step)
                    for name, v in zip(METAR_FEATURES, metar_pc):
                        accelerator.log({f"Loss_metar_chan/{name}": v}, step=global_step)

                total_loss += loss.item()
                total_loss_sat += loss_sat.item()
                total_loss_metar += loss_metar.item()
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sat=f"{loss_sat.item():.4f}",
                    metar=f"{loss_metar.item():.4f}",
                    mw=f"{metar_w:.2f}",
                )

            if epoch_step_limit is not None and n_steps_epoch >= epoch_step_limit:
                break

        avg_loss = total_loss / max(n_steps_epoch, 1)
        avg_loss_sat = total_loss_sat / max(n_steps_epoch, 1)
        avg_loss_metar = total_loss_metar / max(n_steps_epoch, 1)
        accelerator.log({"Loss/train_epoch": avg_loss}, step=epoch)
        accelerator.log({"Loss_sat/train_epoch": avg_loss_sat}, step=epoch)
        accelerator.log({"Loss_metar/train_epoch": avg_loss_metar}, step=epoch)
        accelerator.print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {avg_loss:.4f} (sat: {avg_loss_sat:.4f}, metar: {avg_loss_metar:.4f})"
        )

        # --- Visualization (main process only) ---
        # Bypass torch.compile for the once-per-epoch 128-step Euler loop via
        # _orig_mod (see the PEFT script for the inductor bug rationale).
        if accelerator.is_main_process:
            with torch.no_grad():
                unwrapped_model = accelerator.unwrap_model(model)
                gen_model = getattr(unwrapped_model, "_orig_mod", unwrapped_model)
                generated, x_target = full_image_generation(
                    gen_model,
                    batch,
                    steps=128,
                    device=accelerator.device,
                    parametrization=PARAMETRIZATION,
                    interpolation=INTERPOLATION,
                    use_residual=residual,
                    noise_rho=args.noise_rho if args.noise_rho > 0 else 0.5,
                )

                tb_tracker = accelerator.get_tracker("tensorboard")
                writer = getattr(tb_tracker, "writer", None)
                if writer is not None:
                    # GMGSI LWIR (sat channel 0)
                    gen = generated[0, 0].cpu()
                    tgt = x_target[0, 0].cpu()
                    frames = torch.cat([gen, tgt], dim=0).clamp(-10, 10)
                    grid = make_grid(frames.unsqueeze(1), nrow=2)
                    grid_normalized = make_grid(
                        (frames.unsqueeze(1) - frames.min())
                        / (frames.max() - frames.min() + 1e-8),
                        nrow=2,
                    )
                    writer.add_image("Generated vs Target (GMGSI LWIR)", grid, epoch)
                    writer.add_image(
                        "Generated vs Target (normalized)", grid_normalized, epoch
                    )

                    # METAR tmpc (first metar channel) -- the branch this run
                    # is about; normalized space, clamp to a sane range.
                    gen_t = generated[0, c_sat].cpu()
                    tgt_t = x_target[0, c_sat].cpu()
                    frames_t = torch.cat([gen_t, tgt_t], dim=0).clamp(-4, 4)
                    grid_t = make_grid(frames_t.unsqueeze(1), nrow=2)
                    writer.add_image("Generated vs Target (METAR tmpc)", grid_t, epoch)
                else:
                    print(
                        "[viz] TensorBoard writer unavailable; skipping image "
                        "logging this epoch.",
                        flush=True,
                    )

        # --- Checkpoint ---
        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                model_to_save = getattr(unwrapped_model, "_orig_mod", unwrapped_model)
                save_path = os.path.join(
                    args.out_dir, f"flashedges_e2e_epoch_{epoch + 1}.safetensors"
                )
                save_file(model_to_save.state_dict(), save_path)
                save_file(model_to_save.state_dict(), resume_path)
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    },
                    resume_opt_path,
                )
                accelerator.print(f"[save] model -> {save_path}")
                accelerator.print(f"[save] optimizer + step state -> {resume_opt_path}")

        accelerator.wait_for_everyone()

        if args.exit_after_epoch:
            accelerator.end_training()
            accelerator.print(
                f"[chain] exit after epoch {epoch + 1}; relaunch resumes at "
                f"global_step={global_step} from {args.out_dir}"
            )
            sys.exit(0)

    accelerator.end_training()
    accelerator.print("E2E METAR fine-tune complete.")


if __name__ == "__main__":
    main()
