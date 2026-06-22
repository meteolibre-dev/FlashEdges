"""
Training script for FlashEdges using Hugging Face Accelerate with Rectified Flow
(shortcut version, x-prediction).

Trains a DualJiT3D (DiT) on the global GMGSI + METAR parquet dataset.  The
satellite branch (5ch: GMGSI + elevation) and METAR branch (7ch, p01m in dBZ)
are forecast jointly.  METAR loss is masked by the valid-station mask because
station observations are extremely sparse.

Usage:
    uv run python scripts/train_rf_satellite_metar.py
    # multi-GPU:
    accelerate launch scripts/train_rf_satellite_metar.py
"""

import argparse
import os
import sys
import math
import random
from datetime import datetime, timezone

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from safetensors.torch import save_file
from safetensors.torch import load_file
from tqdm.auto import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset_global_satellite_metar import FlashEdgesGlobalDataset
from meteolibre_model.dataset.dataset_global_satellite_metar import METAR_FEATURES
from meteolibre_model.diffusion.rectified_flow_satellite_metar_v1 import (
    trainer_step,
    full_image_generation,
)
from meteolibre_model.models.jit3d_dual_v2 import DualJiT3D

# sat channel names mirror scripts/compute_mean_std.py
SAT_CHANNEL_NAMES = ["gmgsi_lwir", "gmgsi_vis", "gmgsi_wv", "gmgsi_sw", "elevation"]


def load_config(config_name: str):
    config_path = os.path.join(project_root, "meteolibre_model", "config", "configs.yml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if config_name not in config:
        raise KeyError(f"Config '{config_name}' not found in {config_path}")
    return config[config_name]


def get_grouped_params(model):
    """Split params into 2D (Muon) and others (AdamW)."""
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


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


def main():
    parser = argparse.ArgumentParser(description="FlashEdges RF training")
    parser.add_argument(
        "--config",
        type=str,
        default="model_v1_global_satellite_metar",
        help="Config name in meteolibre_model/config/configs.yml",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override dataset_path from config (local map-style dataset).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset from Hugging Face instead of reading locally.",
    )
    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        default="meteolibre-dev/global_sat_metar",
        help="HF dataset repo id for streaming (e.g. meteolibre-dev/flashedges_global_v1).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Hub mode: restrict to one subfolder (e.g. data_2022_02). None = all.",
    )
    parser.add_argument(
        "--prefetch_rows",
        type=int,
        default=8,
        help="Streaming: rows to prefetch in a background thread (overlaps I/O "
        "with compute). 0 disables.",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=200,
        help="Streaming: rows held in the shuffle buffer for decorrelation "
        "(~4 MB/row; larger => better shuffle but more RAM). Set to 1 to "
        "disable shuffling.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="DataLoader num_workers (file-level sharding lets each worker "
        "stream its own parquet files in parallel). Default 4.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=4000,
        help="Steps per epoch when streaming (no length available). Required for streaming.",
    )
    parser.add_argument(
        "--metar_drop_frac",
        type=float,
        default=0.05,
        help="Fraction of valid-station METAR pixels to randomly hide in the "
        "conditioning context each step (self-supervised spatial fill): the "
        "model must reconstruct them in the forecast from satellite + the "
        "remaining stations, so it learns to output a full METAR image even "
        "where no station reported. 0 disables. Default 0.05.",
    )
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

    # --- Hyperparameters ---
    LOG_EVERY_N_STEPS = params["log_every_n_steps"]
    SAVE_EVERY_N_EPOCHS = params["save_every_n_epochs"]
    MODEL_DIR = params["model_dir"]
    PARAMETRIZATION = params["parametrization"]
    INTERPOLATION = params.get("interpolation", "linear")
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    num_epochs = params["num_epochs"]
    seed = params["seed"] + int(random.random() * 1000)
    residual = bool(params.get("residual", False))
    sigma_noise_input = params.get("sigma_noise_input", 0.0)
    gradient_clip_value = params["gradient_clip_value"]
    dataset_path = args.dataset_path or params["dataset_path"]

    id_run = str(datetime.now(timezone.utc))[:19]

    # Initialize the trackers so accelerator.log(...) and get_tracker(...)
    # actually work. Without this call, self.trackers stays empty: scalar logs
    # are silently dropped and get_tracker("tensorboard") returns a blank
    # GeneralTracker(_blank=True) with no `.writer`, which crashed the
    # end-of-epoch image logging with AttributeError.
    accelerator.init_trackers(
        f"flashedges_global_satellite_metar_{id_run}",
    )

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
        print(f"  streaming: {args.hf_dataset_repo} ({scope}, "
              f"buffer={args.shuffle_buffer}, prefetch={args.prefetch_rows}, "
              f"steps/epoch={args.steps_per_epoch})")
    else:
        dataset = FlashEdgesGlobalDataset(
            localrepo=dataset_path,
            cache_size=10,
            seed=seed,
            nb_temporal=7,
            precip_to_dbz=True,
        )
        streaming = False

    # persistent_workers is REQUIRED for streaming so the dataset's file
    # cursor survives across epochs -- otherwise each epoch restarts at file 0
    # and (with a small steps_per_epoch) re-reads the same leading files every
    # epoch, never reaching the rest of the shard. Only meaningful with >0
    # workers; a 0-worker DataLoader always reuses the same (main) instance.
    use_persistent = args.streaming and args.num_workers > 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # streaming shuffles via buffer; map-style relies on
                        # per-worker file shuffling in __getitem__
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )


    # --- Model ---
    model_params = params["model"]
    assert params["model_type"] == "jit", "Only 'jit' model_type is supported"
    model = DualJiT3D(**model_params)

    # Learnable uncertainty weights for adaptive sat/metar loss balancing
    # (Kendall, Gal & Cipolla, CVPR 2018). log_vars = log(sigma^2) per branch;
    # the effective weight on branch i is exp(-log_vars[i]). Registered as a
    # model Parameter so it is (a) picked up by get_grouped_params / the
    # optimizer, (b) kept in sync across DDP processes by accelerator.prepare,
    # and (c) saved/loaded with the checkpoint.
    #
    # Init sat weight = 1.0, metar weight = 0.3 -- the manual rebalance we'd
    # have picked by hand to stop METAR's high-variance loss from starving the
    # satellite branch. Uncertainty weighting then adapts these from this
    # warm-start. log_vars[i] = -log(weight_i), so sat -> 0.0, metar -> 1.204.
    model.log_vars = nn.Parameter(
        torch.tensor([0.0, -math.log(0.3)])
    )  # [sat, metar] -> exp(-s) = [1.0, 0.3]

    model_path = "models/checkpoint.safetensors"
    state_dict = load_file(model_path)
    # strict=False: load only the keys present in the checkpoint; new params
    # (log_vars, the split decoder heads) keep their initialization. Starting
    # a full retrain, so the split decoder heads are initialized fresh.
    model.load_state_dict(state_dict, strict=False)

    model = torch.compile(model)

    # --- Optimizer: Muon (2D) + AdamW (rest) ---
    muon_params, adamw_params = get_grouped_params(model)
    # opt_muon = Muon(muon_params, lr=learning_rate, momentum=0.95, weight_decay=0.1)
    opt_muon = torch.optim.AdamW(
        muon_params, lr=learning_rate, weight_decay=0.01
    )
    opt_adam = torch.optim.AdamW(
        adamw_params, lr=learning_rate / 3, weight_decay=0.01
    )
    optimizer = [opt_muon, opt_adam]

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if isinstance(optimizer, list):
        optimizer = CombinedOptimizer(optimizer)

    global_step = 0

    # --- Training loop ---
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_steps_epoch = 0

        # For streaming datasets there is no length; cap the epoch at
        # steps_per_epoch. For map-style, len(dataloader) bounds it.
        epoch_step_limit = args.steps_per_epoch if streaming else None

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=epoch_step_limit,
            disable=not accelerator.is_main_process,
        )
        for batch in progress_bar:
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
                )

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
                    # Per-channel masked-MSE (FastNet-style diagnostics). These
                    # reveal which channels dominate each branch's loss -- the
                    # key signal for deciding whether the sat/metar imbalance is
                    # structural (e.g. dBZ precip is always hard) or a weight bug.
                    sat_pc = components["sat_per_chan"].tolist()
                    metar_pc = components["metar_per_chan"].tolist()
                    for name, v in zip(SAT_CHANNEL_NAMES, sat_pc):
                        accelerator.log(
                            {f"Loss_sat_chan/{name}": v}, step=global_step
                        )
                    for name, v in zip(METAR_FEATURES, metar_pc):
                        accelerator.log(
                            {f"Loss_metar_chan/{name}": v}, step=global_step
                        )
                    # Effective learned branch weights (uncertainty weighting).
                    # Tracks how the model reallocates gradient between sat and
                    # metar: expect weight_metar to drift down as it absorbs the
                    # METAR noise floor, and weight_sat to stay O(1).
                    accelerator.log(
                        {
                            "LossWeight/sat": components["loss_weight_sat"].item(),
                            "LossWeight/metar": components["loss_weight_metar"].item(),
                        },
                        step=global_step,
                    )

                total_loss += loss.item()
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sat=f"{loss_sat.item():.4f}",
                    metar=f"{loss_metar.item():.4f}",
                    w_metar=f"{components['loss_weight_metar'].item():.2f}",
                )

            if epoch_step_limit is not None and n_steps_epoch >= epoch_step_limit:
                break

        avg_loss = total_loss / max(n_steps_epoch, 1)
        accelerator.log({"Loss/train_epoch": avg_loss}, step=epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # --- Visualization (main process only) ---
        if accelerator.is_main_process:
            with torch.no_grad():
                unwrapped_model = accelerator.unwrap_model(model)
                generated, x_target = full_image_generation(
                    unwrapped_model,
                    batch,
                    steps=128,
                    device=accelerator.device,
                    parametrization=PARAMETRIZATION,
                    interpolation=INTERPOLATION,
                    use_residual=residual,
                )

                # Visualize GMGSI LWIR channel (channel 0 of sat branch)
                generated_sample = generated[0, 0]
                target_sample = x_target[0, 0].cpu()

                all_frames = torch.cat([generated_sample, target_sample], dim=0)
                all_frames = all_frames.clamp(-10, 10)

                grid = make_grid(all_frames.unsqueeze(1), nrow=2)
                grid_normalized = make_grid(
                    (all_frames.unsqueeze(1) - all_frames.min())
                    / (all_frames.max() - all_frames.min() + 1e-8),
                    nrow=2,
                )

                tb_tracker = accelerator.get_tracker("tensorboard")
                # Be defensive: a missing/blank tracker (or a version without
                # .writer) must never crash a full epoch of training. Use the
                # tracker's SummaryWriter if present, else skip the images.
                writer = getattr(tb_tracker, "writer", None)
                if writer is not None:
                    writer.add_image(
                        "Generated vs Target (GMGSI LWIR)", grid, epoch
                    )
                    writer.add_image(
                        "Generated vs Target (normalized)", grid_normalized, epoch
                    )
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
                os.makedirs(MODEL_DIR, exist_ok=True)
                save_path = f"{MODEL_DIR}flashedges_v1_epoch_{epoch + 1}.safetensors"
                save_path_checkpoint = f"{MODEL_DIR}checkpoint.safetensors"

                model_to_save = getattr(
                    unwrapped_model, "_orig_mod", unwrapped_model
                )
                save_file(model_to_save.state_dict(), save_path)
                save_file(model_to_save.state_dict(), save_path_checkpoint)
                accelerator.print(f"Model saved to {save_path}")

        accelerator.wait_for_everyone()

    accelerator.end_training()
    print("Training complete.")


# Muon optimizer: import here so the script works on PyTorch builds that ship it.
try:
    from torch.optim import Muon
except ImportError:
    Muon = None


if __name__ == "__main__":
    if Muon is None:
        # Fallback: if this PyTorch build lacks Muon, monkeypatch a thin shim.
        import warnings

        warnings.warn(
            "torch.optim.Muon not available; falling back to AdamW for all params. "
            "Install a Muon-enabled PyTorch or use the heavyball package for best results."
        )

        class Muon(torch.optim.Optimizer):  # type: ignore[no-redef]
            def __init__(self, params, lr=0.001, momentum=0.95, weight_decay=0.0):
                super().__init__(params, {"lr": lr, "momentum": momentum, "weight_decay": weight_decay})
                self._inner = None

            @torch.no_grad()
            def step(self, closure=None):
                for group in self.param_groups:
                    if self._inner is None:
                        self._inner = torch.optim.AdamW(
                            group["params"],
                            lr=group["lr"],
                            weight_decay=group["weight_decay"],
                        )
                    self._inner.step()

    main()
