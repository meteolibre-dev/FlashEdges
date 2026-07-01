"""
PEFT (LoRA) fine-tune of the FlashEdges global weather DiT to forecast METAR.

Strategy
--------
Start from a checkpoint that was trained to **denoise the satellite branch
only** (``metar_loss_weight == 0``). On top of that frozen base we:

  * apply **LoRA** (rank r) to the DiT trunk linear layers
    (attention ``qkv``/``proj`` and SwiGLU ``w1``/``w2``/``w3``) so the shared
    representation can be nudged to also carry METAR signal without
    forgetting the satellite structure;
  * **full fine-tune** the METAR-specific modules — the split metar decoder
    head (``final_layer_kpi``) and the new gated persistence path
    (``persist_proj`` + ``gate_proj``) — via PEFT's
    ``modules_to_save``;
  * turn the METAR loss **on** (``metar_loss_weight`` raised).

Everything else (patch embed, context MLP, sat head, RoPE, norms, corruptor)
stays frozen. The satellite branch therefore keeps its hard-won fine detail,
while the metar branch gets dedicated trainable capacity + a direct
same-position persistence path.

The LoRA injection / freezing / mixed-trainable-set is handled entirely by the
HuggingFace ``peft`` library (https://github.com/huggingface/peft). We only
hand it a ``LoraConfig`` and call ``get_peft_model``.

Checkpoints: we save (a) the standalone PEFT adapter directory
(``adapter_model.safetensors`` + ``adapter_config.json``) AND (b) a merged
"full" ``.safetensors`` checkpoint (base + LoRA merged back in) that is
drop-in compatible with the existing non-PEFT inference path
(``full_image_generation`` / ``DualJiT3D``).

Usage
-----
    uv run python scripts/train_rf_satellite_metar_peft.py
    accelerate launch scripts/train_rf_satellite_metar_peft.py \
        --lora_rank 16 --metar_loss_weight 1.0
"""

import argparse
import os
import sys
import random
from datetime import datetime, timezone

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from safetensors.torch import save_file, load_file
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from meteolibre_model.dataset.dataset_global_satellite_metar import (
    FlashEdgesGlobalDataset,
    METAR_FEATURES,
)
from meteolibre_model.diffusion.rectified_flow_satellite_metar_v1 import (
    trainer_step,
    full_image_generation,
)
from meteolibre_model.models.jit3d_dual_v2 import DualJiT3D

SAT_CHANNEL_NAMES = ["gmgsi_lwir", "gmgsi_vis", "gmgsi_wv", "gmgsi_sw", "elevation"]

# DiT trunk linear layers that receive LoRA adapters, expressed as a single
# regex so that ONLY the transformer block internals match (and the input
# ``patch_embed.proj`` conv does NOT). peft applies re.fullmatch against the
# full module name, e.g. ``base_model.model.jit.blocks.3.attn.qkv``. The
# satellite head (final_layer_sat), patch_embed conv, context_mlp and norms are
# not matched -> they stay frozen, preserving the satellite representation.
LORA_TARGET_MODULES = r".*blocks\.\d+\.(attn\.(qkv|proj)|mlp\.w[123])$"

# Modules fully fine-tuned alongside the adapters (peft ``modules_to_save``).
#   final_layer_kpi   : the split METAR decoder head
#   persist_proj / gate_proj : the gated previous-step METAR persistence path
# These are matched by suffix too, so ``jit.final_layer_kpi``, ``jit.persist_proj``
# and ``jit.gate_proj`` all match.
MODULES_TO_SAVE = ["final_layer_kpi", "persist_proj", "gate_proj"]


def load_config(config_name: str):
    config_path = os.path.join(project_root, "meteolibre_model", "config", "configs.yml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if config_name not in config:
        raise KeyError(f"Config '{config_name}' not found in {config_path}")
    return config[config_name]


def get_grouped_params(model):
    """Split trainable params into 2D (Muon-style) and others (AdamW).

    With PEFT only adapter + modules_to_save params have requires_grad=True;
    the frozen base is filtered out automatically.
    """
    muon_params = []
    adamw_params = []
    for _, p in model.named_parameters():
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
    parser = argparse.ArgumentParser(description="FlashEdges RF METAR PEFT fine-tune")
    parser.add_argument(
        "--config",
        type=str,
        default="model_v1_global_satellite_metar",
        help="Config name in meteolibre_model/config/configs.yml",
    )
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default=None,
        help="Path to the sat-denoised base checkpoint to load (strict=False). "
        "Defaults to the config model_dir checkpoint.safetensors.",
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank r")
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA alpha (scaling = alpha/r)"
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--metar_loss_weight",
        type=float,
        default=1.0,
        help="METAR loss weight for this fine-tune (sat loss weight stays 1.0 to "
        "keep the frozen+LoRA trunk aligned with the satellite field).",
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--hf_dataset_repo", type=str, default="meteolibre-dev/global_sat_metar")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--prefetch_rows", type=int, default=8)
    parser.add_argument("--shuffle_buffer", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--metar_drop_frac", type=float, default=0.05)
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
    base_checkpoint = args.base_checkpoint or os.path.join(MODEL_DIR, "checkpoint.safetensors")

    id_run = str(datetime.now(timezone.utc))[:19]
    accelerator.init_trackers(f"flashedges_metar_peft_{id_run}")

    set_seed(seed)

    # --- Dataset ---
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
    else:
        dataset = FlashEdgesGlobalDataset(
            localrepo=dataset_path,
            cache_size=10,
            seed=seed,
            nb_temporal=7,
            precip_to_dbz=True,
        )
        streaming = False

    use_persistent = args.streaming and args.num_workers > 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )

    # --- Model: build, load sat-only base, wrap with PEFT ---
    assert params["model_type"] == "jit", "Only 'jit' model_type is supported"
    # The config's model dict already carries kpi_in_channels, so the metar
    # ref encoder + dual head are constructed automatically.
    model = DualJiT3D(**params["model"])

    if os.path.exists(base_checkpoint):
        state_dict = load_file(base_checkpoint)
        # strict=False: the base may predate the dual-head / persistence-path
        # additions; those modules keep their fresh init.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(
            f"[load] base checkpoint: {base_checkpoint}\n"
            f"        missing keys (fresh-init): {len(missing)}\n"
            f"        unexpected keys (ignored): {len(unexpected)}"
        )
    else:
        accelerator.print(
            f"[load] WARNING: base checkpoint not found at {base_checkpoint}; "
            "starting from random init."
        )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,  # regex string; see note below
        modules_to_save=MODULES_TO_SAVE,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Sanity: report trainable params per group.
    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        accelerator.print(
            f"[peft] trainable: {trainable:,} / {total:,} "
            f"({100.0 * trainable / total:.3f}%)  "
            f"LoRA r={args.lora_rank} alpha={args.lora_alpha} "
            f"targets={LORA_TARGET_MODULES}  save={MODULES_TO_SAVE}"
        )

    # torch.compile AFTER peft wrapping so the adapter layers are compiled too.
    model = torch.compile(model)

    # --- Optimizer (only trainable params are collected) ---
    muon_params, adamw_params = get_grouped_params(model)
    opt_muon = torch.optim.AdamW(muon_params, lr=learning_rate, weight_decay=0.01)
    opt_adam = torch.optim.AdamW(adamw_params, lr=learning_rate / 3, weight_decay=0.01)
    optimizer = [opt_muon, opt_adam]

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if isinstance(optimizer, list):
        optimizer = CombinedOptimizer(optimizer)

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_steps_epoch = 0
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
                    metar_loss_weight=args.metar_loss_weight,
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
                    sat_pc = components["sat_per_chan"].tolist()
                    metar_pc = components["metar_per_chan"].tolist()
                    for name, v in zip(SAT_CHANNEL_NAMES, sat_pc):
                        accelerator.log({f"Loss_sat_chan/{name}": v}, step=global_step)
                    for name, v in zip(METAR_FEATURES, metar_pc):
                        accelerator.log({f"Loss_metar_chan/{name}": v}, step=global_step)

                total_loss += loss.item()
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sat=f"{loss_sat.item():.4f}",
                    metar=f"{loss_metar.item():.4f}",
                )

            if epoch_step_limit is not None and n_steps_epoch >= epoch_step_limit:
                break

        avg_loss = total_loss / max(n_steps_epoch, 1)
        accelerator.log({"Loss/train_epoch": avg_loss}, step=epoch)
        accelerator.print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

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
                generated_sample = generated[0, 0]
                target_sample = x_target[0, 0].cpu()
                all_frames = torch.cat([generated_sample, target_sample], dim=0).clamp(-10, 10)
                grid = make_grid(all_frames.unsqueeze(1), nrow=2)
                grid_normalized = make_grid(
                    (all_frames.unsqueeze(1) - all_frames.min())
                    / (all_frames.max() - all_frames.min() + 1e-8),
                    nrow=2,
                )
                tb_tracker = accelerator.get_tracker("tensorboard")
                writer = getattr(tb_tracker, "writer", None)
                if writer is not None:
                    writer.add_image("Generated vs Target (GMGSI LWIR)", grid, epoch)
                    writer.add_image("Generated vs Target (normalized)", grid_normalized, epoch)

        # --- Checkpoint ---
        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_peft_checkpoint(accelerator.unwrap_model(model), MODEL_DIR, epoch)
                accelerator.print(f"[save] PEFT checkpoint saved under {MODEL_DIR} (epoch {epoch})")

        accelerator.wait_for_everyone()

    accelerator.end_training()
    accelerator.print("PEFT METAR fine-tune complete.")


def save_peft_checkpoint(accelerator_unwrapped_model, model_dir, epoch):
    """Save both the standalone adapter and a merged full checkpoint.

    1. ``<model_dir>/peft/`` : adapter_model.safetensors + adapter_config.json
       (small; only LoRA + modules_to_save weights). Reloadable via
       ``PeftModel.from_pretrained`` for further fine-tuning.
    2. ``<model_dir>/flashedges_peft_epoch_<n>.safetensors`` and
       ``<model_dir>/checkpoint.safetensors`` : the base with LoRA **merged**
       back in, as a plain ``DualJiT3D`` state_dict — drop-in for the existing
       non-PEFT inference / ``full_image_generation`` path.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Unwrap torch.compile and DistributedDataParallel to reach the PeftModel.
    peft_model = accelerator_unwrapped_model
    peft_model = getattr(peft_model, "_orig_mod", peft_model)

    # (1) standalone adapter
    adapter_dir = os.path.join(model_dir, "peft")
    os.makedirs(adapter_dir, exist_ok=True)
    peft_model.save_pretrained(adapter_dir)

    # (2) merged full checkpoint
    try:
        merged = peft_model.merge_and_unload()
        # ``merge_and_unload`` returns the underlying base nn.Module with LoRA
        # folded into the weights. Its state_dict keys match DualJiT3D exactly.
        base_state = merged.state_dict()
        save_path = os.path.join(model_dir, f"flashedges_peft_epoch_{epoch + 1}.safetensors")
        save_path_checkpoint = os.path.join(model_dir, "checkpoint.safetensors")
        save_file(base_state, save_path)
        save_file(base_state, save_path_checkpoint)
    except Exception as e:
        # merge_and_unload can fail with certain torch.compile graphs / custom
        # modules. Fall back to a CPU-side deepcopy merge so training never
        # blocks on a save.
        import copy
        print(f"[save] merge_and_unload failed ({e}); falling back to deepcopy merge.")
        cpu_model = copy.deepcopy(peft_model).to("cpu")
        merged = cpu_model.merge_and_unload()
        base_state = merged.state_dict()
        save_path = os.path.join(model_dir, f"flashedges_peft_epoch_{epoch + 1}.safetensors")
        save_path_checkpoint = os.path.join(model_dir, "checkpoint.safetensors")
        save_file(base_state, save_path)
        save_file(base_state, save_path_checkpoint)


if __name__ == "__main__":
    main()
