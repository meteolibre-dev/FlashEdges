#!/bin/bash
#SBATCH --job-name=flashedges-e2e        # distinct from the base train_austral.sl logs
#SBATCH --partition=gpu_h200            # CRIANN Austral H200 nodes
                                        #   alt (A100): gpu | hpda | gpu_all | gpu_debug
#SBATCH --gpus=1                        # EXPLICIT GPU COUNT REQUIRED on Austral (use 4 for multi-GPU)
#SBATCH --cpus-per-task=24              # 24 for gpu_h200 | 16 for gpu/hpda/gpu_all | 4 for hpda_mig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:10:00
#SBATCH --output=logs/%x-%j.out         # %x=job-name  %j=jobid  (relative to submit dir)
#SBATCH --error=logs/%x-%j.err

# Full end-to-end METAR fine-tune (scripts/train_rf_satellite_metar_e2e.py):
# starts from the sat-trained base checkpoint, isolate_metar_grad=False,
# differential LRs (trunk slow / metar head fast) + metar weight ramp +
# per-loss trunk gradient-norm probing.
#
# Chainable: each job trains one epoch, exits, and the next resumes from
# OUT_DIR (weights + optimizer + global_step), so the warmup/ramp schedules
# stay continuous across relaunches:
#     SLURM_SCRIPT=slurm/train_e2e_austral.sl bash slurm/submit_chain.sh
# Balance knobs can be overridden per chain, e.g.:
#     TRUNK_LR=3e-7 METAR_LOSS_WEIGHT=0.5 \
#         SLURM_SCRIPT=slurm/train_e2e_austral.sl bash slurm/submit_chain.sh
# (tune METAR_LOSS_WEIGHT from the TensorBoard grads/metar_over_sat probe,
# target band ~0.3-1 after the ramp completes -- NOT from the loss ratio).

# ===== CRIANN Austral environment ===========================================
module purge
module load aidl/pytorch/2.6.0-cuda12.6   # python 3.13 — the ONLY module that works on H200
                                           # alt for A100: aidl/pytorch/2.5.1-cuda12.4 (py3.12)

# ===== Paths ================================================================
# /dlocal/home/<projet>  = PERSISTENT, no hard quota (Lustre). Use for code+data+checkpoints.
# /home/<projet>/<login> = only 50 Go — never train or write checkpoints here.
# /dlocal/run/$SLURM_JOB_ID = fast scratch, but AUTO-DELETED after 30 days.
PROJ=/dlocal/home/$(id -gn)               # your projet_id group  (verify: echo $HOME ; cri_quota)
CODE_DIR=/home/1997048/abufor01/FlashEdges            # repo cloned via init.sh
DATA_DIR=/home/1997048/PARTAGE/dataset-disk/flashedges                   # data you pulled with `hf download`

export HF_TOKEN                            # set once in ~/.bashrc (hf auth login)
export HF_HOME=$PROJ/hf_cache              # keep HF cache on persistent Lustre (resumable)

mkdir -p logs
cd "$CODE_DIR"                            # checkpoints land in ./models_e2e/ (persistent)

# ===== Run knobs (env-overridable, defaults match the script) ===============
BASE_CKPT="${BASE_CKPT:-models/checkpoint.safetensors}"   # sat-trained base (ignored once OUT_DIR has a checkpoint)
OUT_DIR="${OUT_DIR:-models_e2e/}"                         # this run's checkpoints; never touches models/
TRUNK_LR="${TRUNK_LR:-1e-6}"                              # shared trunk LR (0.1x from-scratch)
METAR_HEAD_LR="${METAR_HEAD_LR:-1e-4}"                    # ConvGRU metar head LR
METAR_LOSS_WEIGHT="${METAR_LOSS_WEIGHT:-1.0}"             # target branch weight (ramped 0->target over 1000 steps)

if [[ ! -f "$OUT_DIR/checkpoint.safetensors" ]]; then
    echo "[e2e] no resume checkpoint in $OUT_DIR -> starting from base: $BASE_CKPT"
    ls -lh "$BASE_CKPT" 2>/dev/null || echo "[e2e] WARNING: base checkpoint missing -- random init!"
fi

# ===== Install deps once per module =========================================
# The train script adds the repo root to sys.path itself, so no package install needed.
# Only fetch the pure-python deps the aidl module doesn't ship (don't touch torch/CUDA).
# (no peft here — the e2e fine-tune is full-parameter, no adapters)
pip install --user --no-cache-dir \
    suncalc einops pyarrow safetensors tensorboard \
    accelerate datasets torchmetrics scikit-learn pyproj imageio >/dev/null

# ===== Run ==================================================================
# Option A — stream the dataset straight from HuggingFace (default repo):
# python3 scripts/train_rf_satellite_metar_e2e.py \
#    --streaming \
#    --hf_dataset_repo meteolibre-dev/global_sat_metar \
#    --base_checkpoint "$BASE_CKPT" --out_dir "$OUT_DIR" \
#    --trunk_lr "$TRUNK_LR" --metar_head_lr "$METAR_HEAD_LR" \
#    --metar_loss_weight "$METAR_LOSS_WEIGHT"

# Option B — use the data you already downloaded locally:
python3 scripts/train_rf_satellite_metar_e2e.py \
     --dataset_path "$DATA_DIR/" \
     --base_checkpoint "$BASE_CKPT" --out_dir "$OUT_DIR" \
     --trunk_lr "$TRUNK_LR" --metar_head_lr "$METAR_HEAD_LR" \
     --metar_loss_weight "$METAR_LOSS_WEIGHT"

# Option C — multi-GPU: set --gpus=4 above, then use accelerate launch:
# accelerate launch scripts/train_rf_satellite_metar_e2e.py \
#     --streaming --hf_dataset_repo meteolibre-dev/global_sat_metar \
#     --base_checkpoint "$BASE_CKPT" --out_dir "$OUT_DIR" \
#     --trunk_lr "$TRUNK_LR" --metar_head_lr "$METAR_HEAD_LR" \
#     --metar_loss_weight "$METAR_LOSS_WEIGHT"
