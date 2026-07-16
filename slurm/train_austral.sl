#!/bin/bash
#SBATCH --job-name=flashedges
#SBATCH --partition=gpu_h200            # CRIANN Austral H200 nodes
                                        #   alt (A100): gpu | hpda | gpu_all | gpu_debug
#SBATCH --gpus=1                        # EXPLICIT GPU COUNT REQUIRED on Austral (use 4 for multi-GPU)
#SBATCH --cpus-per-task=24              # 24 for gpu_h200 | 16 for gpu/hpda/gpu_all | 4 for hpda_mig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out         # %x=job-name  %j=jobid  (relative to submit dir)
#SBATCH --error=logs/%x-%j.err

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
cd "$CODE_DIR"                            # checkpoints land in ./models/ (persistent)

# ===== Install deps once per module =========================================
# The train script adds the repo root to sys.path itself, so no package install needed.
# Only fetch the pure-python deps the aidl module doesn't ship (don't touch torch/CUDA).
pip install --user --no-cache-dir \
    suncalc einops pyarrow safetensors tensorboard \
    accelerate datasets peft torchmetrics scikit-learn pyproj imageio >/dev/null

# ===== Run ==================================================================
# Option A — stream the dataset straight from HuggingFace (default repo):
# python3 scripts/train_rf_satellite_metar.py \
#    --streaming \
#    --hf_dataset_repo meteolibre-dev/global_sat_metar

# Option B — use the data you already downloaded locally:
python3 scripts/train_rf_satellite_metar.py \
     --dataset_path "$DATA_DIR/"

# Option C — multi-GPU: set --gpus=4 above, then use accelerate launch:
# accelerate launch scripts/train_rf_satellite_metar.py \
#     --streaming --hf_dataset_repo meteolibre-dev/global_sat_metar

