#!/bin/bash
# ============================================================
# SLURM Job: FNO operator-learning pipeline — ampere/GPU route.
# Identical pipeline to scripts/slurm_fno.sh; runs on a single
# A100 GPU. The training scripts auto-detect CUDA via cfg
# device=auto, so no code changes are required.
# ============================================================
#SBATCH --job-name=qnm_fno_gpu
#SBATCH --output=qnm_fno_%j.out
#SBATCH --error=qnm_fno_%j.err
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

export PYTHONUNBUFFERED=1
set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG=${1:-configs/fno_zerilli_l2.yaml}

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Account:       $SLURM_JOB_ACCOUNT"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Started:       $(date)"
echo "Config:        $CONFIG"
echo "============================================"

cd "$WORKDIR"

# --- Storage guard: outputs/ MUST be a symlink to RDS (/home is 52 GB quota).
# Truncated v5-C checkpoint on 2026-05-17 was caused by /home quota overflow.
# See /memories/repo/csd3_storage.md.
if [ ! -L outputs ]; then
    echo "[FATAL] outputs/ is not a symlink. Refusing to write to /home (quota=52GB)."
    echo "        Run: mv outputs /rds/user/\$USER/hpc-work/<dest> && ln -s /rds/user/\$USER/hpc-work/<dest> outputs"
    exit 3
fi
OUTPUTS_TARGET=$(readlink -f outputs)
echo "[STORAGE] outputs -> $OUTPUTS_TARGET"
case "$OUTPUTS_TARGET" in
    /rds/*) ;;
    *) echo "[FATAL] outputs symlink does NOT point into /rds/ (got: $OUTPUTS_TARGET)."; exit 3 ;;
esac

# --- Environment setup ---
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

# Self-bootstrap venv_gpu on the compute node if missing.
# torch is pinned to a driver-compatible cu118 wheel so we never
# repeat the cu130 vs driver-12.0.80 mismatch that broke the last run.
VENV=venv_gpu
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[SETUP] Creating $VENV on compute node ..."
    python3 -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export TMPDIR="$WORKDIR/.pip_tmp"
    mkdir -p "$TMPDIR"
    python -m pip install --quiet --upgrade pip wheel setuptools
    echo "[SETUP] Installing torch 2.5.1+cu118 (driver-compatible) ..."
    python -m pip install --quiet \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu118
    echo "[SETUP] Installing project + neuraloperator ..."
    python -m pip install --quiet .
    python -m pip install --quiet neuraloperator
    rm -rf "$TMPDIR"; unset TMPDIR
else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
echo "[SETUP] Python: $(python --version) at $(which python)"

# --- HARD CUDA assertion: exit before any training if GPU broken ---
python - <<'PYEOF'
import sys, torch
print(f"[CUDA] torch={torch.__version__} cuda_build={torch.version.cuda}")
if not torch.cuda.is_available():
    print("[FATAL] torch.cuda.is_available() = False — aborting before wasting GPU hours.")
    sys.exit(2)
print(f"[CUDA] device 0: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
# Touch the GPU with a real op so a broken driver/runtime fails NOW.
x = torch.randn(2048, 2048, device='cuda')
torch.cuda.synchronize()
y = (x @ x).sum().item()
print(f"[CUDA] matmul ok, sum={y:.3e}")
PYEOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- 1) Generate dataset (skip if already present) ---
DATA_PATH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['data_path'])")
if [ ! -f "$DATA_PATH" ]; then
    echo "[FNO] Generating dataset -> $DATA_PATH"
    python scripts/generate_fno_dataset.py --config "$CONFIG"
else
    echo "[FNO] Dataset already exists at $DATA_PATH — skipping generation"
fi

# --- 2) Train FNO (resume-safe) ---
echo ""
echo "[FNO] Training (with --resume if a checkpoint exists)..."
python scripts/train_fno.py --config "$CONFIG" --resume

# --- 3) Eval on test split (writes FD-schema .npz) ---
echo ""
echo "[FNO] Evaluating on test split..."
python scripts/eval_fno.py --config "$CONFIG" --n 100

# --- 4) Optional inverse demo ---
echo ""
echo "[FNO] Inverse-M demo..."
python scripts/inverse_fno.py --config "$CONFIG" || echo "[FNO] inverse step failed (non-fatal)"

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
