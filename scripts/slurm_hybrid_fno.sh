#!/bin/bash
# ============================================================
# SLURM Job: Hybrid coarse-FD + FNO-residual training.
# Modelled on scripts/slurm_fno_ampere.sh — same A100 partition,
# same venv_gpu, same storage guard. Trains one (config = one k value).
# Usage:
#   sbatch scripts/slurm_hybrid_fno.sh configs/hybrid_sw_train_k2.yaml
#   sbatch scripts/slurm_hybrid_fno.sh configs/hybrid_sw_train_k4.yaml
# ============================================================
#SBATCH --job-name=qnm_hybrid
#SBATCH --output=qnm_hybrid_%j.out
#SBATCH --error=qnm_hybrid_%j.err
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00

export PYTHONUNBUFFERED=1
set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG=${1:-configs/hybrid_sw_train_k2.yaml}

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Started:       $(date)"
echo "Config:        $CONFIG"
echo "============================================"

cd "$WORKDIR"

# --- Storage guard (matches slurm_fno_ampere.sh) ----------------------------
if [ ! -L outputs ]; then
    echo "[FATAL] outputs/ is not a symlink. Refusing to write to /home (quota=52GB)."
    exit 3
fi
OUTPUTS_TARGET=$(readlink -f outputs)
echo "[STORAGE] outputs -> $OUTPUTS_TARGET"
case "$OUTPUTS_TARGET" in
    /rds/*) ;;
    *) echo "[FATAL] outputs symlink does NOT point into /rds/ (got: $OUTPUTS_TARGET)."; exit 3 ;;
esac

# --- Environment -------------------------------------------------------------
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

VENV=venv_gpu
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[SETUP] Creating $VENV on compute node ..."
    python3 -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export TMPDIR="$WORKDIR/.pip_tmp"
    mkdir -p "$TMPDIR"
    python -m pip install --quiet --upgrade pip wheel setuptools
    python -m pip install --quiet \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu118
    python -m pip install --quiet .
    python -m pip install --quiet neuraloperator
    rm -rf "$TMPDIR"; unset TMPDIR
else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
echo "[SETUP] Python: $(python --version) at $(which python)"

# --- HARD CUDA assertion ----------------------------------------------------
python - <<'PYEOF'
import sys, torch
print(f"[CUDA] torch={torch.__version__} cuda_build={torch.version.cuda}")
if not torch.cuda.is_available():
    print("[FATAL] torch.cuda.is_available() = False"); sys.exit(2)
print(f"[CUDA] device 0: {torch.cuda.get_device_name(0)}")
x = torch.randn(2048, 2048, device='cuda'); torch.cuda.synchronize()
print(f"[CUDA] matmul ok, sum={(x @ x).sum().item():.3e}")
PYEOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Sanity: dataset must exist ---------------------------------------------
DATA_PATH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['dataset']['path'])")
if [ ! -f "$DATA_PATH" ]; then
    echo "[FATAL] hybrid dataset missing at $DATA_PATH — build with scripts/build_hybrid_dataset.py"
    exit 4
fi
echo "[HYBRID] dataset: $DATA_PATH ($(du -h "$DATA_PATH" | cut -f1))"

# Richardson pipeline (optional): a second, sample-aligned k2 coarse dataset.
DATA_PATH_K2=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['dataset'].get('path_k2',''))")
if [ -n "$DATA_PATH_K2" ]; then
    if [ ! -f "$DATA_PATH_K2" ]; then
        echo "[FATAL] Richardson k2 dataset missing at $DATA_PATH_K2"
        exit 4
    fi
    echo "[HYBRID] k2 dataset: $DATA_PATH_K2 ($(du -h "$DATA_PATH_K2" | cut -f1))"
fi

# --- Train ------------------------------------------------------------------
echo ""
echo "[HYBRID] Training residual FNO (resume-safe) ..."
python scripts/train_hybrid_fno.py --config "$CONFIG" --resume

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
