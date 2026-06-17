#!/bin/bash
# ============================================================
# SLURM Job: FNO eval-only — for runs that hit wallclock during
# training but have valid best.pt / model.pt.  Runs eval_fno.py
# and (optionally) inverse_fno.py; skips dataset gen and training.
# ============================================================
#SBATCH --job-name=qnm_fno_eval
#SBATCH --output=qnm_fno_%j.out
#SBATCH --error=qnm_fno_%j.err
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

export PYTHONUNBUFFERED=1
set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG=${1:-configs/fno_zerilli_l2.yaml}

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Config:        $CONFIG"
echo "Started:       $(date)"
echo "Mode:          EVAL-ONLY (no training, no dataset gen)"
echo "============================================"

cd "$WORKDIR"

# --- Storage guard: outputs/ MUST be a symlink to RDS (/home is 52 GB quota).
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

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

# venv_gpu must already exist (created by the training job)
if [ ! -f venv_gpu/bin/activate ]; then
    echo "[FATAL] venv_gpu not present — run training job first."
    exit 2
fi
# shellcheck disable=SC1091
source venv_gpu/bin/activate
echo "[SETUP] Python: $(python --version) at $(which python)"

python - <<'PYEOF'
import sys, torch
print(f"[CUDA] torch={torch.__version__} cuda_build={torch.version.cuda}")
if not torch.cuda.is_available():
    print("[FATAL] cuda not available"); sys.exit(2)
print(f"[CUDA] device 0: {torch.cuda.get_device_name(0)}")
PYEOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "[FNO] Evaluating on test split..."
python scripts/eval_fno.py --config "$CONFIG" --n 100

echo ""
echo "[FNO] Inverse-M demo..."
python scripts/inverse_fno.py --config "$CONFIG" || echo "[FNO] inverse step failed (non-fatal)"

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
