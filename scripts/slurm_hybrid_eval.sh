#!/bin/bash
# ============================================================
# SLURM Job: Hybrid surrogate evaluation (field MSE + QNM extraction).
# Reuses the same environment as slurm_hybrid_fno.sh.
# Usage:
#   sbatch scripts/slurm_hybrid_eval.sh configs/hybrid_sw_train_k2_h64.yaml
# ============================================================
#SBATCH --job-name=qnm_eval
#SBATCH --output=qnm_eval_%j.out
#SBATCH --error=qnm_eval_%j.err
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
CONFIG=${1:-configs/hybrid_sw_train_k2_h64.yaml}

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Started:       $(date)"
echo "Config:        $CONFIG"
echo "============================================"

cd "$WORKDIR"

if [ ! -L outputs ]; then
    echo "[FATAL] outputs/ is not a symlink."; exit 3
fi
echo "[STORAGE] outputs -> $(readlink -f outputs)"

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

# shellcheck disable=SC1091
source venv_gpu/bin/activate
echo "[SETUP] Python: $(python --version) at $(which python)"

python - <<'PYEOF'
import torch
print(f"[CUDA] torch={torch.__version__}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[CUDA] device 0: {torch.cuda.get_device_name(0)}")
PYEOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "[EVAL] running eval_hybrid_sw.py ..."
python scripts/eval_hybrid_sw.py --config "$CONFIG"

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
