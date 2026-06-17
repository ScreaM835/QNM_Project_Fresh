#!/bin/bash
# ============================================================
# SLURM Job: fp64 L-BFGS finetune on top of trained hybrid FNO.
# Resume-safe via --resume; 12 h walltime per /memories/repo/slurm_rules.md.
# Usage:
#   sbatch scripts/slurm_hybrid_finetune_fp64.sh configs/hybrid_sw_train_k2.yaml
#   sbatch scripts/slurm_hybrid_finetune_fp64.sh configs/hybrid_sw_train_k4.yaml
# ============================================================
#SBATCH --job-name=qnm_hyb_fp64
#SBATCH --output=qnm_hyb_fp64_%j.out
#SBATCH --error=qnm_hyb_fp64_%j.err
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

export PYTHONUNBUFFERED=1
set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG=${1:-configs/hybrid_sw_train_k2.yaml}

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID    Config: $CONFIG"
echo "Started:   $(date)"
echo "============================================"
cd "$WORKDIR"

if [ ! -L outputs ]; then echo "[FATAL] outputs/ is not a symlink"; exit 3; fi
case "$(readlink -f outputs)" in /rds/*) ;; *) echo "[FATAL] outputs not on /rds"; exit 3 ;; esac

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

VENV=venv_gpu
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "[SETUP] Python: $(python --version) at $(which python)"

python - <<'PYEOF'
import sys, torch
print(f"[CUDA] torch={torch.__version__} cuda_build={torch.version.cuda}")
if not torch.cuda.is_available():
    print("[FATAL] cuda unavailable"); sys.exit(2)
print(f"[CUDA] device 0: {torch.cuda.get_device_name(0)}")
PYEOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATA_PATH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['dataset']['path'])")
OUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['logging']['out_dir'])")
[ -f "$DATA_PATH" ] || { echo "[FATAL] dataset missing: $DATA_PATH"; exit 4; }
[ -f "$OUT_DIR/model_best.pt" ] || { echo "[FATAL] fp32 starting weights missing: $OUT_DIR/model_best.pt"; exit 4; }

echo "[FP64] starting weights: $OUT_DIR/model_best.pt"
python scripts/finetune_hybrid_lbfgs_fp64.py --config "$CONFIG" --resume

echo "Finished: $(date)"
