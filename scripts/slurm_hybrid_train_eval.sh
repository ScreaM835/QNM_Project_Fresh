#!/bin/bash
# ============================================================
# SLURM Job: Hybrid FNO TRAIN + EVAL in one job (modes_t sweep).
# Trains the residual FNO for one config, then immediately evaluates it
# (field MSE + QNM extraction) into <out_dir>/eval. Modelled on
# scripts/slurm_hybrid_fno.sh (same A100/ampere, venv_gpu, storage guard).
#   sbatch scripts/slurm_hybrid_train_eval.sh configs/hybrid_sw_modes32.yaml
# ============================================================
#SBATCH --job-name=qnm_msweep
#SBATCH --output=qnm_msweep_%j.out
#SBATCH --error=qnm_msweep_%j.err
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=11:00:00

export PYTHONUNBUFFERED=1
set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG=${1:?usage: sbatch slurm_hybrid_train_eval.sh <config.yaml>}

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Started:       $(date)"
echo "Config:        $CONFIG"
echo "============================================"

cd "$WORKDIR"

# --- Storage guard ----------------------------------------------------------
if [ ! -L outputs ]; then
    echo "[FATAL] outputs/ is not a symlink. Refusing to write to /home."
    exit 3
fi
OUTPUTS_TARGET=$(readlink -f outputs)
echo "[STORAGE] outputs -> $OUTPUTS_TARGET"
case "$OUTPUTS_TARGET" in
    /rds/*) ;;
    *) echo "[FATAL] outputs symlink does NOT point into /rds/ (got: $OUTPUTS_TARGET)."; exit 3 ;;
esac

# --- Environment ------------------------------------------------------------
module purge
module load rhel8/default-amp
module load python/3.11.0-icl
# shellcheck disable=SC1091
source venv_gpu/bin/activate
echo "[SETUP] Python: $(python --version) at $(which python)"

python - <<'PYEOF'
import sys, torch
print(f"[CUDA] torch={torch.__version__} cuda_build={torch.version.cuda}")
if not torch.cuda.is_available():
    print("[FATAL] torch.cuda.is_available() = False"); sys.exit(2)
print(f"[CUDA] device 0: {torch.cuda.get_device_name(0)}")
PYEOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Train ------------------------------------------------------------------
echo ""
echo "[SWEEP] modes_t = $(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['fno']['modes_t'])")"
echo "[SWEEP] Training residual FNO (resume-safe) ..."
python scripts/train_hybrid_fno.py --config "$CONFIG" --resume

# --- Eval -------------------------------------------------------------------
echo ""
echo "[SWEEP] Evaluating (field MSE + QNM) ..."
python scripts/eval_hybrid_sw.py --config "$CONFIG"

# --- Figures + consolidated metrics (one-shot; non-fatal) -------------------
# Emits the standard plot set (pointwise error, grad-vs-error 3-panel, ...) and
# a single machine-readable <out_dir>/eval/report.json (canonical field rL2 +
# speckle/clean-region contamination + grad-error correlation, merged with the
# population eval summary). Wrapped so a plotting failure can never fail the job
# after the eval metrics are already written.
echo ""
echo "[SWEEP] Generating figures + report.json ..."
python scripts/make_hybrid_paper_figs.py --config "$CONFIG" \
    --dataset-cfg configs/hybrid_sw_dataset.yaml --recompute \
    || echo "[WARN] figure/metrics step failed (eval metrics already saved)"

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
