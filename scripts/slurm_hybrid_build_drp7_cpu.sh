#!/bin/bash
# ============================================================
# SLURM Job: Build the DRP7 hybrid coarse/fine FD dataset (CPU).
# Mirrors the original build procedure (scripts/build_hybrid_dataset.py),
# but with the improved DRP7 coarse prior (configs/hybrid_sw_drp7_dataset.yaml).
# Builds BOTH k=2 and k=4 in one short CPU job (~minutes).
# Usage:
#   sbatch scripts/slurm_hybrid_build_drp7_cpu.sh
# ============================================================
#SBATCH --job-name=qnm_hyb_build
#SBATCH --output=qnm_hyb_build_%j.out
#SBATCH --error=qnm_hyb_build_%j.err
#SBATCH --account=fergusson-sl3-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

export PYTHONUNBUFFERED=1
set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG=configs/hybrid_sw_drp7_dataset.yaml
VENV=venv_csd3   # improved-repo CPU venv (numpy 2.4.4, scipy 1.17.1)

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Started:       $(date)"
echo "Config:        $CONFIG"
echo "============================================"

cd "$WORKDIR"

# --- Storage guard (outputs must be the RDS symlink) ------------------------
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

if [ ! -f "$VENV/bin/activate" ]; then
    echo "[FATAL] CPU venv not found at $VENV"; exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "[SETUP] Python: $(python --version) at $(which python)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Build both coarsening factors ------------------------------------------
for K in 2 4; do
    OUT="outputs/hybrid/dataset_sw_drp7_k${K}.npz"
    echo ""
    echo "[BUILD] k=${K} -> ${OUT}"
    python scripts/build_hybrid_dataset.py --config "$CONFIG" --k "$K" --out "$OUT"
    echo "[BUILD] done k=${K}: $(du -h "$OUT" | cut -f1)"
done

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
