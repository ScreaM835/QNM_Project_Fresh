#!/bin/bash
# ============================================================
# SLURM Job: Richardson A/B overfit diagnostic (CPU).
# Runs scripts/exploration/_tmp_richardson_overfit.py which fits the REAL
# hybrid FNO on ONE M=1 sample for TWO delta targets from identical init:
#   (1) supervised  Phi_fine - P4   (KNOWN-GOOD positive control, real
#       pipeline fits this to 0.07%)
#   (2) richardson  Phi_R   - P4   (label-free method under test)
# Verdict logic:
#   both stall ~4.9%  -> single-sample toy is the bottleneck (NOT the target)
#                        => move to real multi-sample training.
#   supervised descends but richardson stalls -> Richardson target is the
#                        problem => investigate before wiring.
# Usage:  sbatch scripts/slurm_richardson_ab.sh
# ============================================================
#SBATCH --job-name=qnm_rich_ab
#SBATCH --output=qnm_rich_ab_%j.out
#SBATCH --error=qnm_rich_ab_%j.err
#SBATCH --account=fergusson-sl3-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

export PYTHONUNBUFFERED=1
set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV=venv_csd3

echo "============================================"
echo "Job ID:        ${SLURM_JOB_ID:-NA}"
echo "Node:          ${SLURM_NODELIST:-NA}"
echo "Partition:     ${SLURM_JOB_PARTITION:-NA}"
echo "Started:       $(date)"
echo "============================================"

cd "$WORKDIR"

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

python scripts/exploration/_tmp_richardson_overfit.py

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
