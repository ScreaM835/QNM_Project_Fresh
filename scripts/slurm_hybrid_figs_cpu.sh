#!/bin/bash
# ============================================================
# SLURM Job: regenerate the hybrid paper figures for one run (CPU).
# Same as slurm_hybrid_figs.sh but on the icelake CPU partition: figure
# generation is a single-BH FD solve plus one FNO forward pass, which runs
# comfortably on CPU and avoids the GPU-minutes budget. Uses venv_gpu (torch
# in CPU mode) because the figure script imports torch + neuraloperator.
# Figures are written under the run's logging.out_dir/figs, so a new
# experiment never overwrites the figures of another run.
# Usage:
#   sbatch scripts/slurm_hybrid_figs_cpu.sh \
#       configs/hybrid_sw_drp7_k2_h64.yaml configs/hybrid_sw_drp7_dataset.yaml
# ============================================================
#SBATCH --job-name=qnm_figs
#SBATCH --output=qnm_figs_%j.out
#SBATCH --error=qnm_figs_%j.err
#SBATCH --account=fergusson-sl3-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

export PYTHONUNBUFFERED=1
set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG=${1:-configs/hybrid_sw_train_k2_h64.yaml}
DATASET_CFG=${2:-configs/hybrid_sw_dataset.yaml}

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Started:       $(date)"
echo "Config:        $CONFIG"
echo "Dataset cfg:   $DATASET_CFG"
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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONWARNINGS=ignore

echo ""
echo "[FIGS] running make_hybrid_paper_figs.py ..."
python scripts/make_hybrid_paper_figs.py \
    --config "$CONFIG" --dataset-cfg "$DATASET_CFG" --recompute

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
