#!/bin/bash
# ============================================================
# SLURM Job: Inverse PINN + 2-MODE QNM extraction (variant E)
# Combines D-combo hyperparameters with a two-mode ringdown template.
# ============================================================
#SBATCH --job-name=qnm_pinn_iqEf
#SBATCH --output=qnm_pinn_%j.out
#SBATCH --error=qnm_pinn_%j.err
#SBATCH --account=fergusson-sl3-cpu
#SBATCH --qos=cpu2
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300

export PYTHONUNBUFFERED=1

set -e

WORKDIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
VENV_DIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn/venv_csd3
CONFIG=configs/zerilli_l2_inverse_qnm_2mode.yaml

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Started:       $(date)"
echo "============================================"

cd "$WORKDIR"

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] Shared venv not found at $VENV_DIR" >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "[SETUP] Python: $(python --version)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[SETUP] OMP_NUM_THREADS=$OMP_NUM_THREADS"

(
    flock -x 200
    rm -rf build/ dist/ src/*.egg-info
    export TMPDIR="$WORKDIR/.pip_tmp"
    mkdir -p "$TMPDIR"
    pip install --quiet .
    rm -rf "$TMPDIR" build/ dist/
    unset TMPDIR
) 200>"$WORKDIR/.pip.lock"

echo ""
echo "============================================"
echo "[INVERSE+QNM E: 2-mode template]"
echo "  Config: $CONFIG"
echo "  Fundamental: omega_init=0.30  tau_init=8.0"
echo "  Overtone:    omega1_init=0.3467  tau1_init=3.674"
echo "  Window:      t_ring_min=18  lambda_ring=100  n_ring=1000"
echo "============================================"
python scripts/run_pinn_inverse_qnm_2mode.py --config "$CONFIG"

echo ""
echo "============================================"
echo "[QNM] Extracting QNMs from PINN output..."
echo "============================================"
python scripts/extract_qnm.py --config "$CONFIG" --source pinn --two-mode

echo ""
echo "============================================"
echo "Finished:      $(date)"
echo "============================================"
