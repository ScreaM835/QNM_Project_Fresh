#!/bin/bash
# ============================================================
# SLURM Job: Inverse PINN + QNM (DIAGNOSTIC D, A+B+C combo)
#   t_ring_min:    10.0  -> 18.0
#   lambda_ring:   1.0   -> 100.0
#   n_ring_points: 200   -> 1000
# Output directory differs via experiment.name so this run cannot
# overwrite the baseline or the A/B/C single-variable runs.
# ============================================================
#SBATCH --job-name=qnm_pinn_iqD
#SBATCH --output=qnm_pinn_%j.out
#SBATCH --error=qnm_pinn_%j.err
#SBATCH --account=mphil-dis-sl2-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300

export PYTHONUNBUFFERED=1

set -e

WORKDIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
VENV_DIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn/venv_csd3
CONFIG=configs/zerilli_l2_inverse_qnm_combo.yaml

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
echo "[INVERSE+QNM D: A+B+C combo]"
echo "  Config: $CONFIG"
echo "  t_ring_min=18, lambda_ring=100, n_ring_points=1000"
echo "============================================"
python scripts/run_pinn_inverse_qnm.py --config "$CONFIG"

echo ""
echo "============================================"
echo "[QNM] Extracting QNMs from PINN output..."
echo "============================================"
python scripts/extract_qnm.py --config "$CONFIG" --source pinn

echo ""
echo "============================================"
echo "Finished:      $(date)"
echo "============================================"
