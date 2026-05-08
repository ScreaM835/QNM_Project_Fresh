#!/bin/bash
# ============================================================
# SLURM Job: Inverse PINN + QNM (DIAGNOSTIC, t_ring_min = 18)
# Sole change vs slurm_pinn_inverse_qnm.sh: CONFIG points to the
# t_ring_min = 18 variant. Output directory differs via
# experiment.name in the config, so this run cannot overwrite the
# baseline inverse_qnm artefacts.
# ============================================================
#SBATCH --job-name=qnm_pinn_iq18
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
CONFIG=configs/zerilli_l2_inverse_qnm_tring18.yaml

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
echo "[INVERSE+QNM tring18] Learning M, ω, τ from data"
echo "  Config: $CONFIG"
echo "  M_init = 0.8 (true = 1.0)"
echo "  ω_init = 0.30 (true = 0.3737)"
echo "  τ_init = 8.0  (true = 11.241)"
echo "  Noise = 1%"
echo "  t_ring_min = 18.0  (baseline: 10.0)"
echo "  Training: 10k Adam + 15k L-BFGS"
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
