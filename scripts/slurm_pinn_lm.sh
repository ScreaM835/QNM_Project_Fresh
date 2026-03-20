#!/bin/bash
# ============================================================
# SLURM Job: PINN Training with Levenberg-Marquardt + QNM extraction
# Adam warm-up -> LM phase (replaces L-BFGS)
# ============================================================
#SBATCH --job-name=qnm_pinn_lm
#SBATCH --output=qnm_pinn_%j.out
#SBATCH --error=qnm_pinn_%j.err
#SBATCH --account=fergusson-sl3-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=76
#SBATCH --time=12:00:00

export PYTHONUNBUFFERED=1

set -e

WORKDIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
VENV_DIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn/venv_csd3
CONFIG=configs/zerilli_l2_lm.yaml

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Started:       $(date)"
echo "============================================"

cd "$WORKDIR"

# --- Environment setup ---
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

source "$VENV_DIR/bin/activate"
echo "[SETUP] Python: $(python --version)"

# Install deps (with flock to prevent pip races)
(
    flock -x 200
    pip install --quiet -e /home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
) 200>"$VENV_DIR/.pip_lock"

# --- Maximize CPU Utilization ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[SETUP] Using $SLURM_CPUS_PER_TASK CPU threads for PyTorch"

# --- Train PINN (Adam + LM) ---
echo ""
echo "============================================"
echo "[PINN-LM] Training with Adam + Levenberg-Marquardt"
echo "  Config: $CONFIG"
echo "============================================"
python scripts/run_pinn_lm.py --config "$CONFIG"

# --- Extract QNMs ---
echo ""
echo "============================================"
echo "[QNM] Extracting QNMs from PINN output..."
echo "============================================"
python scripts/extract_qnm.py --config "$CONFIG" --source pinn

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
