#!/bin/bash
# ============================================================
# SLURM Job: Curriculum PINN Training (3 windows) + QNM extraction
# Splits t=[0,50] into [0,17], [17,34], [34,50].
# ============================================================
#SBATCH --job-name=qnm_pinn_curr3w
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
CONFIG=configs/zerilli_l2_curriculum_3w.yaml

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

# Serialize pip install across concurrent jobs sharing this workspace
(
    flock -x 200
    rm -rf build/ dist/ src/*.egg-info
    export TMPDIR="$WORKDIR/.pip_tmp"
    mkdir -p "$TMPDIR"
    pip install --quiet .
    rm -rf "$TMPDIR" build/ dist/
    unset TMPDIR
) 200>"$WORKDIR/.pip.lock"

echo "[GPU] Checking GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
"
nvidia-smi 2>/dev/null || echo "(No GPU driver — running on CPU)"

echo ""
echo "============================================"
echo "[CURRICULUM 3W] Training 3-window PINN..."
echo "  Config: $CONFIG"
echo "  Window 1: t=[0, 17]  (analytic IC)"
echo "  Window 2: t=[17, 34] (numerical IC from W1)"
echo "  Window 3: t=[34, 50] (numerical IC from W2)"
echo "  Each window: 10k Adam + 30k L-BFGS, greedy f=0.3"
echo "============================================"
python scripts/run_pinn_curriculum_nw.py --config "$CONFIG" --checkpoint-every 500

echo ""
echo "============================================"
echo "[QNM] Extracting QNMs from PINN output..."
echo "============================================"
python scripts/extract_qnm.py --config "$CONFIG" --source pinn

echo ""
echo "============================================"
echo "Finished:      $(date)"
echo "============================================"
