#!/bin/bash
# ============================================================
# SLURM Job: Soft-overlap curriculum PINN + QNM extraction
# Mirrors slurm_pinn_greedy_f03_lbfgs30k.sh resources/workflow.
# ============================================================
#SBATCH --job-name=qnm_pinn_soft_overlap
#SBATCH --output=qnm_pinn_%j.out
#SBATCH --error=qnm_pinn_%j.err
#SBATCH --account=mphil-dis-sl2-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300

export PYTHONUNBUFFERED=1

set -e

WORKDIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
VENV_DIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn/venv_csd3
CONFIG=configs/zerilli_l2_soft_overlap_curriculum.yaml
OUTDIR="$WORKDIR/outputs/pinn/zerilli_l2_soft_overlap_curriculum"

requeue_handler() {
    echo "[SIGNAL] Caught USR1 - job approaching time limit."
    echo "[SIGNAL] Resubmitting job to continue training..."
    sbatch "$WORKDIR/scripts/slurm_pinn_soft_overlap_curriculum.sh"
    exit 0
}
trap requeue_handler USR1

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
nvidia-smi 2>/dev/null || echo "(No GPU driver - running on CPU)"

RESUME_FLAG=""
if ls "$OUTDIR"/checkpoints_w*/model*.pt 1>/dev/null 2>&1; then
    echo "[CKPT] Found existing checkpoint - will resume training"
    RESUME_FLAG="--resume"
fi

echo ""
echo "============================================"
echo "[CURRICULUM] Training soft-overlap temporal curriculum..."
echo "  W1: t=[0,30]"
echo "  W2: t=[20,50]"
echo "  Blend: t=[20,30] smoothstep"
echo "  Base: greedy f=0.3 + 10k Adam + 30k L-BFGS"
echo "  Checkpoint every 500 iters"
if [ -n "$RESUME_FLAG" ]; then
    echo "  >>> RESUMING from checkpoint <<<"
fi
echo "============================================"
python scripts/run_pinn_soft_overlap_curriculum.py --config "$CONFIG" --checkpoint-every 500 $RESUME_FLAG

echo ""
echo "============================================"
echo "[QNM] Extracting QNMs from PINN output..."
echo "============================================"
python scripts/extract_qnm.py --config "$CONFIG" --source pinn

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"