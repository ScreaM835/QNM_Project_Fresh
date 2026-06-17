#!/bin/bash
# ============================================================
# SLURM Job: FNO operator-learning pipeline — sapphire/SL4 route.
# Identical pipeline to scripts/slurm_fno.sh; only the queue/account
# changes so jobs schedule sooner when icelake is congested.
# ============================================================
#SBATCH --job-name=qnm_fno
#SBATCH --output=qnm_fno_%j.out
#SBATCH --error=qnm_fno_%j.err
#SBATCH --account=mphil-dis-sl2-cpu
#SBATCH --qos=cpu1
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00

export PYTHONUNBUFFERED=1
set -e

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG=${1:-configs/fno_zerilli_l2.yaml}

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Account:       $SLURM_JOB_ACCOUNT"
echo "Started:       $(date)"
echo "Config:        $CONFIG"
echo "============================================"

cd "$WORKDIR"

# --- Environment setup ---
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

if [ ! -f "venv_csd3/bin/activate" ]; then
    echo "[SETUP] Creating venv..."
    python3 -m venv venv_csd3
fi
source venv_csd3/bin/activate
echo "[SETUP] Python: $(python --version)"

export TMPDIR="$WORKDIR/.pip_tmp"
mkdir -p "$TMPDIR"
pip install --quiet .
pip install --quiet neuraloperator
rm -rf "$TMPDIR"
unset TMPDIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- 1) Generate dataset (skip if already present) ---
DATA_PATH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['data_path'])")
if [ ! -f "$DATA_PATH" ]; then
    echo "[FNO] Generating dataset -> $DATA_PATH"
    python scripts/generate_fno_dataset.py --config "$CONFIG"
else
    echo "[FNO] Dataset already exists at $DATA_PATH — skipping generation"
fi

# --- 2) Train FNO ---
echo ""
echo "[FNO] Training..."
python scripts/train_fno.py --config "$CONFIG" --resume

# --- 3) Eval on test split (writes FD-schema .npz) ---
echo ""
echo "[FNO] Evaluating on test split..."
python scripts/eval_fno.py --config "$CONFIG" --n 100

# --- 4) Optional inverse demo ---
echo ""
echo "[FNO] Inverse-M demo..."
python scripts/inverse_fno.py --config "$CONFIG" || echo "[FNO] inverse step failed (non-fatal)"

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
