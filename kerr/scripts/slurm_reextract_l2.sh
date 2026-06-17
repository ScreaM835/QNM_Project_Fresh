#!/bin/bash
# ============================================================
# SLURM job: re-extract the finished ell=2 hybrid QNMs with the
# m1-m5 multi-method ENSEMBLE (single vs consensus side-by-side).
# Runs the FNO forward on a GPU node, then the QNM ensemble on CPUs.
# Usage:
#   sbatch kerr/scripts/slurm_reextract_l2.sh [kerr/configs/hybrid_kerr.yaml]
# ============================================================
#SBATCH --job-name=qnm_reext
#SBATCH --output=qnm_reext_%j.out
#SBATCH --error=qnm_reext_%j.err
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

export PYTHONUNBUFFERED=1
set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG=${1:-kerr/configs/hybrid_kerr.yaml}

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      ${SLURM_NODELIST:-?}"
echo "Partition: ${SLURM_JOB_PARTITION:-?}"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-?}"
echo "Started:   $(date)"
echo "Config:    $CONFIG"
echo "============================================"

cd "$WORKDIR"

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

VENV=venv_gpu
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[SETUP] Creating $VENV on compute node ..."
    python3 -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export TMPDIR="$WORKDIR/.pip_tmp"; mkdir -p "$TMPDIR"
    python -m pip install --quiet --upgrade pip wheel setuptools
    python -m pip install --quiet torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu118
    python -m pip install --quiet neuraloperator pyyaml scipy matplotlib qnm
    rm -rf "$TMPDIR"; unset TMPDIR
else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
echo "[SETUP] Python: $(python --version) at $(which python)"
python -c "import torch; print('[SETUP] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

OMP_NUM_THREADS=8 python -u kerr/scripts/reextract_qnm_ensemble_l2.py \
    --config "$CONFIG" --workers 16

echo "Finished: $(date)"
