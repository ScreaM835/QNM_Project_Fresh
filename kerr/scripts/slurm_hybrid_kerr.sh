#!/bin/bash
# ============================================================
# SLURM job: comprehensive Kerr hybrid-FNO run (assemble + train + eval + plot)
# in a single job. A100 / venv_gpu, same recipe as the SW hybrid GPU job.
# Usage:
#   sbatch kerr/scripts/slurm_hybrid_kerr.sh [kerr/configs/hybrid_kerr.yaml]
# ============================================================
#SBATCH --job-name=qnm_khyb
#SBATCH --output=qnm_khyb_%j.out
#SBATCH --error=qnm_khyb_%j.err
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00

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

# kerr/outputs is a real dir on /home (dataset already lives there, ~13 GB);
# this job writes only small artefacts (model.pt, figs, json), so no /rds guard.

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

# --- Ensure the FULL qnm package (with qnm.angular) + python deps ------------
# The pre-existing venv_gpu shipped a single-file qnm.py STUB (the SW FNO never
# imported the Teukolsky operator); the Kerr QNM extractor needs the real
# package's qnm.angular. Self-heal unconditionally (idempotent).
export TMPDIR="$WORKDIR/.pip_tmp"; mkdir -p "$TMPDIR"
if ! python -c "import qnm.angular" 2>/dev/null; then
    echo "[SETUP] full qnm package missing -> installing (removing stub) ..."
    SITE=$(python -c "import site; print(site.getsitepackages()[0])")
    rm -f "$SITE/qnm.py" "$SITE/qnm.pyc"
    python -m pip install --quiet --force-reinstall "qnm==0.4.4"
fi
for mod in yaml scipy matplotlib neuralop threadpoolctl; do
    python -c "import $mod" 2>/dev/null || python -m pip install --quiet \
        "$( [ "$mod" = yaml ] && echo pyyaml || ([ "$mod" = neuralop ] && echo neuraloperator || echo "$mod") )"
done
rm -rf "$TMPDIR"; unset TMPDIR
python -c "import qnm.angular, yaml, scipy, matplotlib, neuralop; print('[SETUP] deps OK: qnm.angular + yaml/scipy/matplotlib/neuralop')"

# Let the CPU-side assembly (einsum upsample) use the allocation.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python kerr/scripts/train_eval_hybrid_kerr.py --config "$CONFIG"

echo "Finished: $(date)"
