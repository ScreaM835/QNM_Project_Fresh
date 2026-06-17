#!/bin/bash
# Build the ell=4 Kerr surrogate corpus (the genuine QNM-rescue regime: fine
# teacher trustworthy <0.1%, coarse k4 prior fails the QNM 42-58% at high spin).
# Same audited solver + grids as the ell=2 corpus; only ELL is changed (via the
# build_kerr_dataset_lscan.py wrapper, which sets the module global pre-fork).
# Writes to kerr/outputs/phase_c_l4/ (does NOT touch the ell=2 corpus).
#SBATCH --job-name=qnm_l4data
#SBATCH --account=FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=qnm_l4data_%j.out
#SBATCH --error=qnm_l4data_%j.err

set -euo pipefail

cd /home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
PY=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved/venv_csd3/bin/python

# One Sobol sample per pool worker -> pin every BLAS/OMP pool to 1 thread so 32
# workers do not oversubscribe (validated bit-identical to serial).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
WORKERS=${SLURM_CPUS_PER_TASK:-32}

OUTDIR=kerr/outputs/phase_c_l4
mkdir -p "${OUTDIR}"
LOG=${OUTDIR}/build_${SLURM_JOB_ID}.log

echo "=== ell=4 Kerr corpus build, job ${SLURM_JOB_ID} ===" | tee "${LOG}"
hostname | tee -a "${LOG}"
date     | tee -a "${LOG}"
echo "workers=${WORKERS}  ell=4  (fine 801 + coarse 401/201; Sobol a/M,r0,w)" | tee -a "${LOG}"

for SPLIT in train val test; do
    echo "--- building ell=4 split: ${SPLIT} ---" | tee -a "${LOG}"
    "${PY}" -u kerr/scripts/build_kerr_dataset_lscan.py \
        --ell 4 --split "${SPLIT}" --workers "${WORKERS}" \
        --out "${OUTDIR}/dataset_${SPLIT}.npz" 2>&1 | tee -a "${LOG}"
done

echo "--- verifying ell=4 corpus ---" | tee -a "${LOG}"
"${PY}" -u kerr/scripts/build_kerr_dataset_lscan.py --ell 4 --verify-corpus \
    --out "${OUTDIR}/dataset_train.npz" 2>&1 | tee -a "${LOG}"

echo "=== done at $(date) ===" | tee -a "${LOG}"
