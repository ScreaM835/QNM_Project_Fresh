#!/bin/bash
#SBATCH --job-name=qnm_cdata
#SBATCH --account=FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=qnm_cdata_%j.out
#SBATCH --error=qnm_cdata_%j.err

set -euo pipefail

cd /home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
# NB: must use the _improved-root venv -- the Teukolsky operator imports
# qnm.angular (spheroidal separation constants), which the parent-root Phase A
# venv lacks (its qnm is a single-file stub; Phase A RWZ never needed angular).
PY=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved/venv_csd3/bin/python

# Each Sobol sample is one process-pool worker; pin every numerical library to a
# SINGLE thread so 32 workers do not oversubscribe the node (BLAS/OMP would
# otherwise spawn 32x32 threads). Validated bit-identical to the serial path.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
WORKERS=${SLURM_CPUS_PER_TASK:-32}

OUTDIR=kerr/outputs/phase_c
mkdir -p "${OUTDIR}"
LOG=${OUTDIR}/build_${SLURM_JOB_ID}.log

echo "=== C.2 Kerr surrogate corpus build, job ${SLURM_JOB_ID} ===" | tee "${LOG}"
hostname | tee -a "${LOG}"
date     | tee -a "${LOG}"
echo "workers=${WORKERS}  (fine N=801 + coarse N=401/201; Sobol over a/M,r0,w)" | tee -a "${LOG}"

# Authoritative corpus: 1024 train / 128 val / 128 test, full window
# (T_STORE=220, DT_STORE=0.25). Splits are disjoint Sobol slices (seed 0).
for SPLIT in train val test; do
    echo "--- building split: ${SPLIT} ---" | tee -a "${LOG}"
    "${PY}" -u kerr/scripts/build_kerr_dataset.py \
        --split "${SPLIT}" --workers "${WORKERS}" \
        --out "${OUTDIR}/dataset_${SPLIT}.npz" 2>&1 | tee -a "${LOG}"
done

echo "--- verifying corpus (C.2 acceptance) ---" | tee -a "${LOG}"
"${PY}" -u kerr/scripts/build_kerr_dataset.py --verify-corpus \
    --out "${OUTDIR}/dataset_train.npz" 2>&1 | tee -a "${LOG}"

echo "=== done at $(date) ===" | tee -a "${LOG}"
