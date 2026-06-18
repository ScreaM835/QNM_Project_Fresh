#!/bin/bash
# Build the ell=4 Kerr surrogate corpus on the COARSE LADDER (prior k4=101).
# Same audited solver as the ell=2 / ell=4 corpora; the ONLY change is the
# coarse grid sizes (k2=201, k4=101 instead of 401/201), passed via --coarse-n
# so the validated kerr_dataset.py is untouched. This is the genuine BOTH-AXES
# regime (spin-graded QNM headroom + large field headroom); see
# probe_coarse_ladder_l4.py / probe_worst_corner_l4.py.
# Writes to kerr/outputs/phase_c_l4_n101/ (does NOT touch other corpora).
#SBATCH --job-name=qnm_l4n101
#SBATCH --account=FERGUSSON-SL3-CPU
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=qnm_l4n101_%j.out
#SBATCH --error=qnm_l4n101_%j.err

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

COARSE="2:201,4:101"
OUTDIR=kerr/outputs/phase_c_l4_n101
mkdir -p "${OUTDIR}"
LOG=${OUTDIR}/build_${SLURM_JOB_ID}.log

echo "=== ell=4 coarse-ladder corpus build, job ${SLURM_JOB_ID} ===" | tee "${LOG}"
hostname | tee -a "${LOG}"
date     | tee -a "${LOG}"
echo "workers=${WORKERS}  ell=4  (fine 801 + coarse ${COARSE}; Sobol a/M,r0,w)" | tee -a "${LOG}"

for SPLIT in train val test; do
    echo "--- building ell=4 n101 split: ${SPLIT} ---" | tee -a "${LOG}"
    "${PY}" -u kerr/scripts/build_kerr_dataset_lscan.py \
        --ell 4 --coarse-n "${COARSE}" --split "${SPLIT}" --workers "${WORKERS}" \
        --out "${OUTDIR}/dataset_${SPLIT}.npz" 2>&1 | tee -a "${LOG}"
done

echo "--- verifying ell=4 n101 corpus ---" | tee -a "${LOG}"
"${PY}" -u kerr/scripts/build_kerr_dataset_lscan.py --ell 4 --coarse-n "${COARSE}" \
    --verify-corpus --out "${OUTDIR}/dataset_train.npz" 2>&1 | tee -a "${LOG}"

echo "=== done at $(date) ===" | tee -a "${LOG}"
