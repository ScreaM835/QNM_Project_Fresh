#!/bin/bash
#SBATCH --job-name=qnm_pinn_c3
#SBATCH --account=FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=qnm_pinn_c3_%j.out
#SBATCH --error=qnm_pinn_c3_%j.err

set -euo pipefail

cd /home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
# Must use the _improved-root venv (Teukolsky operator imports qnm.angular).
PY=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved/venv_csd3/bin/python

# A SINGLE PINN trains here (not a process pool), so let torch use all cores for
# the dense matmuls / autodiff. Pin the per-op thread pools to the allocation.
NT=${SLURM_CPUS_PER_TASK:-32}
export OMP_NUM_THREADS=${NT}
export MKL_NUM_THREADS=${NT}
export OPENBLAS_NUM_THREADS=${NT}
export NUMEXPR_NUM_THREADS=${NT}

OUTDIR=kerr/outputs/phase_c
mkdir -p "${OUTDIR}"
LOG=${OUTDIR}/pinn_c3_${SLURM_JOB_ID}.log

echo "=== C.3 single-config Kerr PINN proof, job ${SLURM_JOB_ID} ===" | tee "${LOG}"
hostname | tee -a "${LOG}"
date     | tee -a "${LOG}"
echo "threads=${NT}" | tee -a "${LOG}"

# Two single-config proofs: a/M=0 (real-field sanity) and a/M=0.7 (genuinely
# complex). Each writes its own JSON with the C.3 metrics + PASS/FAIL.
# Baseline architecture: plain FNN (no Fourier, no causal), FP64, hard-IC
# ansatz; Adam -> L-BFGS. Add --fourier to opt into Fourier features.
#
# Iteration budget sized from a single-core login-node calibration (the login
# node exposes nproc=1, so this is the WORST case; the icelake compute node has
# 32 real cores and should be faster, finishing early). At 64^4 / 15k points:
#   Adam   ~0.58 s/iter  -> 10k = ~1.6 h
#   L-BFGS ~5.4  s/iter  ->  5k = ~7.5 h   (each iter does a ~10-eval line search)
# Total ~9.1 h + FD oracle + eval, fitting the 12 h wall with ~2.9 h margin.
# (Matching the SW paper's 15k L-BFGS would be ~24 h single-core -> infeasible.)
ADAM=${ADAM:-10000}
LBFGS=${LBFGS:-5000}
NDOM=${NDOM:-15000}
NF=${NF:-64}
HIDDEN="${HIDDEN:-64 64 64 64}"
SPINS="${SPINS:-0.0}"
# FOURIER=1 opts into the Fourier-feature input embedding (Tancik 2020 / Ding
# 2024); the plain-FNN baseline (FOURIER=0) failed C.3 with spectral bias.
FOURIER=${FOURIER:-0}
# SUFFIX is appended to the output JSON name so a Fourier re-run does NOT clobber
# the committed-area baseline result (kept as failure evidence).
SUFFIX="${SUFFIX:-}"

FOURIER_FLAG=""
if [ "${FOURIER}" = "1" ]; then
    FOURIER_FLAG="--fourier"
fi

echo "fourier=${FOURIER} (flag='${FOURIER_FLAG}')  suffix='${SUFFIX}'" | tee -a "${LOG}"

for SPIN in ${SPINS}; do
    TAG=$(echo "${SPIN}" | sed 's/\.//')
    echo "--- training PINN at a/M=${SPIN} ---" | tee -a "${LOG}"
    "${PY}" -u kerr/scripts/train_kerr_pinn.py \
        --spin "${SPIN}" \
        --adam "${ADAM}" --lbfgs "${LBFGS}" \
        --num-domain "${NDOM}" --n-fourier "${NF}" \
        --hidden ${HIDDEN} ${FOURIER_FLAG} \
        --out "${OUTDIR}/pinn_single_a${TAG}${SUFFIX}.json" 2>&1 | tee -a "${LOG}"
done

echo "=== done at $(date) ===" | tee -a "${LOG}"
