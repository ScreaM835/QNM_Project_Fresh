#!/bin/bash
# ============================================================
# SLURM Job: Richardson-target vs FNO QNM diagnostic (CPU, pure numpy/scipy).
# Tests whether the QNM degradation lives in the Richardson TARGET (Phi_R) or
# in the FNO network, over the full 100-BH test set.
#   sbatch scripts/slurm_richardson_target_qnm.sh
# ============================================================
#SBATCH --job-name=qnm_rtgt
#SBATCH --output=qnm_rtgt_%j.out
#SBATCH --error=qnm_rtgt_%j.err
#SBATCH --account=FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PY=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved/venv_csd3/bin/python

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST started $(date)"
"$PY" scripts/diag_richardson_target_qnm.py 100
echo "Finished $(date)"
