#!/bin/bash
#SBATCH --job-name=qnm_v1
#SBATCH --account=FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=qnm_v1_%j.out
#SBATCH --error=qnm_v1_%j.err

set -euo pipefail

cd /home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
PY=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn/venv_csd3/bin/python

mkdir -p kerr/outputs/phase_a
LOG=kerr/outputs/phase_a/v1_flat_${SLURM_JOB_ID}.log

echo "=== V.1 flat-propagation gate, job ${SLURM_JOB_ID} ===" | tee "${LOG}"
hostname | tee -a "${LOG}"
date     | tee -a "${LOG}"

"${PY}" kerr/scripts/v1_flat_propagation.py 2>&1 | tee -a "${LOG}"

echo "=== done at $(date) ===" | tee -a "${LOG}"
