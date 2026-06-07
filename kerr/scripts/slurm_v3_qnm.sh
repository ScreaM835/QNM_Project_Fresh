#!/bin/bash
#SBATCH --job-name=qnm_v3
#SBATCH --account=FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=qnm_v3_%j.out
#SBATCH --error=qnm_v3_%j.err

set -euo pipefail

cd /home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
# Use the IMPROVED-repo venv: it has the REAL `qnm` package (modes_cache).
# The sibling project32_qnm_pinn venv only has a stray qnm.py that shadows it.
PY=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved/venv_csd3/bin/python

mkdir -p kerr/outputs/phase_a
LOG=kerr/outputs/phase_a/v3_qnm_${SLURM_JOB_ID}.log

echo "=== V.3 QNM extraction gate, job ${SLURM_JOB_ID} ===" | tee "${LOG}"
hostname | tee -a "${LOG}"
date     | tee -a "${LOG}"

"${PY}" kerr/scripts/v3_qnm_extraction.py 2>&1 | tee -a "${LOG}"

echo "=== done at $(date) ===" | tee -a "${LOG}"
