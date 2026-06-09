#!/bin/bash
#SBATCH --job-name=qnm_kv1
#SBATCH --account=FERGUSSON-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=qnm_kv1_%j.out
#SBATCH --error=qnm_kv1_%j.err

set -euo pipefail

cd /home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
# NB: must use the _improved-root venv -- the Teukolsky operator imports
# qnm.angular (spheroidal separation constants), which the parent-root Phase A
# venv lacks (its qnm is a single-file stub; Phase A RWZ never needed angular).
PY=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved/venv_csd3/bin/python

mkdir -p kerr/outputs/phase_b
LOG=kerr/outputs/phase_b/kv1_${SLURM_JOB_ID}.log

echo "=== KV.1 Kerr propagation/stability gate, job ${SLURM_JOB_ID} ===" | tee "${LOG}"
hostname | tee -a "${LOG}"
date     | tee -a "${LOG}"

"${PY}" -u kerr/scripts/kv1_propagation.py 2>&1 | tee -a "${LOG}"

echo "=== done at $(date) ===" | tee -a "${LOG}"
