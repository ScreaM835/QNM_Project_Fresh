#!/bin/bash
#SBATCH -J qnm_diag_a0
#SBATCH -A fergusson-sl3-cpu
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o qnm_diag_a0_%j.out
#SBATCH -e qnm_diag_a0_%j.err

set -euo pipefail

cd /home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved/kerr
mkdir -p outputs/phase_a

echo "host=$(hostname)  job=$SLURM_JOB_ID  start=$(date -Iseconds)"

../venv_csd3/bin/python scripts/diag_gauge_and_control.py 2>&1 | tee outputs/phase_a/diag_${SLURM_JOB_ID}.log

echo "end=$(date -Iseconds)"
