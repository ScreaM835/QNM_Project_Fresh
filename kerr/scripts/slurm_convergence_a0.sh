#!/bin/bash
#SBATCH -J qnm_conv_a0
#SBATCH -A fergusson-sl3-cpu
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH -o qnm_conv_a0_%j.out
#SBATCH -e qnm_conv_a0_%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd /home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved/kerr
mkdir -p outputs/phase_a

echo "host=$(hostname)  cpus=$SLURM_CPUS_PER_TASK  job=$SLURM_JOB_ID"
echo "start=$(date -Iseconds)"

../venv_csd3/bin/python scripts/convergence_a0.py 2>&1 | tee outputs/phase_a/conv_sweep_${SLURM_JOB_ID}.log

echo "end=$(date -Iseconds)"
