#!/bin/bash
# 12h variant of slurm_hybrid_fno.sh (per workflow rule: resume-safe + 12h).
# Identical otherwise — single config arg.
# Usage: sbatch scripts/slurm_hybrid_fno_12h.sh <config.yaml>
#SBATCH --job-name=qnm_hybrid
#SBATCH --output=qnm_hybrid_%j.out
#SBATCH --error=qnm_hybrid_%j.err
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

exec bash scripts/slurm_hybrid_fno.sh "$@"
