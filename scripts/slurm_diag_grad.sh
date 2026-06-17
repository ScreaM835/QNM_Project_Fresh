#!/bin/bash
#SBATCH -J qnm_diag
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --qos=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=qnm_diag_%j.out
#SBATCH --error=qnm_diag_%j.err

set -euo pipefail

CONFIG=${1:?usage: sbatch $0 <config.yaml>}

cd "$SLURM_SUBMIT_DIR"
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source venv_gpu/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('[CUDA]', torch.cuda.get_device_name(0))"

python scripts/diag_grad_precision.py --config "$CONFIG" --n_samples 64 --chunk 4 --device cuda
