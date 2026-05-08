#!/bin/bash
# ============================================================
# SLURM Job: Method 5 (2-D (t0, t_end) stability scan) on
# all available PINN waveforms (forward + inverse variants).
# Pure post-hoc curve_fit, no PINN training. Single CPU.
# ============================================================
#SBATCH --job-name=qnm_m5_2d
#SBATCH --output=qnm_m5_%j.out
#SBATCH --error=qnm_m5_%j.err
#SBATCH --account=mphil-dis-sl2-cpu
#SBATCH --qos=cpu1
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

export PYTHONUNBUFFERED=1
set -e

WORKDIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
VENV_DIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn/venv_csd3

echo "============================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Started:  $(date)"
echo "============================================"

cd "$WORKDIR"

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

source "$VENV_DIR/bin/activate"

# Configs to scan. Each must have:
#   outputs/pinn/<name>/<name>_pinn.npz
# already on disk. Variant E (zerilli_l2_inverse_qnm_2mode) is included
# but skipped automatically if its npz isn't there yet.
CONFIGS=(
    configs/zerilli_l2_greedy_f03_lbfgs30k.yaml
    configs/zerilli_l2_inverse_qnm.yaml
    configs/zerilli_l2_inverse_qnm_tring18.yaml
    configs/zerilli_l2_inverse_qnm_combo.yaml
    configs/zerilli_l2_inverse_qnm_2mode.yaml
)

for CFG in "${CONFIGS[@]}"; do
    NAME=$(basename "$CFG" .yaml)
    NPZ="outputs/pinn/${NAME}/${NAME}_pinn.npz"
    echo ""
    echo "============================================"
    echo "[M5] config = $CFG"
    if [ ! -f "$NPZ" ]; then
        echo "[M5] SKIP: $NPZ not found"
        continue
    fi
    echo "============================================"
    python scripts/extract_qnm.py \
        --config "$CFG" \
        --source pinn \
        --two-mode \
        --two-mode-2d \
        --two-mode-2d-n-t0 10 \
        --two-mode-2d-n-te 6
done

echo ""
echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
