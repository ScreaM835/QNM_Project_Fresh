#!/bin/bash
# ============================================================
# SLURM Job: One-shot venv_gpu setup + CUDA smoke test on ampere.
#
# Purpose: prove that we can actually use the GPU on this cluster
# before committing an 8h training run. Does NOT touch venv_csd3.
#
# Steps:
#   1) Create venv_gpu on the compute node (fresh python venv).
#   2) Install torch 2.5.1+cu118 (cu118 is compatible with the
#      ampere node driver ~525.x; cu121+ would need driver 530+).
#   3) Install the project (pip install .) and neuraloperator.
#   4) Run a CUDA smoke test that asserts cuda.is_available() and
#      runs a real matmul on the GPU. Hard-exit if it fails.
#   5) Run scripts/train_fno.py --quick to confirm the FNO pipeline
#      itself runs end-to-end on CUDA.
# ============================================================
#SBATCH --job-name=fno_gpu_setup
#SBATCH --output=qnm_fno_setup_%j.out
#SBATCH --error=qnm_fno_setup_%j.err
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --qos=gpu1
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00

export PYTHONUNBUFFERED=1
set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPUs visible:  ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Started:       $(date)"
echo "WORKDIR:       $WORKDIR"
echo "============================================"

# --- Modules (same toolchain that already works on CSD3) ---
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

# --- nvidia-smi sanity ---
echo "--- nvidia-smi ---"
nvidia-smi || { echo "[FATAL] nvidia-smi failed — no GPU on this node"; exit 1; }
echo "------------------"

# --- Create venv_gpu (only if missing) ---
VENV=venv_gpu
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[SETUP] Creating $VENV ..."
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "[SETUP] Python: $(python --version) at $(which python)"

# pip cache & tmp on local fast scratch to avoid login filesystem pain
export TMPDIR="$WORKDIR/.pip_tmp"
mkdir -p "$TMPDIR"

python -m pip install --quiet --upgrade pip wheel setuptools

# --- Install torch 2.5.1 + cu118 (driver-compatible) ---
echo "[SETUP] Installing torch 2.5.1+cu118 ..."
python -m pip install --quiet \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118

# --- Install project (no deps re-resolve of torch) ---
echo "[SETUP] Installing project ..."
python -m pip install --quiet .

# --- Install neuraloperator ---
echo "[SETUP] Installing neuraloperator ..."
python -m pip install --quiet neuraloperator

rm -rf "$TMPDIR"
unset TMPDIR

# --- CUDA smoke test (HARD-FAIL on missing CUDA) ---
echo ""
echo "============================================"
echo "[SMOKE] CUDA availability + matmul test"
echo "============================================"
python - <<'PYEOF'
import sys, time
import torch
print(f"torch={torch.__version__}  cuda_build={torch.version.cuda}")
ok = torch.cuda.is_available()
print(f"torch.cuda.is_available() = {ok}")
if not ok:
    print("[FATAL] CUDA not available — aborting before wasting GPU hours.")
    sys.exit(2)
dev = torch.device("cuda")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"device 0: {name}  capability={cap}")
# Real GPU work
x = torch.randn(4096, 4096, device=dev)
torch.cuda.synchronize()
t0 = time.time()
y = x @ x
torch.cuda.synchronize()
dt = time.time() - t0
print(f"4096x4096 matmul on GPU: {dt*1000:.1f} ms  sum={y.sum().item():.3e}")
print("[SMOKE] CUDA OK")
PYEOF

# --- End-to-end FNO smoke (5 epochs on the v2 dataset) ---
# Build a throwaway smoke config that reuses the v2 dataset but
# writes its outputs to a sandbox dir, so we don't clobber the
# real v2 model.pt / history.json / metrics.json.
SMOKE_CFG="$WORKDIR/.smoke_fno_cfg.yaml"
python - <<PYEOF
import yaml, pathlib
src = pathlib.Path("configs/fno_zerilli_l2_v2.yaml")
cfg = yaml.safe_load(src.read_text())
cfg.setdefault("experiment", {})["name"] = "fno_smoke_gpu"
cfg["training"]["out_dir"] = "outputs/fno/_smoke_gpu"
# data_path stays pointing at the existing v2 dataset.npz
pathlib.Path("$SMOKE_CFG").write_text(yaml.safe_dump(cfg))
print("[SMOKE] wrote $SMOKE_CFG with out_dir=", cfg["training"]["out_dir"])
PYEOF

echo ""
echo "============================================"
echo "[SMOKE] FNO pipeline --quick on CUDA"
echo "============================================"
python scripts/train_fno.py --config "$SMOKE_CFG" --quick

# Clean up sandbox so it doesn't get confused with real results
rm -f "$SMOKE_CFG"
rm -rf "outputs/fno/_smoke_gpu"

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
