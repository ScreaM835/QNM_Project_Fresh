"""Generate the two Zerilli Colab notebooks (faithful reproduction + enhanced
forward) with identical end-to-end structure. Run once, then delete.

    python scripts/_gen_repro_notebooks.py
"""
import json
import os

# ---------------------------------------------------------------- shared cells
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text}

MOUNT = """\
# Mount Google Drive (persists across disconnects), then set the run config.
from google.colab import drive
drive.mount('/content/drive')
import os
# ==== the ONLY block that differs between the two notebooks ====
NAME     = {name!r}
CONFIG   = {config!r}
DRIVE    = {drive!r}
ESPRIT_K = 6          # ESPRIT model order for M3 on PINN fields
# ===============================================================
os.makedirs(DRIVE, exist_ok=True)
print('run:', NAME, '| config:', CONFIG, '| drive:', DRIVE)
!ls -lh {{DRIVE}} 2>/dev/null || echo '(empty - first run)'"""

SANITY = """\
# Sanity: which accelerator did Colab give us?
!nvidia-smi -L 2>/dev/null || echo 'no GPU (CPU-only; the run will be slow)'
import torch
print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"""

CLONE = """\
# Clone the repo (with the standardised plotting code) and install DeepXDE.
%cd /content
!rm -rf QNM_Project_Fresh
!git clone --depth 1 https://github.com/ScreaM835/QNM_Project_Fresh.git
REPO = '/content/QNM_Project_Fresh'
%cd {REPO}
!pip -q install "deepxde==1.15.0" tqdm
os.environ['DDE_BACKEND'] = 'pytorch'
os.environ['PYTHONIOENCODING'] = 'utf-8'
OUT       = f'{REPO}/outputs/pinn/{NAME}'
DRIVE_OUT = f'{DRIVE}/{NAME}'
print('repo ready | OUT =', OUT)"""

RUN = """\
# FD data + PINN train + field eval + standardised field figures.
# Resumable: restores any prior run from Drive; skips training if already done.
# NOTE: run_pinn.py often exits with code 1 at the very end -- that is a benign
# DeepXDE/torch teardown artifact; the outputs are written normally.
import os, shutil
%cd {REPO}
if os.path.isdir(DRIVE_OUT):
    os.makedirs(OUT, exist_ok=True)
    shutil.copytree(DRIVE_OUT, OUT, dirs_exist_ok=True)
    print('[run] restored prior run dir from Drive')
done = os.path.exists(f'{OUT}/{NAME}_pinn.npz') and os.path.exists(f'{OUT}/metrics.json')
if done:
    print('[run] completed run found on Drive -> skipping training')
else:
    resume = '--resume' if (os.path.isdir(f'{OUT}/checkpoints') and os.listdir(f'{OUT}/checkpoints')) else ''
    print('[run] training (resume flag =', repr(resume), ')')
    !DDE_BACKEND=pytorch PYTHONIOENCODING=utf-8 python scripts/run_pinn.py --config {CONFIG} {resume} --checkpoint-every 500
# always (re)build the standardised field figures from the saved fields, so the
# snapshot / abs-diff / heatmap scales match the Regge-Wheeler + hybrid figures
!PYTHONIOENCODING=utf-8 python scripts/replot_repro_from_npz.py --run {NAME} --config {CONFIG}
os.makedirs(DRIVE_OUT, exist_ok=True)
shutil.copytree(OUT, DRIVE_OUT, dirs_exist_ok=True)
print('[run] outputs + standardised figures saved -> Drive:', DRIVE_OUT)"""

QNM = """\
# Full M1-M5 QNM suite for BOTH the PINN field and the FD reference.
# (The reproduction paper reports only M1/M2, but the whole suite is run so the
#  extraction mechanism is identical to every other model.)
import os, shutil, glob
%cd {REPO}
# extract_qnm --source fd reads outputs/fd/<name>_fd.npz
os.makedirs('outputs/fd', exist_ok=True)
shutil.copy(f'outputs/pinn/{NAME}/{NAME}_fd.npz', f'outputs/fd/{NAME}_fd.npz')
!PYTHONIOENCODING=utf-8 python scripts/extract_qnm.py --config {CONFIG} --source pinn --esprit --esprit-K {ESPRIT_K} --two-mode --two-mode-2d
!PYTHONIOENCODING=utf-8 python scripts/extract_qnm.py --config {CONFIG} --source fd   --esprit --esprit-K {ESPRIT_K} --two-mode --two-mode-2d
DRIVE_Q = f'{DRIVE}/{NAME}/qnm'
os.makedirs(DRIVE_Q, exist_ok=True)
for f in glob.glob(f'{REPO}/outputs/qnm/{NAME}/*'):
    shutil.copy(f, DRIVE_Q)
print('[qnm] M1-M5 for PINN + FD saved -> Drive')"""

CURVEFIT = """\
# Ringdown curve-fitting overlay (log|Phi| with the M1/M2 damped-cosine fits),
# in the shared FD-blue / PINN-orange convention.
import os, shutil
%cd {REPO}
!PYTHONIOENCODING=utf-8 python scripts/plot_curve_fitting.py --config {CONFIG}
for src in [f'outputs/pinn/{NAME}/curve_fitting.png', f'outputs/qnm/{NAME}/curve_fitting.png']:
    if os.path.exists(src):
        shutil.copy(src, f'{DRIVE_OUT}/')
print('[plot] curve_fitting.png saved -> Drive')"""

REPORT = """\
# *** FINAL REPORT: field metrics + full M1-M5 QNM for PINN and FD ***
import json, os
OUTP = f'outputs/pinn/{NAME}'; QOUT = f'outputs/qnm/{NAME}'
m = json.load(open(f'{OUTP}/metrics.json'))
print('=' * 66)
print('FINAL REPORT  --  ' + NAME)
print('=' * 66)
print('FIELD:  RMSD = {:.4e}   MAD = {:.4e}   RL2 = {:.4f}  ({:.2f}%)'.format(
      m['RMSD'], m['MAD'], m['RL2'], m['RL2'] * 100))
methods = [('M1', 'method1'), ('M2', 'method2'), ('M3', 'method3_esprit'),
           ('M4', 'method4_two_mode'), ('M5', 'method5_2d_scan')]
for src in ['pinn', 'fd']:
    print('\\n' + src.upper() + ' QNM  (Mw = M*omega, tau = tau/M; errors vs Leaver 0.3737 / 11.241):')
    print('  {:<4} {:>10} {:>9} {:>10} {:>9}'.format('meth', 'Mw', 'w_err%', 'tau', 'tau_err%'))
    for lbl, fn in methods:
        p = f'{QOUT}/{src}_{fn}.json'
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        print('  {:<4} {:>10.4f} {:>8.3f}% {:>10.4f} {:>8.3f}%'.format(
              lbl, d.get('omega_dim', float('nan')), d.get('omega_pct_err', float('nan')),
              d.get('tau_dim', float('nan')), d.get('tau_pct_err', float('nan'))))
print('\\nStandardised figures in ' + OUTP + ':')
print('  snapshots.png, abs_diff_snapshots.png, error_heatmap.png,')
print('  loss.png, ringdown_overlay.png, curve_fitting.png')"""

ZIP = """\
# Zip the run (figures + metrics + QNM JSONs + fields) and download to the laptop.
import os
%cd {REPO}
ZIP = f'/content/{NAME}_run.zip'
!rm -f {ZIP}
!cd {REPO} && zip -r {ZIP} outputs/pinn/{NAME} outputs/qnm/{NAME} -x '*checkpoints*'
from google.colab import files
files.download(ZIP)
print('downloaded', ZIP)"""

INTRO = """\
# {title}

{desc}

**One session, end to end:** FD data -> PINN train -> field eval -> QNM (M1-M5)
-> standardised plots, all saved to Google Drive (so a disconnect never loses
work) and zipped for the laptop at the end. Re-running from the top restores any
finished stage from Drive and skips the retrain.

**No separate eval needed** -- the final report cell prints the field metrics
(RMSD, MAD, RL2) and the full M1-M5 QNM table for both the PINN and the FD
reference.

**Plots use the shared house scale** so they line up with the Regge-Wheeler
reproduction and the hybrid figures:
- absolute-difference snapshots: y-axis fixed to `[0, 0.02]`
- field snapshots: y-range fixed to `[-0.85, 0.6]`
- pointwise-error heatmap: `magma_r` log colour fixed to `[1e-6, 2.1e-2]`

Drive layout: `MyDrive/{drive_tail}/{name}/`
"""


def build(name, config, drive, title, desc):
    cells = [
        md(INTRO.format(title=title, desc=desc, drive_tail=drive.split('/')[-1], name=name)),
        code(MOUNT.format(name=name, config=config, drive=drive)),
        code(SANITY),
        code(CLONE),
        code(RUN),
        code(QNM),
        code(CURVEFIT),
        code(REPORT),
        code(ZIP),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NB = [
    ("repro_zerilli_colab.ipynb", "zerilli_l2_paper",
     "configs/zerilli_l2_paper.yaml", "/content/drive/MyDrive/qnm_repro_zerilli",
     "Faithful reproduction (Zerilli) -- end-to-end Colab run",
     "Reproduces the Patel et al. Zerilli baseline at their exact settings: "
     "uniform collocation resampling, Adam 10k + L-BFGS 15k, no residual-greedy "
     "sampling and no extra iterations. Expect ~1-2 h on a Colab GPU."),
    ("forward_zerilli_colab.ipynb", "zerilli_l2_greedy_f03_lbfgs30k",
     "configs/zerilli_l2_greedy_f03_lbfgs30k.yaml", "/content/drive/MyDrive/qnm_forward_zerilli",
     "Enhanced forward PINN (Zerilli) -- end-to-end Colab run",
     "The enhanced forward PINN on Zerilli: residual-greedy collocation sampling "
     "plus a doubled (30k) L-BFGS budget on top of the reproduction settings. "
     "Expect ~2-3 h on a Colab GPU."),
]

os.makedirs("notebooks", exist_ok=True)
for fname, name, config, drive, title, desc in NB:
    nb = build(name, config, drive, title, desc)
    path = os.path.join("notebooks", fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path, "(", len(nb["cells"]), "cells )")
