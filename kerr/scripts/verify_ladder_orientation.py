"""Verify the assembled tensors for the new ladder have the right orientation.

Loads the smoke corpus (k2=201, k4=101), runs the SAME pipeline assemble() the
training uses, and checks:
  - stored grids: psi_k2 has N=201, psi_k4 has N=101, fine 801;
  - prior baseline = up4 = upsample(k4=101) (the field the hybrid corrects);
  - Richardson target Y = (4*up2 - up4)/3 - up4 reconstructs psi_R = up4 + s*Y
    == (4*up2 - up4)/3 with up2=upsample(k2=201) (orientation NOT inverted);
  - the Richardson field is CLOSER to fine than the prior (teacher carries a
    field gain) -- the sanity check that k2 is the finer grid, not k4.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kerr.src.hybrid_data_pipe import load_split, build_upsample_matrix, assemble

PATH = "/tmp/l4n101_smoke/dataset_train.npz"


def rel_l2(p, t):
    return float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-30)) * 100.0


def main():
    sp = load_split(PATH, "train")
    print("stored grid sizes:")
    print(f"  fine sigma N = {sp['sigma_fine'].size}")
    print(f"  k2   sigma N = {sp['sigma_k2'].size}   (Richardson mid; want 201)")
    print(f"  k4   sigma N = {sp['sigma_k4'].size}   (prior; want 101)")
    assert sp["sigma_k2"].size == 201 and sp["sigma_k4"].size == 101 and sp["sigma_fine"].size == 801

    W_k4 = build_upsample_matrix(sp["sigma_k4"], sp["sigma_fine"])
    W_k2 = build_upsample_matrix(sp["sigma_k2"], sp["sigma_fine"])
    asm = assemble(sp, W_k4, W_k2, target_mode="richardson", return_eval=True)

    scale = asm["scale"]; Y = asm["Y"]
    up4_re = asm["up4_re"]; up4_im = asm["up4_im"]
    fine_re = asm["fine_re"]; fine_im = asm["fine_im"]
    sB = scale[:, None, None]

    # reconstruct the Richardson teacher from the normalised target
    rich_re = up4_re + sB * Y[:, 0]
    rich_im = up4_im + sB * Y[:, 1]

    # independent Richardson from raw grids: (4*up2 - up4)/3
    k4r = sp["psi_k4_re"]; k4i = sp["psi_k4_im"]
    k2r = sp["psi_k2_re"]; k2i = sp["psi_k2_im"]
    up4r = np.einsum("fc,ntc->ntf", W_k4, k4r.astype(np.float64))
    up4i = np.einsum("fc,ntc->ntf", W_k4, k4i.astype(np.float64))
    up2r = np.einsum("fc,ntc->ntf", W_k2, k2r.astype(np.float64))
    up2i = np.einsum("fc,ntc->ntf", W_k2, k2i.astype(np.float64))
    richr_raw = (4.0 * up2r - up4r) / 3.0
    richi_raw = (4.0 * up2i - up4i) / 3.0

    # 1) prior baseline matches upsample(k4=101)
    d_prior = rel_l2(up4_re, up4r) + rel_l2(up4_im, up4i)
    print(f"\n[1] prior == upsample(k4=101): rel-L2 diff = {d_prior:.2e}% (want ~0)")
    assert d_prior < 1e-3

    # 2) reconstructed Richardson matches (4*up2-up4)/3 (orientation correct)
    d_rich = rel_l2(rich_re, richr_raw) + rel_l2(rich_im, richi_raw)
    print(f"[2] psi_R == (4*up2-up4)/3 (up2=k2=201): rel-L2 diff = {d_rich:.2e}% (want ~0)")
    assert d_rich < 1e-2

    # 3) Richardson is closer to fine than the prior (k2 is the FINER grid)
    for i in range(up4_re.shape[0]):
        fp = rel_l2(up4_re[i], fine_re[i]) + rel_l2(up4_im[i], fine_im[i])
        fr = rel_l2(rich_re[i], fine_re[i]) + rel_l2(rich_im[i], fine_im[i])
        a = float(sp["P"][i, 0])
        flag = "OK" if fr < fp else "*** INVERTED ***"
        print(f"[3] sample a={a:.3f}: prior_F={fp:6.2f}%  rich_F={fr:6.2f}%  -> {flag}")
        assert fr < fp, "Richardson worse than prior => grids inverted!"

    print("\nORIENTATION VERIFIED: prior=up(k4=101), Richardson=(4*up(k2=201)-up(k4=101))/3, "
          "teacher beats prior on field.")


if __name__ == "__main__":
    main()
