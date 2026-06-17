"""Kerr hybrid data pipe: spatial upsampling + label-free Richardson target.

Kerr analogue of the Schwarzschild ``src/hybrid_data_pipe.py``, with the two
structural differences of the Phase C corpus (``kerr/src/kerr_dataset.py``):

* the field ``psi(tau, sigma)`` is **complex** -> Re/Im are carried as two
  channels and the (linear) Richardson combination is applied to each part
  independently;
* the coarse and fine grids share a **single canonical tau-axis**, so the
  coarse->fine upsampling is purely **spatial** (a 1-D quintic spline along
  ``sigma``, vectorised over tau) -- simpler and more accurate than the SW
  pipe's 2-D tensor-product interpolation.

Deployment prior is the single cheapest coarse solve (``k=4``, ``N=201``),
quintic-upsampled. The ``k=2`` solve is used **only** to build the label-free
Richardson target and is never needed at inference:

    psi_R = (4 * up2 - up4) / 3        (a-priori order p=2 Richardson)
    Y     = psi_R - up4               (label-free training target)

For a 2nd-order-convergent scheme the leading spatial truncation error is
O(h^2); with the k2/k4 grids in exact 2:1 spatial refinement this combination
cancels it to O(h^4) without ever touching the fine FD field. The fine field is
returned as an **eval-only** metric.

Per-sample normalisation. Kerr fields span O(1e2) and vary strongly with
(a, r0, w); an absolute MSE would be dominated by the highest-amplitude black
holes. Each sample is divided by ``s = rms(|up4|)`` (a function of the prior
alone, available at inference), so the network sees an O(1) target for every BH.
The physical hybrid field is reconstructed as ``psi_hyb = up4 + s * FNO``.

Channel layout (in_channels = 4):
    ch 0 : upsample(psi_coarse_k4).real / s     -- prior, real part
    ch 1 : upsample(psi_coarse_k4).imag / s     -- prior, imag part
    ch 2 : a/M  (broadcast; NOT scaled)         -- the spin (changes operator)
    ch 3 : psi0.real / s  (broadcast over tau)  -- the initial pulse at tau=0

Output (out_channels = 2):
    [delta.real, delta.imag] / s -- normalised additive correction; the hybrid
    prediction is psi_hyb = up4 + s * FNO.

Because the sigma nodes are fixed for the whole corpus and nest exactly
(``sigma_fine[::2] == sigma_k2``, ``[::4] == sigma_k4``), the quintic spline is a
fixed linear map applied as a precomputed dense matrix (``up = W @ coarse``),
turning the per-sample spline (~2.5 s) into one batched matmul per split.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import make_interp_spline

HYBRID_IN_CHANNELS = 4
HYBRID_OUT_CHANNELS = 2


def _ascending(sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sigma_sorted_ascending, order) so spline x is strictly increasing."""
    order = np.argsort(sigma)
    return sigma[order], order


def upsample_sigma(
    field: np.ndarray,
    sigma_c: np.ndarray,
    sigma_f: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Spatial-only spline upsample of one ``(Ntau, Nc)`` field onto ``sigma_f``.

    ``tau`` is shared between coarse and fine grids, so only the ``sigma`` axis
    is interpolated. A degree-``k`` (default quintic) B-spline is fit along
    ``sigma`` and evaluated at the fine nodes, vectorised over all ``tau`` rows
    at once (``axis=1``). Returns ``(Ntau, Nf)``.
    """
    sc, order = _ascending(np.asarray(sigma_c, dtype=np.float64))
    fld = field[:, order]
    spl = make_interp_spline(sc, fld, k=k, axis=1)
    return spl(np.asarray(sigma_f, dtype=np.float64))


def build_upsample_matrix(
    sigma_c: np.ndarray, sigma_f: np.ndarray, k: int = 5
) -> np.ndarray:
    """Dense matrix ``W`` (Nf, Nc) with ``W @ coarse == quintic_spline(coarse)``.

    Both axes must be strictly ascending (the Phase C sigma grids already are).
    Built by interpolating each coarse unit vector and evaluating on the fine
    nodes, so ``W`` is exactly the linear map ``make_interp_spline`` applies --
    validated against the direct spline in the smoke test.
    """
    sc = np.asarray(sigma_c, dtype=np.float64)
    sf = np.asarray(sigma_f, dtype=np.float64)
    if np.any(np.diff(sc) <= 0) or np.any(np.diff(sf) <= 0):
        raise ValueError("sigma grids must be strictly ascending")
    Nc = sc.size
    W = np.empty((sf.size, Nc), dtype=np.float64)
    e = np.zeros(Nc, dtype=np.float64)
    for j in range(Nc):
        e[j] = 1.0
        W[:, j] = make_interp_spline(sc, e, k=k)(sf)
        e[j] = 0.0
    return W


def _apply_W(W: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    """Apply upsample matrix to ``(n, Ntau, Nc)`` -> ``(n, Ntau, Nf)`` (float32)."""
    return np.einsum("fc,ntc->ntf", W, coarse.astype(np.float64),
                     optimize=True).astype(np.float32)


def _upsample_complex(
    re: np.ndarray, im: np.ndarray,
    sigma_c: np.ndarray, sigma_f: np.ndarray, k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quintic-upsample a complex field (Re/Im) along sigma. Returns (up_re, up_im)."""
    return (
        upsample_sigma(re, sigma_c, sigma_f, k).astype(np.float32),
        upsample_sigma(im, sigma_c, sigma_f, k).astype(np.float32),
    )


def assemble(
    split: Dict[str, np.ndarray],
    W_k4: np.ndarray,
    W_k2: np.ndarray,
    target_mode: str = "richardson",
    return_eval: bool = False,
    chunk: int = 64,
) -> Dict[str, np.ndarray]:
    """Assemble normalised training tensors for one split (fast matrix path).

    Returns a dict with:
        X      (N, 4, Ntau, Nf) float32   normalised input channels
        Y      (N, 2, Ntau, Nf) float32   normalised Richardson (or fine) target
        scale  (N,)             float32   per-sample s = rms(|up4|)
    and, if ``return_eval`` (kept for the test split only -- memory):
        up4_re, up4_im   (N, Ntau, Nf) float32   physical prior (eval baseline)
        fine_re, fine_im (N, Ntau, Nf) float32   physical fine FD (eval target)

    The prior is the quintic-upsampled k4 solve; the Richardson target uses
    k2 + k4 only (label-free). ``chunk`` bounds peak memory by streaming over
    samples. With ``target_mode='supervised'`` the target is ``fine - up4`` (fine
    label, A/B baseline only); the fine field is otherwise never in the target.
    """
    if target_mode not in ("richardson", "supervised"):
        raise ValueError(
            f"target_mode must be 'richardson' or 'supervised', got {target_mode!r}")

    k4_re = split["psi_k4_re"]; k4_im = split["psi_k4_im"]
    k2_re = split["psi_k2_re"]; k2_im = split["psi_k2_im"]
    fn_re = split["psi_fine_re"]; fn_im = split["psi_fine_im"]
    aM = np.asarray(split["P"][:, 0], dtype=np.float32)

    N, Ntau, _ = k4_re.shape
    Nf = W_k4.shape[0]

    X = np.empty((N, HYBRID_IN_CHANNELS, Ntau, Nf), dtype=np.float32)
    Y = np.empty((N, HYBRID_OUT_CHANNELS, Ntau, Nf), dtype=np.float32)
    scale = np.empty(N, dtype=np.float32)
    eval_out: Dict[str, np.ndarray] = {}
    if return_eval:
        for key in ("up4_re", "up4_im", "fine_re", "fine_im"):
            eval_out[key] = np.empty((N, Ntau, Nf), dtype=np.float32)

    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        up4_re = _apply_W(W_k4, k4_re[lo:hi]); up4_im = _apply_W(W_k4, k4_im[lo:hi])
        up2_re = _apply_W(W_k2, k2_re[lo:hi]); up2_im = _apply_W(W_k2, k2_im[lo:hi])

        s = np.sqrt(np.mean(up4_re ** 2 + up4_im ** 2, axis=(1, 2))).astype(np.float32)
        s = np.maximum(s, 1e-30)
        sB = s[:, None, None]

        if target_mode == "richardson":
            y_re = (4.0 / 3.0) * (up2_re - up4_re)
            y_im = (4.0 / 3.0) * (up2_im - up4_im)
        else:
            y_re = fn_re[lo:hi] - up4_re
            y_im = fn_im[lo:hi] - up4_im

        X[lo:hi, 0] = up4_re / sB
        X[lo:hi, 1] = up4_im / sB
        X[lo:hi, 2] = aM[lo:hi, None, None]
        X[lo:hi, 3] = up4_re[:, 0:1, :] / sB           # psi0.real broadcast over tau
        Y[lo:hi, 0] = y_re / sB
        Y[lo:hi, 1] = y_im / sB
        scale[lo:hi] = s

        if return_eval:
            eval_out["up4_re"][lo:hi] = up4_re
            eval_out["up4_im"][lo:hi] = up4_im
            eval_out["fine_re"][lo:hi] = fn_re[lo:hi]
            eval_out["fine_im"][lo:hi] = fn_im[lo:hi]

    out = {"X": X, "Y": Y, "scale": scale}
    out.update(eval_out)
    return out


def load_split(path: str, split: str) -> Dict[str, np.ndarray]:
    """Load one Kerr corpus split npz into a flat dict with split-prefix stripped.

    Returns the Re/Im field stacks, sigma axes, tau, params ``P`` (a/M, r0, w),
    the Leaver reference ``qnm``, and the scri indices, as plain arrays.
    """
    d = np.load(path, allow_pickle=True)
    out: Dict[str, np.ndarray] = {
        "tau": d["tau"],
        "sigma_fine": d["sigma_fine"], "sigma_k2": d["sigma_k2"],
        "sigma_k4": d["sigma_k4"],
        "scri_idx_fine": d["scri_idx_fine"], "scri_idx_k2": d["scri_idx_k2"],
        "scri_idx_k4": d["scri_idx_k4"],
        "P": d[f"{split}_P"], "qnm": d[f"{split}_qnm"],
    }
    for grid in ("fine", "k2", "k4"):
        for part in ("re", "im"):
            out[f"psi_{grid}_{part}"] = d[f"{split}_psi_{grid}_{part}"]
    return out
