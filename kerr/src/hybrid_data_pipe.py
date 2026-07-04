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
    W_prior: "np.ndarray | None" = None,
    prior_key: str = "k8",
    richardson_p: int = 2,
    norm_mode: str = "scalar",
    envelope_floor: float = 1.0e-5,
    residual_gate: bool = False,
    gate_floor: float = 0.02,
) -> Dict[str, np.ndarray]:
    """Assemble normalised training tensors for one split (fast matrix path).

    Returns a dict with:
        X      (N, 4, Ntau, Nf) float32   normalised input channels
        Y      (N, 2, Ntau, Nf) float32   normalised Richardson (or fine) target
        scale  (N,) or (N,Ntau) float32   per-sample (scalar) / per-time (envelope) norm
    and, if ``return_eval`` (kept for the test split only -- memory):
        up4_re, up4_im   (N, Ntau, Nf) float32   physical prior (eval baseline)
        fine_re, fine_im (N, Ntau, Nf) float32   physical fine FD (eval target)

    The model **prior** (input / scale / eval baseline) is the quintic-upsampled
    coarse solve. By default (``W_prior is None``) the prior is the ``k4`` rung,
    so prior and the coarse Richardson rung coincide (the original coupled
    pipe). Passing ``W_prior`` + ``prior_key`` **decouples** the input grid from
    the Richardson rungs: the prior is upsampled from a separate (coarser,
    headroom-bearing) grid -- e.g. ``k8`` (N=101) -- while the label is built
    from the finer ``k2``/``k4`` pair. The Richardson order is set by
    ``richardson_p`` (2 -> (4 u2 - u4)/3; 4 -> (16 u2 - u4)/15); ``k2`` is the
    finer rung. ``norm_mode='envelope'`` swaps the single per-sample scale for a
    per-time envelope ``w(tau)=max(rms_sigma|prior|, envelope_floor*peak)`` (the
    tau fix: the low-amplitude late ringdown becomes O(1) so the FNO/loss can
    see it); ``scale`` is then ``(N, Ntau)`` and the hybrid is reconstructed as
    ``prior + w(tau)*FNO``. ``chunk`` bounds peak memory by streaming over
    samples. With
    ``target_mode='supervised'`` the target is ``fine - prior`` (fine label, A/B
    baseline only); the fine field is otherwise never in the target.
    """
    if target_mode not in ("richardson", "supervised"):
        raise ValueError(
            f"target_mode must be 'richardson' or 'supervised', got {target_mode!r}")
    if richardson_p not in (2, 4):
        raise ValueError(f"richardson_p must be 2 or 4, got {richardson_p}")
    if norm_mode not in ("scalar", "envelope"):
        raise ValueError(f"norm_mode must be 'scalar' or 'envelope', got {norm_mode!r}")

    k4_re = split["psi_k4_re"]; k4_im = split["psi_k4_im"]
    k2_re = split["psi_k2_re"]; k2_im = split["psi_k2_im"]
    fn_re = split["psi_fine_re"]; fn_im = split["psi_fine_im"]
    aM = np.asarray(split["P"][:, 0], dtype=np.float32)

    if W_prior is not None:
        pr_re = split[f"psi_{prior_key}_re"]; pr_im = split[f"psi_{prior_key}_im"]
    else:
        pr_re = pr_im = None

    N, Ntau, _ = k4_re.shape
    Nf = W_k4.shape[0]
    coef = (2.0 ** richardson_p) / (2.0 ** richardson_p - 1.0)

    X = np.empty((N, HYBRID_IN_CHANNELS, Ntau, Nf), dtype=np.float32)
    Y = np.empty((N, HYBRID_OUT_CHANNELS, Ntau, Nf), dtype=np.float32)
    # scalar norm -> one scale per sample; envelope norm -> a scale per (sample, tau)
    scale = np.empty((N, Ntau) if norm_mode == "envelope" else N, dtype=np.float32)
    eval_out: Dict[str, np.ndarray] = {}
    if return_eval:
        for key in ("up4_re", "up4_im", "fine_re", "fine_im"):
            eval_out[key] = np.empty((N, Ntau, Nf), dtype=np.float32)

    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        up4_re = _apply_W(W_k4, k4_re[lo:hi]); up4_im = _apply_W(W_k4, k4_im[lo:hi])
        up2_re = _apply_W(W_k2, k2_re[lo:hi]); up2_im = _apply_W(W_k2, k2_im[lo:hi])

        if W_prior is not None:
            prior_re = _apply_W(W_prior, pr_re[lo:hi])
            prior_im = _apply_W(W_prior, pr_im[lo:hi])
        else:
            prior_re = up4_re; prior_im = up4_im

        if norm_mode == "envelope":
            # Per-time envelope w(tau) = rms_sigma|prior(tau,.)|, floored at
            # envelope_floor * peak. Dividing by it lifts the low-amplitude late
            # ringdown to O(1) so the FNO/loss can see it (the tau fix), while
            # the floor stops the deep tail (noise floor) from being amplified.
            w_raw = np.sqrt(np.mean(prior_re ** 2 + prior_im ** 2, axis=2)).astype(np.float32)
            peak = np.maximum(w_raw.max(axis=1), 1e-30)               # (n,)
            w = np.maximum(w_raw, envelope_floor * peak[:, None])     # (n, Ntau)
            sB = np.maximum(w, 1e-30)[:, :, None]                     # field scale (n, Ntau, 1)
            s0 = sB[:, 0:1, :]                                        # psi0 scale = w(tau=0)
            scale[lo:hi] = sB[:, :, 0]
        else:
            s = np.sqrt(np.mean(prior_re ** 2 + prior_im ** 2, axis=(1, 2))).astype(np.float32)
            s = np.maximum(s, 1e-30)
            sB = s[:, None, None]                                     # (n, 1, 1)
            s0 = sB
            scale[lo:hi] = s

        if target_mode == "richardson":
            # p-th order Richardson extrapolant minus prior. k2 is the finer
            # rung: (2^p u2 - u4)/(2^p-1) = coef*u2 - (coef-1)*u4. With the
            # coupled prior (== up4) this reduces to coef*(u2 - u4) exactly.
            rich_re = coef * up2_re - (coef - 1.0) * up4_re
            rich_im = coef * up2_im - (coef - 1.0) * up4_im
            y_re = rich_re - prior_re
            y_im = rich_im - prior_im
        else:
            y_re = fn_re[lo:hi] - prior_re
            y_im = fn_im[lo:hi] - prior_im

        if residual_gate:
            # Amplitude gate (variant B): keep the FNO correction only where the
            # prior has appreciable amplitude and taper it to 0 in the low-amplitude
            # tail, so the residual cannot corrupt the late ringdown that sets tau
            # (the hybrid reverts to the clean prior there). g(tau) is built from
            # the prior envelope only -- no fine, no Leaver, generalises.
            w_g = np.sqrt(np.mean(prior_re ** 2 + prior_im ** 2, axis=2))     # (n, Ntau)
            pk_g = np.maximum(w_g.max(axis=1, keepdims=True), 1e-30)          # (n, 1)
            g = np.clip((w_g / pk_g) / max(gate_floor, 1e-12), 0.0, 1.0)[:, :, None]
            y_re = y_re * g
            y_im = y_im * g

        X[lo:hi, 0] = prior_re / sB
        X[lo:hi, 1] = prior_im / sB
        X[lo:hi, 2] = aM[lo:hi, None, None]
        X[lo:hi, 3] = prior_re[:, 0:1, :] / s0          # psi0.real broadcast over tau
        Y[lo:hi, 0] = y_re / sB
        Y[lo:hi, 1] = y_im / sB

        if return_eval:
            eval_out["up4_re"][lo:hi] = prior_re
            eval_out["up4_im"][lo:hi] = prior_im
            eval_out["fine_re"][lo:hi] = fn_re[lo:hi]
            eval_out["fine_im"][lo:hi] = fn_im[lo:hi]

    out = {"X": X, "Y": Y, "scale": scale}
    out.update(eval_out)
    return out


def load_split(path: str, split: str) -> Dict[str, np.ndarray]:
    """Load one Kerr corpus split npz into a flat dict with split-prefix stripped.

    Returns the Re/Im field stacks, sigma axes, tau, params ``P`` (a/M, r0, w),
    the Leaver reference ``qnm``, and the scri indices, as plain arrays. Every
    grid present in the npz is loaded (``fine``, ``k2``, ``k4``, and -- for the
    decoupled corpus -- the coarse prior ``k8``), so a 4-grid corpus round-trips
    without changing this loader.
    """
    d = np.load(path, allow_pickle=True)
    out: Dict[str, np.ndarray] = {
        "tau": d["tau"],
        "P": d[f"{split}_P"], "qnm": d[f"{split}_qnm"],
    }
    grids = [k[len("sigma_"):] for k in d.files if k.startswith("sigma_")]
    for grid in grids:
        out[f"sigma_{grid}"] = d[f"sigma_{grid}"]
        out[f"scri_idx_{grid}"] = d[f"scri_idx_{grid}"]
        for part in ("re", "im"):
            key = f"{split}_psi_{grid}_{part}"
            if key in d.files:
                out[f"psi_{grid}_{part}"] = d[key]
    return out
