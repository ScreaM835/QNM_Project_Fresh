"""Coarse/fine COMPLEX-field dataset for the Phase C Kerr surrogate.

Mirrors ``src/hybrid_dataset.py`` (the Schwarzschild hybrid) but for the Kerr
Teukolsky field, which is **complex**. For each Sobol sample ``(a/M, r0, w)`` we
evolve the verified Phase B operator on three nested grids -- fine ``N=801`` and
coarse ``N=401`` (``k=2``) / ``N=201`` (``k=4``) -- and store the full complex
field ``psi(tau, sigma)`` on a **single canonical tau-axis shared by all grids**,
so the coarse->fine upsampling the data-pipe performs is purely *spatial*
(simpler and more accurate than the SW pipe's 2-D interpolation).

Design facts locked by the C.0 stability spot-check (see
``kerr/notes/phase_c_plan.md``):

* amplitude ``A`` is **not** a sweep axis -- the Teukolsky equation is linear, so
  ``psi(2A) = 2 psi(A)`` exactly; ``A`` is fixed at ``ID_AMP``;
* the box ``r0 in [8, 11]``, ``w in [1.0, 1.5]`` keeps >=5 grid points across the
  pulse even on the coarsest grid (``N=201``);
* the sigma-grids nest exactly (``801[::2] == 401``, ``801[::4] == 201``);
* ``T_STORE = 220`` reaches the B.9 plateau-scan end (``14 * tau_ref``, clamped to
  ``TAU_FINAL_MAX``) at every spin, so the stored field reproduces the B.9
  extraction window.

**No new physics path.** ``build_teukolsky_op`` / ``make_initial_pulse`` /
``rhs_teuk`` / ``ko_dissipation`` / ``rk4_step_state`` / ``state_from_psi`` /
``d1_central`` / ``kerr_qnm`` and the constants ``M, ELL, MM, SAFETY, SIGMA_KO,
ID_AMP`` are imported verbatim from the Phase B modules (via ``kv3_qnm``); only
the recording (full field on a fixed tau-grid) and the Sobol orchestration here
are new.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import qmc

# Re-use the EXACT Phase B physics + constants (byte-identical to B.9).
from kerr.scripts.kv3_qnm import (
    M,
    ELL,
    MM,
    SAFETY,
    SIGMA_KO,
    ID_AMP,
    build_teukolsky_op,
    make_initial_pulse,
    state_from_psi,
    rhs_teuk,
    ko_dissipation,
    cfl_dt,
    scri_index,
    d1_central,
)
from kerr.src.mol_rk4 import rk4_step_state
from kerr.src.qnm_kerr_reference import kerr_qnm
from kerr.src.fd_stencils import d1_4
from kerr.src.dissipation import ko_dissipation_6

# --- Sweep box (locked by the C.0 spot-check) --------------------------------
SPIN_RANGE: Tuple[float, float] = (0.0, 0.95)
R0_RANGE: Tuple[float, float] = (8.0, 11.0)
W_RANGE: Tuple[float, float] = (1.0, 1.5)

# --- Grids (nest exactly) -----------------------------------------------------
FINE_N: int = 801
COARSE_N: Dict[int, int] = {2: 401, 4: 201}

# --- Per-grid spatial discretisation order ------------------------------------
# Maps grid key ``k`` (1 = fine, else the coarse refinement factor) to the FD
# order used by ``evolve_full_field`` (2 = central + KO-4, 4 = central-4 + KO-6).
# Empty default => every grid uses order 2 (backward compatible). Mutated
# pre-fork by the build script (same pattern as ``COARSE_N``) to evolve, e.g.,
# the Richardson target rungs at order 4 while the headroom-bearing prior stays
# order 2. ``GRID_ORDER.get(k, 2)`` is read per grid inside the worker.
GRID_ORDER: Dict[int, int] = {}

# --- Stored space-time window -------------------------------------------------
# T_STORE reaches B.9's plateau-scan end (14 * tau_ref, clamped to 220) for every
# spin; DT_STORE is the canonical record cadence (validated against B.9 in the
# C.1 spot-check -- 0.25 M gives >=32 pts/period even at a/M=0.95).
T_STORE: float = 220.0
DT_STORE: float = 0.25


@dataclass
class Grids:
    """Canonical axes shared by every sample in a corpus."""

    tau: np.ndarray                     # (Ntau,)  shared by fine + all coarse
    sigma: Dict[int, np.ndarray]        # k -> (N_k,) sigma axis  (k=1 is fine)
    scri_idx: Dict[int, int]            # k -> sigma index of scri (the waveform)


# --- Train/val/test split (reproducible; reused by the C.3 eval) -------------
SPLIT_SIZES: Dict[str, int] = {"train": 1024, "val": 128, "test": 128}
SPLIT_ORDER: Tuple[str, ...] = ("train", "val", "test")


def sobol_params(n: int, seed: int = 0) -> np.ndarray:
    """Sobol quasi-random sweep over ``(a/M, r0, w)`` -> ``(n, 3)`` float64.

    Powers of two for ``n`` give the best Sobol balance; ``n`` need not be a
    power of two but a warning is emitted by ``scipy`` if it is not.
    """
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    pts = sampler.random(n=n)
    lo = np.array([SPIN_RANGE[0], R0_RANGE[0], W_RANGE[0]])
    hi = np.array([SPIN_RANGE[1], R0_RANGE[1], W_RANGE[1]])
    return lo + pts * (hi - lo)


def params_for_split(split: str, seed: int = 0, n: int | None = None,
                     sizes: Dict[str, int] | None = None) -> np.ndarray:
    """Deterministic, **disjoint** Sobol draw for one split.

    All splits are sliced from a single power-of-two Sobol draw (best balance),
    so ``train``/``val``/``test`` never overlap and are reproducible from
    ``seed`` regardless of which split is generated first. ``n`` overrides the
    split count (used by ``--smoke`` to take the first few train points).
    """
    sizes = sizes or SPLIT_SIZES
    total = sum(sizes[s] for s in SPLIT_ORDER)
    pow2 = int(2 ** np.ceil(np.log2(total)))     # exact-balance Sobol draw
    allp = sobol_params(pow2, seed)
    offset = sum(sizes[s] for s in SPLIT_ORDER[:SPLIT_ORDER.index(split)])
    count = sizes[split] if n is None else int(n)
    return allp[offset:offset + count]


def canonical_tau(t_store: float = T_STORE, dt_store: float = DT_STORE) -> np.ndarray:
    """The shared tau-axis ``[0, dt_store, ..., t_store]``."""
    n_rec = int(round(t_store / dt_store))
    return np.arange(n_rec + 1, dtype=np.float64) * dt_store


def _rhs_closure(op, d1=d1_central, ko=ko_dissipation):
    """The B.9 RHS: Teukolsky + Kreiss-Oliger on the two auxiliary fields.

    ``d1``/``ko`` select the spatial discretisation order: the defaults are the
    2nd-order central stencil + 4th-difference KO; pass ``d1_4`` + ``ko_dissipation_6``
    for the 4th-order variant.
    """

    def rhs_fn(s):
        dPsi, dU, dW = rhs_teuk(s, op, d1)
        dU = dU + ko(s[1], SIGMA_KO)
        dW = dW + ko(s[2], SIGMA_KO)
        return dPsi, dU, dW

    return rhs_fn


def evolve_full_field(
    a_over_M: float,
    N: int,
    r0: float,
    w: float,
    t_store: float = T_STORE,
    dt_store: float = DT_STORE,
    amp: float = ID_AMP,
    order: int = 2,
):
    """Evolve one sample and record the **full complex field** ``psi(tau, sigma)``.

    The record cadence is chosen so the stored times land *exactly* on the
    canonical grid ``[0, dt_store, ..., t_store]`` (no time interpolation): we
    pick ``record_every = ceil(dt_store / dt_cfl)`` and shrink ``dt`` to
    ``dt_store / record_every <= dt_cfl`` (still CFL-stable, exactly as B.9
    shrinks ``dt`` to make ``n_steps`` integer).

    ``order`` selects the spatial discretisation: 2 (default, 2nd-order central
    + 4th-difference KO) or 4 (4th-order central + 6th-difference KO). The CFL
    formula is unchanged: at safety 0.4 the step sits well inside the RK4
    stability limit for both stencils (the 4th-order symbol is only ~1.37x the
    2nd-order one).

    Returns ``(tau, psi, op, info)`` with ``psi`` of shape ``(Ntau, N)`` complex.
    """
    ref = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=0)
    omega_ref = complex(ref.M_omega_R, ref.M_omega_I)
    op = build_teukolsky_op(
        N=N, a_over_M=a_over_M, M=M, ell=ELL, m=MM,
        omega_ref=omega_ref, include_potential=True,
    )

    if order == 2:
        d1, ko = d1_central, ko_dissipation
    elif order == 4:
        d1, ko = d1_4, ko_dissipation_6
    else:
        raise ValueError(f"order must be 2 or 4, got {order}")

    dt_cfl = cfl_dt(op, safety=SAFETY)
    record_every = int(np.ceil(dt_store / dt_cfl))
    dt = dt_store / record_every                 # <= dt_cfl
    n_rec = int(round(t_store / dt_store))
    n_steps = n_rec * record_every

    state = state_from_psi(make_initial_pulse(amp, r0, w), op, d1)
    rhs_fn = _rhs_closure(op, d1=d1, ko=ko)

    rec = np.empty((n_rec + 1, N), dtype=np.complex128)
    rec[0] = state[0]
    j = 1
    for n in range(1, n_steps + 1):
        state = rk4_step_state(state, dt, rhs_fn)
        if n % record_every == 0:
            rec[j] = state[0]
            j += 1

    tau = np.arange(n_rec + 1, dtype=np.float64) * dt_store
    info = dict(
        N=N, dt=float(dt), n_steps=int(n_steps), record_every=int(record_every),
        finite=bool(np.all(np.isfinite(rec))),
        scri_idx=int(scri_index(op)),
    )
    return tau, rec, op, info


def reference_qnm(a_over_M: float) -> Tuple[float, float, float]:
    """``(M*omega_R, M*omega_I, tau/M)`` from the ``qnm`` package."""
    ref = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=0)
    return float(ref.M_omega_R), float(ref.M_omega_I), float(ref.tau_over_M)


def _evolve_one(task):
    """Evolve one sample on every grid; return float32 Re/Im (picklable).

    Module-level so it can run inside a ``multiprocessing`` worker. Returns the
    fields already split into float32 Re/Im (half the IPC of complex128 and no
    complex buffer kept in the parent).
    """
    i, a, r0, w, ks, t_store, dt_store = task
    grid_N = {1: FINE_N, **{k: COARSE_N[k] for k in ks}}
    re: Dict[int, np.ndarray] = {}
    im: Dict[int, np.ndarray] = {}
    sig: Dict[int, np.ndarray] = {}
    scri: Dict[int, int] = {}
    finite = True
    for k, Nk in grid_N.items():
        _t, rec, op, info = evolve_full_field(
            a, Nk, r0, w, t_store, dt_store, order=GRID_ORDER.get(k, 2))
        re[k] = rec.real.astype(np.float32)
        im[k] = rec.imag.astype(np.float32)
        sig[k] = op.sigma.astype(np.float64)
        scri[k] = int(info["scri_idx"])
        finite = finite and bool(info["finite"])
    qrow = np.asarray(reference_qnm(a), dtype=np.float64)
    return i, qrow, re, im, sig, scri, finite


def generate_split(
    params: Sequence[Sequence[float]],
    ks: Sequence[int] = (2, 4),
    t_store: float = T_STORE,
    dt_store: float = DT_STORE,
    progress_prefix: str = "",
    workers: int = 1,
) -> Tuple[Dict[str, np.ndarray], Grids]:
    """Run fine + every coarse grid for each ``(a/M, r0, w)`` sample.

    Returns ``(arrays, grids)`` where ``arrays`` holds float32 Re/Im fields keyed
    by grid (``psi_fine_re``, ``psi_k2_re``, ...), the parameter matrix ``P`` and
    the per-sample ``qnm`` reference matrix.

    The per-sample work is embarrassingly parallel (each sample builds its own
    operator, no shared state), so ``workers > 1`` fans it out over a process
    pool. The result is **bit-identical** to ``workers == 1`` because each
    evolution is deterministic and is written back by sample index; the pool
    only changes *when* each sample is computed, not *what* is stored. Fields are
    stored straight into the float32 output arrays (no all-samples complex128
    buffer), keeping peak memory at the size of the final corpus.
    """
    params = [tuple(float(x) for x in row) for row in params]
    n = len(params)
    tau = canonical_tau(t_store, dt_store)
    ntau = tau.size
    ks = tuple(int(k) for k in ks)
    grid_N = {1: FINE_N, **{k: COARSE_N[k] for k in ks}}

    psi_re: Dict[int, np.ndarray] = {}       # k -> (n, ntau, N_k) float32
    psi_im: Dict[int, np.ndarray] = {}
    for k, Nk in grid_N.items():
        psi_re[k] = np.empty((n, ntau, Nk), dtype=np.float32)
        psi_im[k] = np.empty((n, ntau, Nk), dtype=np.float32)
    sigma: Dict[int, np.ndarray] = {}
    scri: Dict[int, int] = {}
    P = np.array(params, dtype=np.float32)
    qref = np.empty((n, 3), dtype=np.float64)

    def _store(result) -> None:
        i, qrow, re, im, sig, scri_i, finite = result
        if not finite:
            a, r0, w = params[i]
            raise FloatingPointError(
                f"non-finite field: sample {i} (a={a:.4f}, r0={r0:.3f}, "
                f"w={w:.3f})")
        qref[i] = qrow
        for k in grid_N:
            psi_re[k][i] = re[k]
            psi_im[k][i] = im[k]
            if k not in sigma:
                sigma[k] = sig[k]
                scri[k] = scri_i[k]

    tasks = [(i, a, r0, w, ks, t_store, dt_store)
             for i, (a, r0, w) in enumerate(params)]

    done = 0
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(processes=int(workers)) as pool:
            for result in pool.imap_unordered(_evolve_one, tasks):
                _store(result)
                done += 1
                if progress_prefix:
                    i = result[0]
                    a, r0, w = params[i]
                    print(f"  {progress_prefix} [{done}/{n}] sample {i}: "
                          f"a/M={a:.4f} r0={r0:.3f} w={w:.3f}  "
                          f"Mw={qref[i, 0]:.5f}", flush=True)
    else:
        for task in tasks:
            result = _evolve_one(task)
            _store(result)
            done += 1
            if progress_prefix:
                i = result[0]
                a, r0, w = params[i]
                print(f"  {progress_prefix} {done}/{n}: "
                      f"a/M={a:.4f} r0={r0:.3f} w={w:.3f}  "
                      f"Mw={qref[i, 0]:.5f}", flush=True)

    arrays: Dict[str, np.ndarray] = {"P": P, "qnm": qref}
    for k in grid_N:
        tag = "fine" if k == 1 else f"k{k}"
        arrays[f"psi_{tag}_re"] = psi_re[k]
        arrays[f"psi_{tag}_im"] = psi_im[k]
    grids = Grids(tau=tau, sigma=sigma, scri_idx=scri)
    return arrays, grids


def save_dataset(path: str, split: str, arrays: Dict[str, np.ndarray],
                 grids: Grids, meta: Dict) -> None:
    """Write one split to ``path`` (single ``.npz``; grids stored alongside)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, np.ndarray] = {"tau": grids.tau}
    for k, sig in grids.sigma.items():
        tag = "fine" if k == 1 else f"k{k}"
        payload[f"sigma_{tag}"] = sig
        payload[f"scri_idx_{tag}"] = np.array(grids.scri_idx[k], dtype=np.int64)
    for name, arr in arrays.items():
        payload[f"{split}_{name}"] = arr
    payload["meta_json"] = np.array(json.dumps(meta), dtype=object)
    np.savez_compressed(path, **payload)


def load_dataset(path: str) -> Tuple[str, Dict[str, np.ndarray], Grids, Dict]:
    """Inverse of :func:`save_dataset`. Returns ``(split, arrays, grids, meta)``."""
    npz = np.load(path, allow_pickle=True)
    sigma: Dict[int, np.ndarray] = {}
    scri: Dict[int, int] = {}
    for key in npz.files:
        if key.startswith("sigma_"):
            tag = key[len("sigma_"):]
            k = 1 if tag == "fine" else int(tag[1:])
            sigma[k] = npz[key]
            scri[k] = int(npz[f"scri_idx_{tag}"])
    grids = Grids(tau=npz["tau"], sigma=sigma, scri_idx=scri)
    split = ""
    arrays: Dict[str, np.ndarray] = {}
    for key in npz.files:
        for cand in ("_P", "_qnm", "_psi_"):
            if cand in key:
                split = key.split("_", 1)[0]
                arrays[key[len(split) + 1:]] = npz[key]
                break
    meta = json.loads(str(npz["meta_json"])) if "meta_json" in npz.files else {}
    return split, arrays, grids, meta


def scri_waveform(arrays: Dict[str, np.ndarray], grids: Grids, i: int,
                  k: int = 1) -> np.ndarray:
    """The complex scri waveform of sample ``i`` on grid ``k`` (1=fine)."""
    tag = "fine" if k == 1 else f"k{k}"
    re = arrays[f"psi_{tag}_re"][i, :, grids.scri_idx[k]]
    im = arrays[f"psi_{tag}_im"][i, :, grids.scri_idx[k]]
    return re + 1j * im
