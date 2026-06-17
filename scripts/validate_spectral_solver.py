"""Validate the Chebyshev+PML spectral QNM solver against Leaver reference values.

Reference: fundamental gravitational QNMs of Schwarzschild (M=1 units), obtained
from the qnm package (Stein 2019), which solves Leaver's continued fraction.

Run:
    ./venv_csd3/bin/python scripts/validate_spectral_solver.py
"""

from __future__ import annotations

import numpy as np

from src.potentials import V_of_x
from src.spectral_qnm import PMLParams, solve_qnm_chebyshev_pml


# Leaver reference values, M=1, from the qnm package (Stein 2019).
LEAVER = {
    (2, 0): 0.3736716857 - 0.0889623157j,
    (2, 1): 0.3467109983 - 0.2739148752j,
    (2, 2): 0.3010534563 - 0.4782769828j,
    (2, 3): 0.2515049641 - 0.7051482014j,
    (3, 0): 0.5994432905 - 0.0927030479j,
    (4, 0): 0.8091783803 - 0.0941639610j,
}


def make_V(M: float, ell: int, potential: str):
    def V(x):
        return V_of_x(np.asarray(x, dtype=float), M=M, l=ell, potential=potential)
    return V


def validate(potential: str, M: float, ell: int, n_modes: int, pml: PMLParams, N: int) -> None:
    V = make_V(M, ell, potential)
    shift = LEAVER[(ell, 0)] / M  # convert M*omega -> omega for M != 1
    res = solve_qnm_chebyshev_pml(V, N=N, pml=pml, shift=shift, n_return=max(6, n_modes))
    # Dimensionalize: report M*omega for cross-comparison with Leaver table.
    Momegas = M * res.omegas
    print(f"\n  potential={potential}  l={ell}  N={N}  PML L={pml.L} L_phys={pml.L_phys} sigma0={pml.sigma0}")
    print(f"  {'n':>2} {'M*omega (Cheb+PML)':>34} {'M*omega (Leaver)':>34} {'|delta|':>14}")
    for n in range(n_modes):
        key = (ell, n)
        if key not in LEAVER:
            continue
        ref = LEAVER[key]
        # Match the n-th solver mode by closest to the n-th reference.
        idx = int(np.argmin(np.abs(Momegas - ref)))
        got = Momegas[idx]
        err = abs(got - ref)
        print(f"  {n:>2}  {got.real:>15.10f}{got.imag:>+15.10f}j   "
              f"{ref.real:>15.10f}{ref.imag:>+15.10f}j   {err:>10.2e}")


def main() -> None:
    # Tuned PML giving ~1e-7 accuracy on Schwarzschild Zerilli l=2 fundamental.
    pml = PMLParams(L=200.0, L_phys=5.0, sigma0=400.0, p=6)

    print("=" * 90)
    print("Convergence with N (Zerilli, M=1, l=2 fundamental)")
    print("=" * 90)
    for N in (400, 600, 800, 1000):
        validate("zerilli", M=1.0, ell=2, n_modes=1, pml=pml, N=N)

    print("\n" + "=" * 90)
    print("Higher l (Zerilli, M=1, fundamentals)")
    print("=" * 90)
    for ell in (2, 3, 4):
        validate("zerilli", M=1.0, ell=ell, n_modes=1, pml=pml, N=800)

    print("\n" + "=" * 90)
    print("Regge-Wheeler (M=1, l=2) -- should match Zerilli (isospectrality)")
    print("=" * 90)
    validate("regge-wheeler", M=1.0, ell=2, n_modes=1, pml=pml, N=800)

    print("\n" + "=" * 90)
    print("Mass scaling check (Zerilli, M=2, l=2)")
    print("=" * 90)
    # All length scales (L, L_phys, sigma0) carry units of M, so scale them up.
    pml_m2 = PMLParams(L=400.0, L_phys=10.0, sigma0=800.0, p=6)
    validate("zerilli", M=2.0, ell=2, n_modes=1, pml=pml_m2, N=800)


if __name__ == "__main__":
    main()
