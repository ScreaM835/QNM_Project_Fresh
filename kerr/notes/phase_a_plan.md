# Phase A — task breakdown

Hand-coded Schwarzschild-limit ($a=0$) Teukolsky on hyperboloidal slicing,
validated to the parent paper's Zerilli M4 result. Once $a=0$ is solid the
same code generalises to Kerr by adding spin-dependent terms.

Each task is one commit, one test, one acceptance check. Do not merge two.

## A.1 — radial coordinate map (THIS SESSION)
**File.** `src/hyperboloidal_schwarzschild.py`
**Implements.**
- `sigma_of_r(r, M)` and `r_of_sigma(sigma, M)` with the Macedo–Jaramillo–Ansorg
  choice $\sigma = 2M/r$, so $\sigma=0$ is scri$^+$, $\sigma=1$ is the
  horizon.
- `dr_dsigma(sigma, M)` analytic derivative.
- `tortoise(r, M)` standard Schwarzschild $r_* = r + 2M\ln(r/2M - 1)$.
- `dtortoise_dr(r, M)` analytic derivative $1/(1 - 2M/r)$.

**Acceptance.** Invertibility round-trip $|r - r(\sigma(r))|/r < 10^{-14}$ on a
log-spaced grid $r/M \in [2.001, 10^4]$; tortoise asymptotic
$r_* - r \to 2M\ln(r/2M)$ as $r \to \infty$ to $10^{-6}$ at $r/M = 10^4$;
divergence at horizon $r_* \to -\infty$ as $r \to 2M^+$.

## A.2 — hyperboloidal height function and slicing
**File.** `src/hyperboloidal_schwarzschild.py` (extend).
**Implements.** $h(r)$ minimal-gauge height function such that surfaces
$\tau = t - h(r) = \text{const}$ are hyperboloidal and horizon-penetrating.
Standard choice (Zenginoglu 2011, PRD 83 127502):
$$h(r) = -r + 2M\ln(r/2M) + 4M\ln(r/2M - 1)$$
or equivalently $h'(r) = -1 + 2M/r \cdot \big[1 + 1/(1 - 2M/r)\big]$.
Returns $h(r)$, $h'(r)$, $H(\sigma) \equiv h'(r(\sigma))$.

**Acceptance.** $1 - H(\sigma)^2 \cdot (\text{regularity coefficient}) \ge 0$
on $\sigma \in [0,1]$ (slicing is spacelike); $H(0) = 1$ (null at scri$^+$),
$H(1) = $ finite (horizon-penetrating); plot $h(r)$ matches Zenginoglu's
Fig. 1.

## A.3 — Regge–Wheeler operator on the hyperboloidal grid
**File.** `src/rwz_hyperboloidal.py`
**Implements.** Coefficients $A(\sigma), B(\sigma), C(\sigma), D(\sigma), E(\sigma)$
of the equation
$$\partial_\tau^2 \Phi + 2 H \partial_\tau \partial_\sigma \Phi - (1 - H^2 G) \partial_\sigma^2 \Phi
   + (\text{first-derivative and potential terms}) = 0,$$
in first-order form with $\Pi = \partial_\tau \Phi$. Regge–Wheeler potential
for axial $\ell=2$ on Schwarzschild,
$V_{\rm RW}(r) = (1 - 2M/r)[\ell(\ell+1)/r^2 - 6M/r^3]$.

**Acceptance.** All coefficients regular on the closed interval
$[0,1]$ (no $1/\sigma$ or $1/(1-\sigma)$ singularities after the
hyperboloidal transformation has been carried through symbolically — verify by
evaluating at $\sigma \in \{10^{-12}, 1 - 10^{-12}\}$ and checking no NaN/Inf).

## A.4 — spatial discretisation
**File.** `src/spatial_fd.py`
**Implements.** Second-order central differences on uniform $\sigma$ grid
of $N$ points with one-sided stencils at the two boundaries (no ghost
zones, no BC enforcement: the hyperboloidal slicing makes both boundaries
characteristic outflow).

**Acceptance.** On a manufactured smooth $\Phi(\sigma) = \sin(2\pi\sigma)$ the
discrete $\partial_\sigma$ and $\partial_\sigma^2$ converge at second order
under $N \to 2N \to 4N$.

## A.5 — RK4 time stepper
**File.** `src/rk4_mol.py`
**Implements.** Classical 4-stage RK4 on the first-order system
$(\Phi, \Pi)$. CFL estimate
$\Delta\tau \le \text{cfl\_safety} \cdot \Delta\sigma / \max_\sigma c_+(\sigma)$
where $c_+$ is the larger characteristic speed of the principal symbol of
the hyperboloidal system at $\sigma$.

**Acceptance.** Conservation of constraints (none here, but check $L^2$
energy growth bounded by an exponential whose rate matches the analytic
quasi-normal decay $\sim e^{-t/\tau_{220}}$) on the canonical Gaussian
initial datum, run to $t = 200M$, no NaN.

## A.6 — initial data
**File.** `src/initial_data_hyp.py`
**Implements.** Gaussian pulse at hyperboloidal $\tau = 0$:
$\Phi(0, \sigma) = A_0 \exp(-(r(\sigma) - x_0)^2/(2\sigma_{\rm IC}^2))$,
$\Pi(0, \sigma) = 0$. Time-symmetric, splits into ingoing+outgoing.

**Acceptance.** Peak located at $r = x_0$ in tortoise; total $L^2(\sigma)$
matches the analytic Gaussian $L^2$ to integration-rule precision.

## A.7 — observer extraction
**File.** `src/observers.py`
**Implements.** Lagrange/spline interpolation at fixed tortoise positions
$x_q \in \{2M, 10M\}$ at every time step, returns $\Phi(t, x_q)$ time
series. **NOTE:** the recorded time variable is the hyperboloidal $\tau$;
to compare to the parent paper's Schwarzschild time $t$ at the observer,
shift by $h(r(x_q))$.

**Acceptance.** Reproduces the input on a manufactured solution
$\Phi(\tau, \sigma) = f(\tau - h(r(\sigma)))$ for a smooth $f$.

## A.8 — wire `validate_a0.py`
Replace the `NotImplementedError` with a full driver:
solve $\to$ extract observer time series $\to$ M4 plateau extraction
$\to$ compare to `qnm` package and the parent paper's Schwarzschild
$M\omega_0 = 0.3737$, $\tau_0/M = 11.241$.

**Acceptance.** Phase A gate (verbatim from README):
- $|M\omega_{\rm extracted} - M\omega_{\rm qnm}| / M\omega_{\rm qnm} < 10^{-3}$
- $|M\omega_{\rm extracted} - 0.3737| / 0.3737 < 5\times 10^{-4}$
- integrator stable to $t = 200M$, no boundary growth.

## A.9 — extend to Kerr
Add spin-dependent coefficients to A.3 and the source-term in A.6 to give
$(s=-2, \ell=m=2)$ Teukolsky at $a/M > 0$. This is Phase B.

## References
- Macedo, Jaramillo, Ansorg, PRD 89 064008 (2014). Hyperboloidal slicing,
  Schwarzschild radial map.
- Zenginoglu, PRD 83 127502 (2011). Minimal-gauge height function.
- Harms, Bernuzzi, Brügmann, CQG 31 245004 (2014). Hyperboloidal
  Teukolsky on Kerr.
- Pazos-Avalos, Lousto, PRD 72 084022 (2005). Time-domain Teukolsky
  validation reference (Cauchy slicing, tortoise + sponge).
