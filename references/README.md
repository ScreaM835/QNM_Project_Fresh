# References — local literature for QNM-PINN paper write-up

This folder collects the top external papers we read in full while
preparing the QNM-PINN paper. Each `.txt` file is a structured digest
(header + abstract + sections + equations + tables + references) of the
corresponding ar5iv-rendered paper, saved locally so it can be cited
without re-fetching.

Layout:

- `pinn_methodology/`  — foundational PINN papers (algorithmic backbone).
- `pinn_for_physics/`  — PINNs applied to QNMs / black-hole physics
                         (direct intellectual predecessors).
- `qnm_physics/`       — QNM physics references (ground-truth values,
                         framing, methods we benchmark against).

For each paper below: arXiv ID, citation purpose, and the stylistic /
structural feature(s) we should imitate when writing the paper.

================================================================================
PINN METHODOLOGY (7 papers)
================================================================================

1. **Raissi-Perdikaris-Karniadakis 2017, arXiv:1711.10561** — `pinn_methodology/1711.10561_raissi_perdikaris_karniadakis_2017_pinn_part1.txt`
   - PURPOSE: original PINN paper; mandatory primary citation when we
     define the loss L = L_PDE + L_BC + L_data.
   - STYLE: minimal "set up the PDE, write the loss, show it works on
     several benchmark problems" template. Their abstract is two
     sentences. Their introduction is half a page. Imitate this
     directness in our methods section.

2. **Lu-Meng-Mao-Karniadakis 2019, arXiv:1907.04502** — `pinn_methodology/1907.04502_lu_meng_mao_karniadakis_2019_deepxde.txt`
   - PURPOSE: DeepXDE library paper; cite for the PINN training setup
     ("FNN with tanh activation; Adam → L-BFGS; uniform domain points
     via DeepXDE's Geometry classes"). Our scripts use DeepXDE's
     `dde.PDE`, `dde.NN.FNN`, `dde.IC/BC`.
   - STYLE: software-paper structure (problem statement → API design →
     toy benchmarks). The Adam → L-BFGS two-phase training schedule is
     the standard quoted in this paper.

3. **Wang-Teng-Perdikaris 2020, arXiv:2001.04536** — `pinn_methodology/2001.04536_wang_teng_perdikaris_2021_gradient_pathologies.txt`
   - PURPOSE: explains why naive PINN losses fail (gradient-magnitude
     imbalance between L_PDE and L_BC); motivates loss-weight balancing.
     Cite when justifying our loss-weighting choices.
   - STYLE: diagnostic-then-fix layout — first show the pathology with
     plots of gradient magnitudes, then propose the fix. We will use
     the same "show the failure mode, then patch it" rhetoric for our
     FD refinement story.

4. **Wang-Yu-Perdikaris 2020, arXiv:2007.14527** — `pinn_methodology/2007.14527_wang_yu_perdikaris_2022_ntk_pinn.txt`
   - PURPOSE: NTK analysis of PINN training; theoretical grounding for
     why some loss components dominate and how loss weights should adapt.
   - STYLE: theoretically heavy paper — eigenvalue spectrum of the NTK
     drives the analysis. We won't replicate the depth, but we should
     cite when arguing that loss-weight choice is principled, not ad-hoc.

5. **McClenny-Braga-Neto 2020, arXiv:2009.04544** — `pinn_methodology/2009.04544_mcclenny_braganeto_2023_self_adaptive_pinn.txt`
   - PURPOSE: self-adaptive collocation-point weighting (SA-PINN). One
     of the candidate refinements for our PINN if percent-level accuracy
     is insufficient.
   - STYLE: clean ablation table comparing baseline PINN vs SA-PINN on
     several benchmarks. Imitate this table format for our FD-refinement
     comparison and for our PINN variant sweep.

6. **Wang-Sankaran-Perdikaris 2022, arXiv:2203.07404** — `pinn_methodology/2203.07404_wang_sankaran_perdikaris_2022_causal_training.txt`
   - PURPOSE: causal training for time-dependent PDEs; relevant if we
     write the time-dependent ringdown formulation.
   - STYLE: presents one specific tweak (causal weighting w_i =
     exp(−ε Σ_{j<i} L_j)) and exhaustively benchmarks it. Single-idea
     paper structure.

7. **Wu-Zhu-Tan-Kartha-Lu 2022, arXiv:2207.10289** — `pinn_methodology/2207.10289_wu_zhu_tan_kartha_lu_2023_adaptive_sampling.txt`
   - PURPOSE: comprehensive review of residual-based adaptive sampling
     methods for PINNs (RAR, RAD, RAR-D). Relevant if we want to upgrade
     our uniform-grid sampling.
   - STYLE: benchmark-heavy comparison paper. Useful template for the
     "what we tried, what worked, what didn't" appendix.

================================================================================
PINN FOR PHYSICS (4 papers — direct predecessors)
================================================================================

8. **Cornell-Ncube-Harmsen 2022, arXiv:2205.08284** — `pinn_for_physics/2205.08284_cornell_ncube_harmsen_2022_pinn_qnm_schwarzschild.txt`
   - PURPOSE: closest analogue to our work — first full PINN-QNM paper
     for the Schwarzschild Regge-Wheeler / Zerilli equation. Our paper
     must cite this prominently and clearly distinguish our contribution
     (FD-refined inverse PINN with mass extraction).
   - STYLE: short PRD-format paper. Section structure: Intro → Eq →
     PINN setup → Results (table of ω vs Leaver) → Discussion. We
     should mirror this length and structure.

9. **Luna-Bustillo-Seoane-Torres-Forné-Font 2022, arXiv:2212.06103** — `pinn_for_physics/2212.06103_luna_etal_2022_pinn_teukolsky_kerr.txt`
   - PURPOSE: extends Cornell 2022 to Kerr / Teukolsky. Demonstrates
     hard-BC enforcement (e^{x−1}−1) prefactor and sequential-warm-start
     parameter sweep — both directly applicable to our scheme.
   - STYLE: includes a DETECTABILITY-PROSPECTS section (eqs 14-18)
     using SNR-from-match argument. We must imitate this rhetorical
     move to justify our PINN's percent-level accuracy ("good enough
     for current detectors, insufficient for ET/LISA").

10. **Patel-Aykutalp-Laguna 2024, arXiv:2401.01440** — `pinn_for_physics/2401.01440_patel_aykutalp_laguna_2024_pinn_qnm_schwarzschild.txt`
    - PURPOSE: another recent PINN-QNM paper for Schwarzschild,
      including extensions to overtones. Most directly comparable in
      methodology — uses DeepXDE PyTorch backend like us.
    - STYLE: detailed ablations of network architecture and loss
      weighting. Imitate the systematic-variant-sweep presentation —
      this is exactly the format for our 5-variant inverse_qnm sweep.

11. **Luna et al. 2024, arXiv:2404.11583** — `pinn_for_physics/2404.11583_luna_etal_2024_pinn_qnm_modified_gravity.txt`
    - PURPOSE: PINN-QNM in modified-gravity / numerically-defined
      backgrounds where coefficients are not analytic. The strongest
      argument for the PINN approach over spectral methods.
    - STYLE: introduces ONE new physical application; brief on PINN
      methodology (defers to Luna 2022). Demonstrates that the
      methodology is portable.

================================================================================
QNM PHYSICS (1 paper — canonical review + ground truth)
================================================================================

12. **Berti-Cardoso-Starinets 2009, arXiv:0905.2975** — `qnm_physics/0905.2975_berti_cardoso_starinets_2009_qnm_review.txt`
    - PURPOSE: the canonical QNM review. Cite for: definitions, BCs,
      catalogue of competing methods (WKB, Leaver CF, monodromy,
      Frobenius), Schwarzschild ℓ=2 benchmark Mω = 0.3737 − 0.0890 i,
      Kerr BCW fits, eikonal/light-ring limit, GW-spectroscopy framing,
      no-hair-test rationale.
    - STYLE: this is a 156-page Topical Review — way longer than ours.
      But its INTRODUCTION (milestone bullet list + crisp notation
      table) is a useful template for our intro section.

================================================================================
WRITING-STYLE OBSERVATIONS TO CARRY INTO THE DRAFT
================================================================================

Across the 12 papers, the recurring stylistic features we should imitate:

1. **Crisp abstracts (3-5 sentences)**: state (i) the problem, (ii) the
   method, (iii) the headline result with a number. Cornell 2022 and
   Luna 2022 are the cleanest exemplars.

2. **One-sentence "contributions" bulleted list at end of intro**: Patel
   2024 and Luna 2024 both do this. We should add a "We" paragraph
   listing 3-4 specific contributions.

3. **Equation labelling discipline**: number every displayed equation
   that is referenced later; do NOT number ones that are not. Avoid
   the temptation to number for decoration.

4. **Results presented as a single benchmark table**: one row per
   variant / spin / ℓ value, columns (PINN, reference, % error).
   Cornell 2022 Table I; Luna 2022 Tables 1-3; Patel 2024 Tables 2-4.
   Our 5-variant sweep table should follow this format.

5. **Detectability framing in the discussion**: convert ω → (f, τ) →
   SNR_min via match (Luna 2022 §IV.2). Use this to justify the
   "good enough" accuracy claim.

6. **Conclusions written as actionable guidance**, not as a summary.
   E.g. "PINNs are preferred when the background metric is only known
   numerically" rather than "we showed PINNs work."

7. **Hard BCs via output surrogate** (Luna 2022 eq 12) — cite this
   when we describe our own BC encoding.

8. **Sequential warm-start over a parameter sweep** (Luna 2022 eq 13)
   — cite when we describe our M-sweep / inverse-PINN continuation.

9. **Reference to a single canonical ground-truth source**: BCW QNM
   tables [Berti webpage]. We use these as the absolute benchmark; do
   not re-derive Leaver values ourselves.

================================================================================
