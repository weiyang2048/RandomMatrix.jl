# Changelog

All notable changes to RandomMatrix.jl are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package uses semantic versioning.

Registration notes record when a version landed in the Julia General
registry; a version bump pushed to GitHub is not registered until the
user triggers Registrator.

## [Unreleased]

### Added

- `lssTest(evals, n; f=:x2, beta=1, mu4=nothing, sigma2=1.0)` — the
  Bai–Silverstein CLT test of a covariance spectrum against the
  Marčenko–Pastur null, the bulk companion of `twTest`: the centered
  linear spectral statistic `G = Σ f(λᵢ/σ²) − p∫f dMP_γ` for
  `f ∈ (:x, :x2, :x3, :log)` is asymptotically normal with closed-form
  mean/variance (real case with fourth-moment `mu4` corrections —
  monomial means scale by `(mu4 − 2)`, `log x` carries its own
  coefficients `−γ/2` / `γ`; complex case mean 0, variance halved),
  returning `(statistic, pvalue, G, mean, var)`. Anchored on the
  exact identities `E G(x²) = γ`, `E G(x³) = 3γ(1+γ) + 4γ/n`, the
  corrected-LRT values for `x − 1 − log x`, the Rademacher witness
  (variance of `G(x²)` collapses 10 → 1 at `γ = 0.5`), and a seeded
  Wishart size check at `p = 200`.

- `BetaLaguerre` now covers the singular case `n < p`: the
  constructor accepts any `n >= 1`, and `rand` returns the full
  `p`-eigenvalue spectrum of the singular Gram matrix — `p - n`
  exact zeros plus `n` positive eigenvalues drawn from the dual
  ensemble with `(n, p)` exchanged (the Gram swap between `X'X` and
  `XX'`, routed through the same Dumitriu–Edelman bidiagonal
  builder). Non-singular draws are unchanged, and
  `E[sum λᵢ] = n p` holds in both cases.

### Fixed

- `randElliptic` now validates the pair correlation at entry: `r` must
  be finite and lie in `(-1, 1]` (`r = 1` is the Hermitian case), and
  anything else throws `ArgumentError` naming the valid range.
  Previously `r = -1`, `|r| > 1` and non-finite `r` leaked a
  `PosDefException` from the bivariate-normal construction inside
  `preNORTA`, with no indication that `r` was the problem. Behavior
  for every valid `r` is unchanged; `r = -1` (perfect anticorrelation)
  remains unsupported, as the construction is singular there.

## [0.12.0] - 2026-11-10

Not yet registered in General.

### Added

- `manovaTWTest(theta, p, m, n; beta=1)` and
  `manovaTWScaling(p, m, n; beta=1)`: Johnstone's Tracy–Widom
  greatest-root test for a MANOVA (double Wishart) problem — the
  balanced-regime Jacobi analogue of `twTest`. For the largest
  eigenvalue `theta` of `(A + B)⁻¹ B` with `A ~ W_p(I, m)` the
  error Wishart and `B ~ W_p(I, n)` the hypothesis Wishart, the
  logit transform `W = log(theta / (1 - theta))` is centered and
  scaled by the constants of Johnstone, Ann. Statist. 36(6), 2008
  (eqs. (5)–(6) for real data; the weighted constants of Theorem 2
  for complex data, `beta=2`), giving a `TracyWidom(beta)`
  statistic and upper-tail p-value (Roy's largest-root test with a
  Tracy–Widom p-value). Calibrated against simulation before the
  test bands were set: at `(p, m, n) = (20, 160, 40)`,
  `(50, 100, 50)`, `(40, 120, 20)` and `(30, 60, 90)` the scaled
  statistic's mean/sd match the Tracy–Widom values within Monte
  Carlo error and the 5% rejection rate falls in 0.040–0.054
  (real) and 0.043–0.047 (complex).

## [0.11.0] - 2026-11-08

Not yet registered in General.

### Added

- `fitWachter(evals)`: method-of-moments estimation of the Wachter
  aspect ratios `γ₁ = p/n₁`, `γ₂ = p/n₂` from a MANOVA spectrum (the
  Jacobi analogue of `fitMarchenkoPastur`). The law's mean
  `m = γ₂/(γ₁ + γ₂)` and variance `v = (γ₁γ₂)²/(γ₁ + γ₂)³` invert in
  closed form via `v = s (m (1 - m))²` with `s = γ₁ + γ₂`; atoms at
  0/1 enter the empirical moments directly. Spectra no Wachter law can
  match (degenerate, or at the Bernoulli bound `v = m (1 - m)`) throw
  `ArgumentError`. Calibrated in-suite: exact moment round-trip,
  pooled MANOVA fit (0.504, 0.252) at true (0.5, 0.25), atom-regime
  fit (1.502, 0.500) at true (1.5, 0.5).

- Unbalanced complex Jacobi Tracy-Widom correction (Didier, Gkogkou,
  Lopez Santander & McLaughlin, arXiv:2611.00577). For
  `J = (A + B)^{-1} B` with `A ~ CW_p(I, m)`, `B ~ CW_p(I, n)` in the
  regime `n - p = b p`, `m - p = a p^(1 + nu)`, `nu > 0`,
  `jacobiTWScaling(p, a, b, nu)` returns the finite-p location
  `beta_p` (eq. (2.12)) and scale `checkC_p p^(-2/3)` (eq. (4.23))
  standardizing the largest eigenvalue to `TracyWidom(2)`, and
  `jacobiTWCDF` / `jacobiTWThreshold` implement Theorem 3.1's
  explicit `p^(-2/3)` correction
  `F_2(s) (1 - 2 Chat_p p^(-2/3) v(s)) + O(p^(-1))` and its
  threshold map `phi_p(chi_p^{-1}(s))`. The public scale is the
  finite-p coefficient, NOT Corollary 3.2's closed-form limiting
  scale `1 / (C p^(nu + 2/3))`: the two agree only to `1 + o(1)`
  with a slowly vanishing error (relative deficit ~ `7 p^(-nu)` at
  `b = 1`), and at `nu = 1/2` the limiting scale runs 2.36 / 1.89 /
  1.60 times the finite-p scale at `p = 50 / 100 / 200`, so it is
  not exposed. Simulation-validated: at `nu = 1/2`, `a = b = 1`, the
  standardized largest eigenvalue has mean/sd -1.821/0.906,
  -1.777/0.891, -1.771/0.886 at `p = 50 / 100 / 200`
  (20,000 / 8,000 / 3,000 replications; `TracyWidom(2)` targets
  -1.7711 / 0.9018), the correction term matches its predicted size
  and `p^(-2/3)` decay (max 0.0026 / 0.0016 / 0.0010 of CDF), and
  the remaining error is the theorem's `O(p^(-1))` remainder (sup
  0.014 at `p = 50`), which dominates the displayed term at
  simulable sizes.

## [0.10.0] - 2026-11-06

Registered in General 2026-11-07 (PR #175272).

### Added

- Finite-temperature spectral form factor: `spectralFormFactor` and
  `ensembleSpectralFormFactor` now take a `beta` keyword (inverse
  temperature, default `0.0`) giving Cotler et al.'s
  `g(t, beta) = |Z(beta + i t)|^2 / Z(beta)^2` with
  `Z(beta + i t) = sum_j exp(-(beta + i t) E_j)`. At `beta = 0` this
  reduces exactly to the previous zero-temperature functions. At
  `beta > 0` the levels carry thermal weights `exp(-beta E_j)`:
  `g(0, beta) = 1` still holds, and the late-time plateau rises from
  `1/N` to the participation ratio `sum w^2 / (sum w)^2` of the
  weights — calibrated in the test suite against the time-averaged
  form factor of raw `BetaHermite` spectra (agreement at the
  percent level at `beta = 0.5`). The ensemble decomposition
  normalizes each spectrum by its own `Z(beta)`, so
  `total = connected + disconnected` holds at every `beta`.
  Amplitudes are evaluated on energies shifted by a common constant
  (which cancels exactly in the normalized ratio), so spectra at
  large absolute energies stay finite where the raw exponentials
  would underflow.

## [0.9.0] - 2026-11-06

Registered in General 2026-11-06 (PR #175190, clean AutoMerge).

### Added

- General-β ensemble eigenvalue samplers: `BetaHermite(β, n)` and
  `BetaLaguerre(β, n, p)` are `Distributions` multivariate
  distributions whose `rand` returns the sorted spectrum of the
  Gaussian / Wishart β-ensemble at any `β > 0`, sampled from the
  Dumitriu–Edelman tridiagonal models (J. Math. Phys. 43, 2002).
  The Laguerre law at `β = 1, 2` is exactly the real / complex
  Wishart eigenvalue law, and its `1/n`-scaled bulk follows
  `MarchenkoPastur(p/n)` at every `β`. Both parametrizations were
  calibrated against the exact trace identities
  (`E[Σλ²] = 2n/β + n(n−1)`, `E[Σλ] = np`) and the semicircle /
  Marchenko–Pastur bulk moments and Tracy–Widom edge means before
  the test bands were fixed (trace ratios within 1e-3 at
  `β ∈ {0.5, 1, 2, 3.7, 4}`). The Hermite tridiagonal builder is
  shared with the general-β `rand` of `TracyWidom`.
- Spiked MANOVA outlier fluctuations: `manovaSpikeVariance(θ, γ₁, γ₂, p)`
  gives the asymptotic variance of the outlier sample eigenvalue of
  `(A + B)⁻¹ A` generated by a supercritical population spike `θ`
  (Gaussian populations; `√p (x̂ - manovaSpikeLocation(θ, γ₁, γ₂))` is
  asymptotically centred normal with variance
  `p * manovaSpikeVariance(θ, γ₁, γ₂, p)`; `NaN` inside the critical
  window). The formula is the delta method applied to the spiked
  Fisher CLT of Dharmawansa, Johnstone & Onatski (2014, Prop. 11,
  Setting 1 — the Gaussian case of Wang & Yao 2017, Thm. 4.1) through
  the Fisher-to-MANOVA map `x = t/(1+t)`, `t = (γ₂/γ₁)φ(θ)`. The same
  rational expression holds below the lower critical spike; both
  sides were calibrated against spiked MANOVA simulation before the
  test bands were fixed (variance ratios 0.86–1.10 across a
  nine-point `(θ, γ₁, γ₂)` grid at `p = 100`–`200`, 300–1500 reps,
  on the Fisher and MANOVA scales alike).
- Free compression of a `FreeMeasure`: `freeCompress(μ, τ)` is the
  law of `p a p` in the compressed space `(pAp, φ/τ)`, where `a` has
  law `μ` and `p` is a projection of trace `τ ∈ (0, 1]` free from
  `a` (Nica & Speicher, *Lectures on the Combinatorics of Free
  Probability*, 2006). The R-transform scales as
  `R_{μ_τ}(w) = R_μ(τ w)` (free cumulants
  `κₙ → τⁿ⁻¹ κₙ`, so the mean is preserved), and an atom of mass
  `m` survives exactly when `m > 1 - τ`, with compressed mass
  `(m - (1 - τ))/τ`. Headline case: the Marčenko–Pastur law
  compresses as `MP_γ → MP_{τγ}` — the limiting spectral law of a
  principal submatrix retaining a `τ` fraction of the coordinates,
  with the atom at 0 disappearing exactly at `τγ = 1`. The Cauchy
  transform is evaluated through the implicit equation
  `G_{μ_τ}(z) = G_μ(s)/τ`, `s - (1 - τ)/G_μ(s) = z`, by Newton
  iteration on the exact rational `G_μ`. Anchored in the test suite
  against the `MP_{τγ}` moments (errors ≤ 1.3% on the first three
  moments at the default resolution), the atom rules at and away
  from 0, semicircle variance scaling, Bernoulli cumulant-scaling
  moments, dilation composition, and Wishart principal-submatrix
  simulation.
- Spectral form factor diagnostics, beside the spacing statistics:
  `spectralFormFactor(E, t)` is the single-spectrum form factor
  `|Σⱼ exp(-itEⱼ)|²/N²` of the levels exactly as given (the existing
  `unfoldedeigvals` is applied only with `unfold=true`), and
  `ensembleSpectralFormFactor(spectra, t)` returns the
  `(total, connected, disconnected)` decomposition of Cotler et al.,
  "Black holes and random matrices", J. High Energy Phys.
  2017(5):118. Calibrated against the structural facts before the
  test bands were fixed: unfolded `BetaHermite` spectra dip to
  0.13 (β = 1) / 0.07 (β = 2) of the `1/N` plateau and ramp back to
  it at the Heisenberg time, while Poisson levels sit flat at the
  plateau (minimum 0.84 of `1/N`). The finite-temperature form
  factor is not implemented.

### Fixed

- The `beta = 4` conversion in `_tw_tridiagonal_sample` had the
  `2^(1/6)` factor in the wrong direction: the Dumitriu–Edelman edge
  variable converges to the stochastic-Airy TW_4 (mean -2.05520,
  var 0.41092), and multiplication by `2^(1/6)` — not division —
  gives the classic `TracyWidom(4)` variable. The branch was dead
  code (`rand` takes the analytic path for `beta in {1, 2, 4}`), so
  no observable behavior changes; the `BetaHermite` / `BetaLaguerre`
  docstrings now state the `beta = 4` caveat explicitly, and a
  seeded moment-band test pins the direction.

## [0.8.0] - 2026-11-03

Registered in General 2026-11-03 (PR #174646, clean AutoMerge).

### Added

- Spiked MANOVA (Jacobi BBP) phase transition, the outlier layer over
  the Wachter law: `manovaThreshold(γ₁, γ₂)` gives the upper critical
  spike `(1 + ρ) / (1 - γ₂)` with
  `ρ = sqrt(γ₁ + γ₂ - γ₁γ₂)`, and `manovaSpikeLocation(θ, γ₁, γ₂)`
  gives the almost-sure limit of the sample eigenvalue of
  `(A + B)⁻¹ A` generated by a population spike `θ` — Wang & Yao,
  Ann. Statist. 45(1), 2017, Thm. 3.1, mapped from the Fisher matrix
  to the MANOVA eigenvalues and calibrated against spiked MANOVA
  simulation in the test suite. Landed on `main` 2026-11-02 (commit
  d5c71b9, docs integration completed by the nightly run the same
  night).

## [0.7.0] - 2026-11-01

Registered in General 2026-11-02 (PR #174484, via the
`[merge approved]` override over the skipped v0.6.0).

### Added

- The Wachter law (MANOVA / Jacobi limiting spectral distribution):
  `Wachter(γ₁, γ₂)` is the limiting distribution of an eigenvalue of
  `(A + B)^{-1} A` for independent Wisharts `A ~ W_p(I, n₁)`,
  `B ~ W_p(I, n₂)` with aspect ratios `γ₁ = p / n₁`, `γ₂ = p / n₂`
  (the `MarchenkoPastur` convention). Closed-form density on
  `[x₋, x₊] ⊂ [0, 1]` with `x± = γ₂(γ₁ ± ρ)² / (γ₁ + γ₂)²`,
  `ρ = sqrt(γ₁ + γ₂ - γ₁γ₂)`; an atom at 0 of mass `1 - 1/γ₁` when
  `γ₁ > 1` and at 1 of mass `1 - 1/γ₂` when `γ₂ > 1`; mean
  `γ₂ / (γ₁ + γ₂)` and variance `(γ₁γ₂)² / (γ₁ + γ₂)³`. `pdf`,
  `cdf` (quadrature in the `sin²θ` variable), `mean`, `var` and
  exact rejection `rand` (in the same variable) are provided.
  The parametrization, edges, atoms and moments were verified
  against MANOVA-matrix simulation across the parameter range,
  including both atom regimes; the domain requires
  `γ₁ + γ₂ - γ₁γ₂ > 0` (equivalently `n₁ + n₂ > p`).
  References: Wachter, Ann. Statist. 8(5), 1980; Collins, Probab.
  Theory Related Fields 133, 2005; Johnstone, Ann. Statist. 36(6),
  2008.

## [0.6.0] - 2026-10-30

Never registered in General — skipped at the user's direction; its
content reached General with v0.7.0.

### Added

- Johnstone's Tracy–Widom test for the largest eigenvalue:
  `twTest(lambda, n, p; beta=1)` returns a NamedTuple
  `(statistic, pvalue)` — the scaled statistic
  `(lambda - mu) / sigma` from `twScaling(n, p)` and its upper-tail
  p-value under `TracyWidom(beta)` — answering "is the top sample
  eigenvalue signal or noise?" against the identity-covariance null
  without wiring the two ingredients by hand. `beta = 1` (default)
  is real data, `beta = 2` complex data. Reference: Johnstone,
  Ann. Statist. 29(2), 2001, DOI 10.1214/aos/1009210544.
  Simulation-calibrated under the null (at `n = 400`, `p = 100` the
  test rejects slightly conservatively: 0.033 at the 5% level over
  400 trials, mean p-value 0.504).

## [0.5.1] - 2026-10-30

Not yet registered in General.

### Added

- Constant-correlation target for the Ledoit–Wolf linear shrinkage
  estimator: `ledoitWolf(X; target = :constant_correlation)` and
  `ledoitWolfShrinkage(X; target = :constant_correlation)` implement
  Ledoit & Wolf (2003), shrinking toward the matrix that keeps the
  sample variances and sets every correlation to the average sample
  correlation, with the shrinkage intensity given by the papers'
  asymptotic formula for this target (including the covariance of
  the sample covariance with the random target). The default
  `target = :identity` is unchanged.

## [0.5.0] - 2026-10-30

Not yet registered in General.

### Added

- Ledoit–Wolf (2004) linear shrinkage from the data matrix:
  `ledoitWolf(X)` returns the shrunk covariance
  `(1 - rho) * S + rho * m * I` as a `Symmetric` matrix and
  `ledoitWolfShrinkage(X)` the estimated intensity `rho in [0, 1]`
  (`X` is `p x n`, variables in rows, centered internally). These are
  the package's first data-based covariance estimators — the existing
  cleaners work from a spectrum — and the intensity needs the data
  themselves because it involves fourth moments of the observations.
  Intensities agree with scikit-learn's `ledoit_wolf` to 1e-15 on
  fixed test datasets.
- OAS (oracle approximating shrinkage; Chen, Wiesel, Eldar & Hero
  2010): `oas(X)` and `oasShrinkage(X)`, same form and data layout.
  The intensity implements the paper's Eq. (23) *with* the `2/p`
  terms; scikit-learn's `oas` omits them, which changes `rho` by
  several percent at small `p`.
- Worked example for the linear shrinkage estimators on the Covariance
  Cleaning documentation page.

## [0.4.7] - 2026-10-29

Not yet registered in General.

### Added

- Worked `@example` blocks in the four v0.4.0 documentation pages
  (Spiked Covariance / BBP, Covariance Cleaning, Tracy–Widom, Free
  Probability): seeded, executable examples whose outputs are checked
  against the theory on every docs build — BBP threshold, location and
  overlap with a simulated outlier; Marčenko–Pastur noise fit, clipping
  and RIE shrinkage on spiked data; Tracy–Widom moments, cdf and
  Johnstone scaling; and free Poisson / free convolution moments
  against their exact values. The examples deliberately avoid Plots:
  the Plots/GIF pages already dominate the docs build time.

## [0.4.6] - 2026-10-28

Not yet registered in General.

### Fixed

- `freePoissonLaw` was broken for `gamma` near 1: on the uniform grid
  the mean came out 2.3% high at `gamma = 0.99` and `1.01`, and at
  `gamma = 1` every weight was NaN (the Marčenko–Pastur density at 0
  is NaN), which also made `freeMul` with an `MP_1` factor return NaN
  moments after minutes of Newton iterations. The law is now built on
  a Chebyshev grid, where the transformed density is smooth for every
  `gamma`; mean and variance match `1` and `gamma` to about 1e-7.

### Changed

- `freeMul` accuracy: the output grid is now a Chebyshev grid
  clustered at the support ends, the default `eps` is half the mean
  grid spacing (was: the uniform spacing), the result assembly uses a
  variable-spacing trapezoid, and the Newton sweep retries once from a
  reset state when a point fails to converge instead of silently
  accepting the last iterate. Against the exact S-transform moments of
  `MP_gamma ⊠ MP_delta` at the default resolution, the first three
  moments now err by at most about 2.5% when both factors are bounded
  away from 0 (the third moment of `MP_0.5 ⊠ MP_0.25` erred 2.9%
  before), about 5% for
  `MP_1 ⊠ MP_0.5` and about 11% for `MP_1 ⊠ MP_1` (was: 8% and 15% on
  top of the `gamma = 1` NaN failure). The remaining touching-0 error
  is the `eps^(1/3)` smoothing leak below 0, documented in the
  `freeMul` docstring with `eps` guidance.

## [0.4.5] - 2026-10-27

Registered in General on 2026-10-28.

### Added

- This `CHANGELOG.md`, backfilled for v0.3.0–v0.4.4.

## [0.4.4] - 2026-10-26

Registered in General (JuliaRegistries/General#173655).

### Added

- `rieEigenvalues` accepts the ASCII keyword `eta` as an alias for `η`
  (previously only the Greek spelling worked, although the docstring
  advertised `eta`).

### Changed

- Documented the finite-size bias of the RIE broadening
  `η = p^(-1/2)`: under the null the mean cleaned eigenvalue is biased
  up by about `c(γ) * p^(-1/2)`, and `η = p^(-2/3)` reduces that bias
  roughly threefold without improving spiked-model loss except at
  large `p`. The default prescription is unchanged.

## [0.4.3] - 2026-10-26

Registered in General (JuliaRegistries/General#173584).

### Changed

- Version bump only, superseding v0.4.2, whose General registration
  never landed (its AutoMerge run was stranded by registry
  infrastructure). No code changes relative to v0.4.2.

## [0.4.2] - 2026-10-25

Never registered in General (see v0.4.3).

### Added

- Aqua.jl test-quality checks as a testset (`Aqua.test_all`): ambiguity,
  unbound-argument, stale-dependency, piracy and related checks.
  Aqua is a test-only dependency; no new runtime dependencies.

### Fixed

- A method ambiguity between `Distributions._rand!` and this package's
  `_rand!` for its complex distributions.
- Missing `Test` compat entry.
- Missing docstrings (and docs entries) for the exported names
  `numberVariance`, `randErdosRenyi` and `stieltjes`.

## [0.4.1] - 2026-10-25

Registered in General (JuliaRegistries/General#173341).

### Fixed

- `CovarianceCleaning._mp_median` was wrong for `γ >= 1`: it returned
  the lower Marčenko–Pastur edge for all `γ >= 1`. At `γ = 1` the true
  median is about 0.6528, so median-based fitting returned `Inf`; for
  `γ > 2` the atom at 0 makes the median exactly 0. The median is now
  computed by integrating the continuous part of the law.
- `fitMarchenkoPastur(method=:median)` now throws an `ArgumentError`
  for `γ >= 2`, where median matching is degenerate; use
  `method=:trim` there instead.

## [0.4.0] - 2026-10-24

### Added

- Spiked covariance model (`src/SpikedCovariance.jl`): BBP threshold,
  spike location / overlap / variance, the inverse spike estimator
  `estimateSpike`, the rectangular (singular-value) case, and the
  `randSpikedCovariance` sampler.
- Marčenko–Pastur covariance cleaning (`src/CovarianceCleaning.jl`):
  noise fitting (`fitMarchenkoPastur`, trim and median methods),
  trace-preserving eigenvalue clipping, the Bun–Bouchaud–Potters
  rotationally invariant estimator (`rieEigenvalues`), and the
  `cleanCovariance` / `cleanCorrelation` drivers.
- Tracy–Widom distributions (`src/TracyWidom.jl`): `TracyWidom` for
  `β = 1, 2, 4` from the Hastings–McLeod Painlevé II solution
  (cdf / pdf / quantile through Distributions), general-`β` sampling
  by Dumitriu–Edelman tridiagonalization, and Johnstone's Wishart
  centering and scaling `twScaling`.
- Numerical free probability (`src/FreeProbability.jl`): the
  `FreeMeasure` representation, Cauchy / R / S transforms, free
  additive convolution by subordination (`freeAdd`), free
  multiplicative convolution by the S-transform product (`freeMul`),
  with exact atom rules.
- Documentation pages for all four areas.

## [0.3.0] - 2026-10-24

### Changed

- Packaging modernized for current Julia: the plotting code moved into
  a package extension (`RandomMatrixPlotsExt`, with Plots as a weak
  dependency), stale files removed, and the test suite expanded.
- Documentation build fixed for Documenter 1.x.
