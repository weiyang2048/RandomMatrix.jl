# Changelog

All notable changes to RandomMatrix.jl are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package uses semantic versioning.

Registration notes record when a version landed in the Julia General
registry; a version bump pushed to GitHub is not registered until the
user triggers Registrator.

## [Unreleased]

### Fixed

- Docs/exports drift (sweep finding RM-1): the exported legacy typo
  alias `randEllipic` is now documented as an alias of `randElliptic`
  (its docstring had been attached to the `randElliptic(n)` convenience
  method, which now carries a correct docstring of its own). An
  automated exports-vs-`@docs` check now closes at zero unexplained
  mismatches; the intentional exceptions are the re-exported upstream
  generics `mean`/`var`/`pdf`/`rand` (documented upstream) and the
  unexported bindings `preNORTA`/`qresolvent`, which are documented
  under qualified names (`RandomMatrix.preNORTA`,
  `RandomMatrix.qresolvent`). No binding was added, removed, or
  renamed.

### Added

- `TracyWidom` now supports `cdf`, `pdf`, `quantile`, `mean` and
  `var` for general `beta > 0`, computed by solving Bloemendal's
  boundary-value problem for the stochastic Airy operator
  (finite-difference scheme of Trogdon & Zhang, SIGMA 20 (2024),
  005) — previously those methods threw outside `beta ∈ {1, 2, 4}`
  and `mean`/`var` returned `NaN`. The general-beta law is the
  stochastic-Airy (beta-ensemble) law, the limit of the
  Dumitriu–Edelman sampler `rand` already used: it agrees with the
  classic laws at `beta = 1, 2`; at `beta = 4` it is the classic law
  divided by `2^(1/6)`, and `TracyWidom(4)` keeps the classic
  analytic values. (Trogdon & Zhang's own TracyWidomBeta.jl
  implements the same problem to higher accuracy; this solver is
  for integration with the package's type and test machinery.)
  Calibrated before the test bands were set: the solver reproduces
  the shipped analytic cdf to 2.4e-6 at `beta = 2` (7e-5 at
  `beta = 1`, and 5e-5 at `beta = 4` under the `2^(1/6)` rescaling —
  both at the analytic tables' own accuracy floor); its `beta = 4`
  stochastic-Airy moments match the published −2.05520 / 0.41092;
  moments at `beta = 1/2, 3, 6` match Dumitriu–Edelman Monte Carlo
  within finite-N bias; and at `beta = 10, 20` they match the
  published large-beta asymptotics (mean `−2.33811 + 1.12/beta`,
  variance `1.6697/beta`) to ≤ 0.7%.
- `profileWignerLaw(S)` (also in block form
  `profileWignerLaw(sizes, B)`): the limiting spectral distribution
  of a Wigner matrix with variance profile `S`
  (`E|X_ij|^2 = S_ij / n`), computed from the quadratic vector
  equation `-1/m_i = z + (1/n) sum_j S_ij m_j` (Ajanki, Erdős &
  Krüger, Probab. Theory Related Fields 170 (2018);
  arXiv:1506.05095) and returned as a `FreeMeasure`, like
  `silversteinLaw` — the law is defined only implicitly. Identical
  profile rows share an `m` value, so block-constant profiles solve
  on the block system exactly (Newton for small group counts,
  Anderson-accelerated fixed-point iteration above that); a
  coordinate with an identically zero profile row is an exact atom
  at 0 of mass `(zero rows)/n`, and the all-zero profile is
  `diracLaw(0)`. The constant profile collapses exactly to
  `semicircleLaw`; a block-diagonal profile is the exact mixture of
  block semicircles of radius `2 sqrt(b p)` (block variance `b`,
  coordinate fraction `p`); the exact second moment is the mean
  profile entry. `randProfileWigner([rng,] S; complex)` samples the
  ensemble (GOE/GUE diagonal conventions). Calibrated against the
  exact anchors and formed-matrix simulation before the test bands
  were set: semicircle density to `1.7e-5` in the interior and
  second moment to `2.5e-5` at the default resolution, block-mixture
  moments to `1e-4`, two-block CDF within `0.001` of simulation at
  `n = 400` with the support edge approached from below at the
  shallow-edge Tracy–Widom scale, solver residuals `<= 2e-10`.
- `brownMeasure(points, masses)` (also from a `FreeMeasure` or a
  `Distributions` law): the Brown measure of an R-diagonal element
  `a` whose `a* a` has law `nu` — the package's first non-Hermitian
  limiting law, and the limit of the eigenvalues of `A = U P` (or
  `U T V`) in the single ring theorem. By Haagerup & Larsen
  (J. Funct. Anal. 176 (2000), 331–367, Thm. 4.4, Cor. 4.5;
  formula re-verified against the primary text before
  implementation), the rotation-invariant measure is determined by
  `mu(B(0, S_nu(t-1)^{-1/2})) = t`, with exact radii
  `outer = sqrt(int t dnu)`, `inner = sqrt(1 / int t^{-1} dnu)`
  (0 when `nu` has an atom at 0), an atom at 0 of mass `nu({0})`,
  and planar density `Phi'(r)/(2 pi r)`; the matrix convergence is
  Guionnet–Krishnapur–Zeitouni (Ann. of Math. 174(2), 2011,
  1189–1217). Returned as a new `BrownMeasure` struct (radii, atom,
  radial CDF and density grids), read with `radialCDF` /
  `brownDensity`, since a real-line `FreeMeasure` cannot represent
  a planar law; `nu = delta_c` gives the singular uniform circle
  law of radius `sqrt(c)`. The computation parametrizes the
  Haagerup–Larsen relation through the moment transform of the
  discrete `nu` and differentiates it in closed form. Calibrated
  against the primary sources' own examples before test bands were
  set: the circular element returns `Phi(r) = r^2`, density `1/pi`
  (Haagerup–Larsen Ex. 5.1), the free Poisson family its exact
  `1/pi` annulus/disk densities (Ex. 5.2), the Bernoulli law its
  closed-form CDF and density (Ex. 5.3), a two-point `nu` the exact
  radii and the S-transform radial-moment identity
  `int |z|^{2k} dmu = int_0^1 S_nu(u-1)^{-k} du` to ~1e-4, and
  formed-matrix simulation matches the radial CDF to pooled KS
  ~0.03 at `n = 500` (finite-`n` edge layers).

- `silversteinLaw(points, masses, gamma)` (also from a `FreeMeasure`
  or a `Distributions` law): the limiting spectral distribution of a
  sample covariance `(1/n) T^{1/2} X X' T^{1/2}` with a general
  population spectral distribution `H` and aspect `gamma = p/n` —
  the general-population law of which `MarchenkoPastur` is the
  `H = delta_1` case. The law is computed from the Silverstein
  equation (Silverstein & Choi, J. Multivariate Anal. 54(2), 1995,
  Eq. (1.3); Bai & Silverstein, 2010, Ch. 4): Newton solution of the
  companion Stieltjes transform with the exact conversion
  `munderline = -(1 - gamma)/z + gamma * m_F`, density by Stieltjes
  inversion on a grid, support edges from the critical values of the
  inverse map `z(m)`, and the exact atom at 0 of mass
  `max(H({0}), 1 - 1/gamma)`. Returned as a `FreeMeasure`, since the
  law is defined only implicitly (no closed-form density, cdf, or
  sampler); a `Distributions` population is discretized on an
  equal-weight quantile grid first. Calibrated against the exact
  anchors before the test bands were set: `H = delta_1` reproduces
  `MarchenkoPastur` (edges exact, density to ~1e-3 in the interior),
  the closed-form first two moments hold to <= 0.3% at the default
  resolution (0.9% for the mean at the edge-singular `gamma = 1`),
  and the two-mass support-splitting transition matches
  formed-matrix simulation (no eigenvalues in the predicted gap).

- Free Bessel laws (Banica–Belinschi–Capitaine–Collins, Canad. J.
  Math. 63(1), 2011): `FreeBessel(s, t)` builds the law π_{s,t} as a
  `FreeMeasure` grid law — density from the algebraic equation
  f = 1 + z f^s (f + t − 1) solved on the real axis over the exact
  support, the atom (1 − t)₊ at 0 carried exactly — parametrized by
  (s, t), not by a product depth and Marcenko–Pastur ratio (a product
  of k ≥ 2 laws MP_γ with γ ≠ 1 is π_{s,t} for no (s, t)). Exact
  moments by the Fuss–Narayana formula (`freeBesselMoments`;
  Fuss–Catalan anchors 1, 3, 12, 55, 273 at s = 2 and 1, 4, 22, 140,
  969 at s = 3) and support/atom queries (`freeBesselSupport`) across
  all three regimes; the critical rectangle 0 < s < 1, t > 1 throws
  `ArgumentError`. Grid-law moments match the exact ones to ≤ 0.13%
  at the default resolution; the singular-edge rate
  π ρ(x) √x → 1/√(t − 1) (s = 2, t > 1) is suite-tested.
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
- The spiked-covariance layer now covers a general population
  spectrum: `bbpThreshold`, `spikeLocation`, `spikeOverlap` and
  `estimateSpike` gain methods taking the population spectral
  distribution `H` in the three input forms of `silversteinLaw`
  (discrete points/masses, a `FreeMeasure`, or a `Distributions`
  law), and a new `dTransform` exposes the D-transform
  `D(l) = l (1 + gamma * int t / (l - t) dH(t))` of Baik &
  Silverstein (J. Multivariate Anal. 97, 2006) whose minimizer is
  the generalized threshold, whose value at a supercritical spike
  is the outlier location, and whose logarithmic derivative gives
  the eigenvector overlap. Sub-threshold spikes follow the scalar
  layer's conventions (location returns the bulk edge — the upper
  support edge of `silversteinLaw(H, gamma)` — overlap returns 0,
  `estimateSpike` returns the threshold). At `H = delta_{sigma^2}`
  every formula reduces algebraically to the scalar layer;
  calibrated before the test bands were set: the reduction holds at
  rounding error (<= 5e-16 relative), two-point `H` reproduces the
  closed-form outlier `430/36` and overlap `287/387`, and
  formed-matrix simulation at a spread two-point bulk matches the
  predicted locations/overlaps within the calibrated bands
  (sub-threshold spikes stick to the bulk edge). `spikeVariance`
  is unchanged: it remains the identity-bulk (Paul) formula, with
  no general-`H` method.
- `KestenMcKay(d)` (default `d = 3`): the Kesten–McKay law, the
  limiting spectral distribution of a random `d`-regular graph's
  adjacency matrix (McKay, Linear Algebra Appl. 40, 1981), as a
  `Distributions` law with `pdf`/`cdf`/`mean`/`var`/`rand` in the
  `Wachter` idiom (Simpson `cdf` and exact rejection `rand`, both in
  the variable `x = 2 sqrt(d - 1) sin(theta)`). Support
  `[-2 sqrt(d - 1), 2 sqrt(d - 1)]`, mean 0, variance `d`; `d = 2`
  is the arcsine law, and the `sqrt(d)`-scaled law tends to the
  semicircle law as `d -> infinity`. Companion sampler
  `randRegularGraph([rng,] n, d)`: exactly uniform simple
  `d`-regular graphs via the configuration (pairing) model
  conditioned on simplicity (restart on loops/multiple edges;
  expected attempts about `exp((d^2 - 1) / 4)`, so practical for
  `d <= 5`). Calibrated before the test bands were set: density
  moments match the closed-walk counts on the `d`-regular tree
  (`m2 = d`, `m4 = 2d^2 - d`, `m6 = 5d^3 - 6d^2 + 2d`) to ~1e-12 by
  quadrature, `d = 2` reproduces the arcsine `pdf`/`cdf` to
  ~1e-11, the semicircle limit at `d = 1000` is within 2.3e-4, and
  pooled bulk spectra of seeded graphs (trivial eigenvalue `d`
  removed) match the law's moments and `cdf` (sup distance 0.005
  at `n = 200, d = 3`).
- `randomFeaturesLaw(mu0, mu1, muStar, psi, phi)` (also from an
  activation `Function`, or the presets `:linear`, `:relu`, `:erf`,
  `:quadratic`) and `hermiteCoefficients(sigma)`: the limiting
  spectral distribution of the random-features covariance
  `(1/n) Phi Phi'` for `Phi = sigma.(W X / sqrt(d))` with `W`
  (`m x d`) and `X` (`d x n`) i.i.d. Gaussian — the Mei & Montanari
  model, with `psi = n/d` and `phi = m/d`. The activation enters
  only through its Hermite coefficients for `Z ~ N(0,1)`: the bulk
  law is `silversteinLaw` with the affine Marchenko-Pastur
  population `H = law(mu1^2 * T + muStar^2)`, `T ~ MP_{m/d}`, at
  aspect `m/n` — equivalently the solution of the Pennington &
  Worah (2017) quartic self-consistent equation (Benigni & Peche
  2021; Schroeder & Piccolo 2021, Thm. 2.2, whose `theta_1`,
  `theta_2` are `mu1^2 + muStar^2`, `mu1^2`). Reduces exactly to
  scaled `MarchenkoPastur` when `mu1 = 0` and to the product-Wishart
  law `MP_{m/d} ⊠ MP_{m/n}` for a linear activation; `mu0` leaves
  the bulk unchanged and produces a single outlier at
  `mu0^2 * m (1 + o(1))`. Returned as a `FreeMeasure`.
  `hermiteCoefficients` evaluates `(mu0, mu1, muStar)` by
  Gauss-Hermite quadrature (Golub-Welsch). Calibrated before the
  test bands were set: the `mu1 = 0` reduction matches
  `MarchenkoPastur` (mean to ~5e-4, interior density to <= 2e-3);
  the composed law's moments carry a <= 1.5% positive bias at the
  default Stieltjes regularization (0.1% at `eps = 5e-5`);
  formed-matrix simulations at `(d, n, m) = (300, 450, 375)` match
  the ReLU/erf bulk moments within simulation error (bulk CDF
  distance ~0.01) and put the ReLU outlier at 1.014--1.019 times
  `mu0^2 * m`.

- `BetaLaguerre` now covers the singular case `n < p`: the
  constructor accepts any `n >= 1`, and `rand` returns the full
  `p`-eigenvalue spectrum of the singular Gram matrix — `p - n`
  exact zeros plus `n` positive eigenvalues drawn from the dual
  ensemble with `(n, p)` exchanged (the Gram swap between `X'X` and
  `XX'`, routed through the same Dumitriu–Edelman bidiagonal
  builder). Non-singular draws are unchanged, and
  `E[sum λᵢ] = n p` holds in both cases.

- `randElliptic` now supports the endpoint `r = -1` (perfect
  anticorrelation) exactly. The NORTA / bivariate-normal construction
  is singular there, so the matrix is built directly: upper triangle
  drawn from the entry law, lower triangle its negation, diagonal
  zero — exactly antisymmetric (`max|A + Aᵀ| = 0` in calibration, pair
  correlation exactly −1). This is realizable only when the entry law
  is symmetric about 0 (a negated draw must have the same law), checked
  on the support and a quantile grid; any other law throws
  `ArgumentError`. The spectrum is purely imaginary (max |Re λ| at
  rounding level, ~6e-16 at n = 800) with the imaginary parts
  semicircular of radius `2σ` under `norm = true` (moments m₂ = σ²,
  m₄ = 2σ⁴ verified for Gaussian, uniform and discrete entry laws).
- Spiked Wigner layer (`src/SpikedWigner.jl`), the additive BBP
  transition paralleling the spiked covariance and MANOVA layers:
  `wignerThreshold`, `wignerSpikeLocation` (`theta + sigma^2 / theta`
  above the threshold `|theta| > sigma`, sticking to the nearer
  semicircle edge below it), `wignerSpikeOverlap`
  (`1 - sigma^2 / theta^2`), `wignerEstimateSpike` (the inverse map),
  `wignerSpikeVariance`, and the GOE-based sampler
  `randSpikedWigner`. The noise scale is tied to the package's Wigner
  normalization (`sigma * W / sqrt(n)`, bulk on `[-2 sigma, 2 sigma]`,
  as in `rand(GOE(n), norm=true)` at `sigma = 1`). Fluctuations follow
  Capitaine–Donati-Martin–Féral (Ann. Probab. 37(1), 2009): Gaussian
  with variance `2 sigma^2 (1 - sigma^2 / theta^2) / n` (real; halved
  for complex), and for a spike along a coordinate direction with
  non-Gaussian entries the non-universal fourth-cumulant variance of
  their Thm. 2.2 via the `m4` keyword (at the Gaussian fourth moment
  it reduces exactly to the Gaussian formula). Calibrated against the
  package's GOE/GUE samplers and against Rademacher-entry simulation
  (coordinate-direction variance 1.184 empirical vs 1.219 predicted
  and 1.500 Gaussian, at `theta = 2`, `n = 300`, 1000 reps) before the
  test bands were set.

### Fixed

- `randElliptic` now validates the pair correlation at entry: `r` must
  be finite and lie in `[-1, 1]` (`r = 1` is the Hermitian case,
  `r = -1` the exactly antisymmetric one), and anything else throws
  `ArgumentError` naming the valid range.
  Previously `r = -1`, `|r| > 1` and non-finite `r` leaked a
  `PosDefException` from the bivariate-normal construction inside
  `preNORTA`, with no indication that `r` was the problem. Behavior
  for every previously valid `r` is unchanged.

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
