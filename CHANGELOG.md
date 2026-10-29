# Changelog

All notable changes to RandomMatrix.jl are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package uses semantic versioning.

Registration notes record when a version landed in the Julia General
registry; a version bump pushed to GitHub is not registered until the
user triggers Registrator.

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
