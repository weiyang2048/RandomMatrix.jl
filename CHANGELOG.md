# Changelog

All notable changes to RandomMatrix.jl are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the package uses semantic versioning.

Registration notes record when a version landed in the Julia General
registry; a version bump pushed to GitHub is not registered until the
user triggers Registrator.

## [0.4.5] - 2026-10-27

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
