# Covariance Cleaning

A sample covariance matrix overstates the large eigenvalues and
understates the small ones: its spectrum is spread out by the
Marčenko–Pastur law even when the population covariance is the identity.
Cleaning shrinks the eigenvalues (keeping the sample eigenvectors) toward
a fitted noise model.

* **Noise-level fit** — estimate ``\sigma^2`` from the bulk of the
  spectrum.
* **Eigenvalue clipping** (Laloux et al.) — replace bulk eigenvalues by
  their average, keep the outliers; trace-preserving.
* **Rotationally invariant estimator** (Bun, Bouchaud & Potters, whose
  shrinkage function is the one Ledoit–Wolf analytic nonlinear shrinkage
  also estimates) — a smooth, asymptotically Frobenius-optimal shrinkage.

## Finite-size note: the RIE broadening

The RIE evaluates the empirical Stieltjes transform at
``z_i = \lambda_i - i\eta`` with default broadening ``\eta = p^{-1/2}``.
At moderate ``p`` this leaves a positive bias in the mean cleaned
eigenvalue under the null (``\Sigma = I``): the broadening smooths the
spectrum over a window of ``O(\sqrt p)`` eigenvalues, and the bias is of
the order of ``\eta`` itself. Measured means of ``\mathrm{mean}(\xi)``
over 24 seeded Wishart trials:

| ``p`` | ``\gamma = 0.25``, ``\eta = p^{-1/2}`` | ``\gamma = 0.5``, ``\eta = p^{-1/2}`` | ``\gamma = 0.5``, ``\eta = p^{-2/3}`` |
|:------|:--------------------------------------:|:-------------------------------------:|:-------------------------------------:|
| 100   | 1.072                                  | 1.127                                 | 1.050                                 |
| 200   | 1.052                                  | 1.092                                 | 1.032                                 |
| 400   | 1.036                                  | 1.066                                 | 1.019                                 |

Shrinking ``\eta`` cuts the bias roughly threefold at ``\eta = p^{-2/3}``,
but ``\eta`` cannot approach 0 at fixed ``p``: the self-term
``1/(z_i - \lambda_i)`` is what regularizes the transform at its own
pole, and below the mean eigenvalue spacing the estimator deteriorates.
On a spiked population (``\gamma = 0.5``, Frobenius loss in oracle
units, 8 trials) the default is near-optimal — 1.21 / 1.15 / 1.14 at
``p = 100 / 200 / 400``, against 1.30 / 1.15 / 1.09 for
``\eta = p^{-2/3}`` and 1.56 / 1.32 / 1.21 for ``\eta = p^{-3/4}`` — so
the default stands. Pass a smaller `eta` to `rieEigenvalues` when an
accurate trace matters more than minimum loss, in particular at
``p \ge 200``.

```@docs
fitMarchenkoPastur
clipEigenvalues
rieEigenvalues
cleanCovariance
cleanCorrelation
```

## Worked example

Cleaning a spiked sample covariance (``p = 100``, ``n = 400``, one
population spike at ``\ell = 4``): the noise fit recovers
``\sigma^2 \approx 1``, clipping collapses the 99 bulk eigenvalues to a
single value while keeping the outlier, and the RIE shrinks the outlier
back toward the population spike.

```@example cleaning
using RandomMatrix, Random, LinearAlgebra, Statistics
Random.seed!(20261030)
p, n = 100, 400
gamma = p / n
S = randSpikedCovariance(p, n, [4.0])
ev = eigvals(S)
(fit = round(fitMarchenkoPastur(ev, gamma), digits = 3),
 raw_top = round(ev[end], digits = 3))
```

```@example cleaning
clipped = clipEigenvalues(ev, gamma)
(n_distinct = length(unique(clipped)), top = round(maximum(clipped), digits = 3))
```

```@example cleaning
rie = rieEigenvalues(ev, gamma)
(top = round(maximum(rie), digits = 3), mean = round(mean(rie), digits = 3))
```

The RIE mean above 1 is the finite-size broadening bias quantified in
the previous section (about ``1.09`` expected at ``\gamma = 0.5``;
here ``\gamma = 0.25`` and the spike also contributes).
