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

```@docs
fitMarchenkoPastur
clipEigenvalues
rieEigenvalues
cleanCovariance
cleanCorrelation
```
