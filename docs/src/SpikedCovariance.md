# Spiked Covariance and the BBP Transition

In a spiked covariance model the population covariance has a bulk of
eigenvalues at ``\sigma^2`` plus a few larger spikes. With aspect ratio
``\gamma = p/n``, the Baik–Ben Arous–Péché (BBP) transition says a spike
``\ell`` is visible in the sample covariance — as an outlier eigenvalue
with an eigenvector correlated with the population direction — only when
``\ell > \sigma^2 (1 + \sqrt\gamma)``. Below the threshold the spike is
swallowed by the Marčenko–Pastur bulk.

```@docs
bbpThreshold
spikeLocation
spikeOverlap
estimateSpike
spikeVariance
randSpikedCovariance
```

## Worked example

At ``\gamma = 0.5`` a spike of ``\ell = 4`` is well above the BBP
threshold, so it should appear as an outlier near
`spikeLocation(4, 0.5) = 4.667` with eigenvector overlap about `0.81`;
a spike of ``\ell = 1.3`` is below the threshold and sticks to the
Marčenko–Pastur edge ``(1 + \sqrt\gamma)^2 \approx 2.914``.

```@example spiked
using RandomMatrix
gamma = 0.5
(threshold = bbpThreshold(gamma), loc = spikeLocation(4.0, gamma),
 overlap = round(spikeOverlap(4.0, gamma), digits = 3),
 recovered = estimateSpike(spikeLocation(4.0, gamma), gamma))
```

A seeded simulation agrees, within the fluctuation scale
`sqrt(spikeVariance(4, 0.5, 400)) ≈ 0.27` given by Paul's formula:

```@example spiked
using Random, LinearAlgebra # hide
Random.seed!(20261030) # hide
S = randSpikedCovariance(200, 400, [4.0])
round(eigmax(S), digits = 2)
```

## The rectangular case

For singular values of rectangular data the transition is at
``d > \gamma^{1/4}`` (Benaych-Georges & Nadakuditi).

```@docs
svdThreshold
svdSpikeLocation
svdSpikeOverlaps
```
