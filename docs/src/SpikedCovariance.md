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

## The rectangular case

For singular values of rectangular data the transition is at
``d > \gamma^{1/4}`` (Benaych-Georges & Nadakuditi).

```@docs
svdThreshold
svdSpikeLocation
svdSpikeOverlaps
```
