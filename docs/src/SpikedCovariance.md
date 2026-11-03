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

## The spiked MANOVA (Jacobi) case

The same transition governs the double-Wishart (MANOVA) matrix
``(A + B)^{-1} A``, whose bulk follows the [`Wachter`](@ref) law.
With ``A \sim W_p(\Sigma, n_1)``, ``B \sim W_p(I, n_2)`` and aspect
ratios ``\gamma_1 = p/n_1``, ``\gamma_2 = p/n_2``, a population spike
``\theta`` of ``\Sigma`` (bulk at 1) produces an outlier eigenvalue
only outside the critical window
``(\theta_-, \theta_+)`` with
``\theta_\pm = (1 \pm \rho)/(1 - \gamma_2)`` and
``\rho = \sqrt{\gamma_1 + \gamma_2 - \gamma_1\gamma_2}`` — the same
``\rho`` as in the Wachter support edges. Inside the window the spike
is absorbed and the sample eigenvalue sticks to the nearer Wachter
edge (Wang & Yao, Ann. Statist. 45(1), 2017, Thm. 3.1, mapped from
the Fisher matrix ``B^{-1}A`` to the MANOVA eigenvalues).

```@docs
manovaThreshold
manovaSpikeLocation
```

### Worked example

At ``\gamma_1 = \gamma_2 = 0.5`` the upper critical spike is
``\theta_+ \approx 3.73``: a spike of ``\theta = 6`` separates to
``0.943`` (above the Wachter upper edge ``x_+ \approx 0.933``), a
spike of ``\theta = 2`` is inside the window and sticks to that edge,
and a low spike of ``\theta = 0.1`` separates below the bulk at
``0.040``.

```@example manova
using RandomMatrix
(threshold = round(manovaThreshold(0.5, 0.5), digits = 3),
 outlier = round(manovaSpikeLocation(6.0, 0.5, 0.5), digits = 3),
 stuck = round(manovaSpikeLocation(2.0, 0.5, 0.5), digits = 3),
 lower = round(manovaSpikeLocation(0.1, 0.5, 0.5), digits = 3))
```
