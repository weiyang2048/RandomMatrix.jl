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
dTransform
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

## The general population case

The bulk need not sit at a single point. When the population
eigenvalues have a general limiting distribution ``H`` (the setting
of [`silversteinLaw`](@ref)), Baik & Silverstein (2006) give the
transition through the *D-transform*
``\mathcal D(\ell) = \ell \left(1 + \gamma \int t \, / \, (\ell - t) \, dH(t)\right)``:
a spike ``\ell`` produces an outlier at ``\mathcal D(\ell)`` exactly
when ``\ell`` exceeds the unique minimizer of ``\mathcal D`` — the
generalized `bbpThreshold` — and the squared eigenvector overlap is
``\ell \mathcal D'(\ell) / \mathcal D(\ell)``. The four functions
above accept `H` in the same three input forms as `silversteinLaw`:
discrete points/masses, a `FreeMeasure`, or a `Distributions` law.
At ``H = \delta_{\sigma^2}`` they reduce exactly to the scalar
formulas of the previous section.

For a population split evenly between ``1`` and ``4`` at
``\gamma = 0.5``, the threshold is ``\ell_c \approx 6.01`` — a spike
must clear not just the population bulk (top at ``4``) but the
D-transform minimum — and a spike of ``10`` separates to
``11.94``, while a spike of ``2`` is absorbed at the bulk edge
``9.30`` (the upper edge of `silversteinLaw` for this ``H``):

```@example spiked
Hp, Hm = [1.0, 4.0], [0.5, 0.5]
(threshold = round(bbpThreshold(Hp, Hm, gamma), digits = 3),
 loc = round(spikeLocation(10.0, Hp, Hm, gamma), digits = 3),
 overlap = round(spikeOverlap(10.0, Hp, Hm, gamma), digits = 3),
 recovered = round(estimateSpike(spikeLocation(10.0, Hp, Hm, gamma), Hp, Hm, gamma), digits = 6),
 stuck = round(spikeLocation(2.0, Hp, Hm, gamma), digits = 3))
```

A seeded simulation with a formed population covariance (bulk
eigenvalues ``1`` and ``4``, one spike of ``10``) agrees:

```@example spiked
using Random, LinearAlgebra # hide
Random.seed!(20261113) # hide
p, n = 400, 800
evals = vcat(fill(1.0, 200), fill(4.0, 199), [10.0])
Q = Matrix(qr(randn(p, p)).Q)
A = Q * Diagonal(sqrt.(evals)) * Q'
X = A * randn(p, n) / sqrt(n)
round(eigmax(Symmetric(X * X')), digits = 2)
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
manovaSpikeVariance
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

Above the threshold the outlier's fluctuations are Gaussian at the
``1/p`` scale (Dharmawansa, Johnstone & Onatski 2014, Prop. 11, pushed
through the map from the Fisher matrix to the MANOVA eigenvalues), so
at ``p = 100`` the spike of ``\theta = 6`` above fluctuates with
standard deviation ``\sqrt{\mathrm{manovaSpikeVariance}} \approx 0.0076``:

```@example manova
round(sqrt(manovaSpikeVariance(6.0, 0.5, 0.5, 100)), digits = 4)
```

## The spiked Wigner case

The additive analogue: ``M = \sigma W/\sqrt{n} + A`` with ``W`` a
Wigner matrix (bulk the semicircle law on ``[-2\sigma, 2\sigma]``) and
``A`` a finite-rank deformation. A spike ``\theta`` of ``A`` produces
an outlier eigenvalue only above the threshold ``|\theta| > \sigma``,
at ``\theta + \sigma^2/\theta``, with eigenvector overlap
``1 - \sigma^2/\theta^2`` (Capitaine, Donati-Martin & Féral,
Ann. Probab. 37(1), 2009, Thm. 2.1). Below the threshold the spike
sticks to the nearer semicircle edge.

```@docs
wignerThreshold
wignerSpikeLocation
wignerSpikeOverlap
wignerEstimateSpike
wignerSpikeVariance
randSpikedWigner
```

### Worked example

At ``\sigma = 1`` a spike of ``\theta = 2`` is above the threshold, so
it appears as an outlier near `wignerSpikeLocation(2) = 2.5` with
eigenvector overlap `0.75`; a spike of ``\theta = 0.7`` is below the
threshold and sticks to the semicircle edge ``2``.

```@example wigner
using RandomMatrix
(threshold = wignerThreshold(), loc = wignerSpikeLocation(2.0),
 overlap = wignerSpikeOverlap(2.0),
 recovered = wignerEstimateSpike(wignerSpikeLocation(2.0)),
 stuck = wignerSpikeLocation(0.7))
```

A seeded simulation agrees, within the fluctuation scale
`sqrt(wignerSpikeVariance(2, 400)) ≈ 0.061`:

```@example wigner
using Random, LinearAlgebra # hide
Random.seed!(20261113) # hide
M = randSpikedWigner(400, [2.0])
round(eigmax(M), digits = 2)
```

The outlier's fluctuations are Gaussian at the ``1/\sqrt{n}`` scale
with variance ``2\sigma^2(1 - \sigma^2/\theta^2)/n`` for real data —
*unless* the spike points along a coordinate direction and the entries
are non-Gaussian: the limiting law is then non-universal and acquires a
fourth-cumulant term (Capitaine-Donati-Martin-Féral 2009, Thm. 2.2),
exposed through the `m4` keyword of `wignerSpikeVariance`. For
Rademacher entries (``m_4 = \sigma^4``) at ``\theta = 2`` the
coordinate-direction variance is `1.21875 / n`, against the Gaussian
`1.5 / n`:

```@example wigner
(gaussian = wignerSpikeVariance(2.0, 1), rademacher_e1 = wignerSpikeVariance(2.0, 1; m4 = 1.0))
```
