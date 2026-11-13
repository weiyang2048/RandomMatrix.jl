# Free Probability

Free additive (``\boxplus``) and multiplicative (``\boxtimes``)
convolution of probability measures, computed numerically. A
`FreeMeasure` is a discrete representation (nodes and weights) of a
probability measure; the Cauchy, R and S transforms are then exact
rational functions, and the convolutions are evaluated by the
subordination fixed point (additive case) and the S-transform product
rule (multiplicative case). Atoms of a convolution follow the exact
Bercovici–Voiculescu rules.

## Accuracy notes

`freePoissonLaw` is discretized on a Chebyshev grid, which resolves
the density spike at the lower Marčenko–Pastur edge for `gamma` near 1
(on a uniform grid the mean came out 2.3% high at `gamma = 0.99`, and
at `gamma = 1` the density at 0 is NaN, which poisoned every weight);
its mean and variance now match `1` and `gamma` to about `1e-7`.

For `freeMul`, measured against the exact moments of
`MP_gamma ⊠ MP_delta` (from the S-transform) at the default
resolution: with both factors bounded away from 0, the first three
moments err by at most about 2.5%. When a factor's support touches 0,
the product density diverges at 0, the Stieltjes smoothing (`eps`,
default half the mean grid spacing) leaks mass below 0 at a rate
`~ eps^(1/3)`, and the errors grow to about 5%
(`MP_1 ⊠ MP_0.5`) and 11% (`MP_1 ⊠ MP_1`). A smaller `eps` — down to
about `1e-4` times the support scale — improves the atom-free cases
substantially, but is not recommended for measures with an atom at 0,
where the Newton continuation fails near the lower edge.

## Worked example

The free Poisson (Marčenko–Pastur) law has mean ``1`` and variance
``\gamma``, reproduced here to the accuracy quoted above; the free
multiplicative convolution of two such laws has exact mean
``1 \cdot 1 = 1`` and exact second moment ``1 + \gamma + \delta``
(S-transform), hence variance ``\gamma + \delta = 0.5`` for
``\gamma = \delta = 0.25`` — the numerical convolution lands within
about 1% of both, as expected for factors bounded away from 0.

```@example freeprob
using RandomMatrix
fp = freePoissonLaw(0.5)
(mean = round(freeMean(fp), digits = 6), var = round(freeVar(fp), digits = 6))
```

```@example freeprob
prod = freeMul(freePoissonLaw(0.25), freePoissonLaw(0.25))
(mean = round(freeMean(prod), digits = 3), var = round(freeVar(prod), digits = 3))
```

Free additive convolution of two semicircles of variance 1 is a
semicircle of variance 2 (the R-transforms add):

```@example freeprob
ss = freeAdd(semicircleLaw(1.0), semicircleLaw(1.0))
(mean = round(freeMean(ss), digits = 3), var = round(freeVar(ss), digits = 3))
```

Free compression by a projection of trace ``\tau`` (the law of
``p a p`` in the compressed space, with the trace renormalized by
``1/\tau``) scales the R-transform as ``R(w) \mapsto R(\tau w)``;
in particular the Marčenko–Pastur law ``MP_\gamma`` compresses to
``MP_{\tau\gamma}`` — the limiting spectral law of a principal
submatrix retaining a ``\tau`` fraction of the coordinates. Here
``MP_{0.5}`` compressed by ``\tau = 1/2`` lands on the mean and
variance of ``MP_{0.25}``:

```@example freeprob
comp = freeCompress(freePoissonLaw(0.5), 0.5)
(mean = round(freeMean(comp), digits = 3), var = round(freeVar(comp), digits = 3))
```

## General populations: the Silverstein equation

For a sample covariance ``(1/n) T^{1/2} X X' T^{1/2}`` whose
population spectral distribution ``H`` is not a point mass,
`silversteinLaw` computes the limiting law ``F`` from the
Silverstein equation — the companion Stieltjes transform
``\underline m`` is the unique upper-half-plane solution of
``\underline m = -(z - \gamma \int t/(1 + t\underline m)\, dH(t))^{-1}``,
with ``\underline m(z) = -(1-\gamma)/z + \gamma m_F(z)`` recovering
the transform of ``F`` itself. The exact moments are
``\int x\,dF = \int t\,dH`` and ``\int x^2 dF = \int t^2 dH +
\gamma(\int t\,dH)^2``; below, a two-point population with mean
``2.5`` and second moment ``8.5`` at ``\gamma = 0.5`` has exact
(mean, variance) ``(2.5,\ 8.5 + 0.5 \cdot 2.5^2 - 2.5^2) =
(2.5,\ 5.375)``, and the solver lands within its calibrated ~0.3%
of both:

```@example freeprob
sg = silversteinLaw([1.0, 4.0], [0.5, 0.5], 0.5)
(mean = round(freeMean(sg), digits = 3), var = round(freeVar(sg), digits = 3))
```

At ``\gamma = 0.1`` the same population's support splits into two
clusters (the density at the former gap midpoint ``x = 1.95`` is
essentially zero, against ``0.74`` inside the first cluster):

```@example freeprob
sg2 = silversteinLaw([1.0, 4.0], [0.5, 0.5], 0.1)
round.(freeDensity(sg2, [0.9, 1.95, 4.0]), digits = 3)
## Free Bessel laws

`FreeBessel(s, t)` is the free Bessel law ``\pi_{s,t}`` of Banica,
Belinschi, Capitaine & Collins (Canad. J. Math. 63(1), 2011): the law
whose moment series satisfies ``f = 1 + z f^s (f + t - 1)``. At
``t = 1`` and integer ``s`` it is the limiting spectral distribution
of a product of ``s`` independent square Ginibre (equivalently square
Wishart) matrices — ``\pi_{2,1}`` is the free square of the
Marčenko–Pastur law — and its moments are the Fuss–Catalan numbers.
For ``t < 1`` it carries an atom ``1 - t`` at 0 and its support is
bounded away from 0. The parametrization is ``(s, t)`` as in the
paper, deliberately not a product depth and a Marčenko–Pastur ratio:
a product of ``k \ge 2`` laws ``\mathrm{MP}_\gamma`` with
``\gamma \ne 1`` is ``\pi_{s,t}`` for no ``(s, t)``. Exact moments
(`freeBesselMoments`, the Fuss–Narayana formula) and the support and
atom (`freeBesselSupport`) are separate queries; the constructor
returns the grid law itself, whose first four moments match the exact
ones to at most about 0.13% at the default resolution.

```@example freeprob
fb = FreeBessel(2, 1)             # product of two square Wisharts' limit
(mean = round(freeMean(fb), digits = 3), var = round(freeVar(fb), digits = 3))
```

```@example freeprob
freeBesselMoments(2, 1, 5)        # Fuss-Catalan: 1, 3, 12, 55, 273
```

```@example freeprob
freeAtoms(FreeBessel(2, 0.5))     # atom 1 - t at 0
```

## Variance profiles: the quadratic vector equation

For a Wigner matrix whose entry variances vary across the matrix,
``\mathbb{E}|X_{ij}|^2 = S_{ij}/n`` for a deterministic profile
``S``, the limiting Stieltjes transform is no longer a scalar: the
vector ``m(z)`` solves the quadratic vector equation
``-1/m_i = z + \frac{1}{n}\sum_j S_{ij} m_j`` on the upper half
plane, and the limiting density is the average of the component
densities (Ajanki–Erdős–Krüger). `profileWignerLaw` solves the QVE
and returns the law as a `FreeMeasure`; `randProfileWigner` samples
the ensemble. The constant profile is the semicircle; structured
profiles give laws no single classical object covers. Two
communities with within-variances `2` and `1` and cross-variance
`0.5` form a single cluster whose variance is the mean profile
entry, `(2 + 1 + 2\cdot 0.5)/4 = 1`:

```@example freeprob
mu = profileWignerLaw([100, 100], [2.0 0.5; 0.5 1.0])
round(freeVar(mu), digits = 4)
```

```@example freeprob
round.(freeDensity(mu, [-1.0, 0.0, 1.0]), digits = 3)
```

A block-diagonal profile is an exact mixture of semicircles, with
the full-size normalization folding the block fraction into the
radius: a block of variance `b` on a fraction `p` of the coordinates
contributes radius `2\sqrt{bp}` with weight `p`. Here
``0.6\,\mathrm{SC}(\sqrt{1.35}) + 0.4\,\mathrm{SC}(\sqrt{0.256})``:

```@example freeprob
mix = profileWignerLaw([60, 40], [2.25 0.0; 0.0 0.64])
round.(freeDensity(mix, [0.0, 0.5, 2.0]), digits = 3)
```

A coordinate with an identically zero profile row is an exact zero
eigenvalue of every realization; such rows leave the solve and come
back as an exact atom at `0`:

```@example freeprob
Sz = fill(1.0, 30, 30)
Sz[7, :] .= 0.0
Sz[:, 7] .= 0.0
freeAtoms(profileWignerLaw(Sz))
```

## Random features (Mei–Montanari)

For a random-features map ``\Phi = \sigma(W X / \sqrt d)`` with
``W`` (``m \times d``) and ``X`` (``d \times n``) i.i.d. Gaussian,
`randomFeaturesLaw` computes the limiting spectral law of the
feature covariance ``\Phi \Phi' / n``. The activation enters only
through its Hermite coefficients ``(\mu_0, \mu_1, \mu_\star)`` for
``Z \sim N(0,1)`` (`hermiteCoefficients`): the bulk law is the
Silverstein law above, with the affine Marchenko–Pastur population
``H = \mathrm{law}(\mu_1^2 T + \mu_\star^2)``, ``T \sim MP_{m/d}``,
at aspect ``\gamma = m/n`` — equivalently, the solution of the
Pennington–Worah quartic self-consistent equation. It reduces
exactly to a scaled Marchenko–Pastur law when ``\mu_1 = 0``, and to
the product-Wishart law ``MP_{m/d} \boxtimes MP_{m/n}`` for a linear
activation. ``\mu_0`` does not enter the bulk; it produces a single
outlier at ``\mu_0^2 m (1 + o(1))``. For ReLU features at ``n/d =
1.5``, ``m/d = 1.25`` the mean is ``\mu_1^2 + \mu_\star^2 = 0.3408``
(the solver lands ~0.6% above it at the default regularization):

```@example freeprob
F = randomFeaturesLaw(:relu, 1.5, 1.25)
(mean = round(freeMean(F), digits = 3), var = round(freeVar(F), digits = 3))
```

```@docs
FreeMeasure
semicircleLaw
freePoissonLaw
diracLaw
discreteLaw
empiricalLaw
FreeBessel
freeBesselMoments
freeBesselSupport
cauchyTransform
rTransform
sTransform
freeAdd
freeMul
freeCompress
silversteinLaw
profileWignerLaw
randProfileWigner
randomFeaturesLaw
hermiteCoefficients
freeDensity
freeAtoms
freeMean
freeVar
```
