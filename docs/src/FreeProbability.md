# Free Probability

Free additive (``\boxplus``) and multiplicative (``\boxtimes``)
convolution of probability measures, computed numerically. A
`FreeMeasure` is a discrete representation (nodes and weights) of a
probability measure; the Cauchy, R and S transforms are then exact
rational functions, and the convolutions are evaluated by the
subordination fixed point (additive case) and the S-transform product
rule (multiplicative case). Atoms of a convolution follow the exact
Bercovici–Voiculescu rules. The page also covers the package's first
non-Hermitian limiting law, the Brown measure of an R-diagonal
element, computed from the same S-transform machinery.

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
```

## Non-Hermitian: the Brown measure (single ring law)

For an R-diagonal element ``a`` — equivalently ``a = up`` with ``u``
a Haar unitary free from a positive ``p`` — the Brown measure
``\mu_a`` is the two-dimensional limiting eigenvalue distribution.
It is rotation invariant, so a radial law determines it, and
Haagerup & Larsen compute that law from ``\nu``, the distribution
of ``a^*a``, through its S-transform (J. Funct. Anal. 176 (2000),
331–367, Thm. 4.4, Cor. 4.5):
``\mu_a(B(0, S_\nu(t-1)^{-1/2})) = t`` for
``t \in (\nu(\{0\}), 1]``. The support is the annulus (the *single
ring*) with exact radii ``\sqrt{\int t\,d\nu}`` (outer) and
``\sqrt{1/\int t^{-1}\,d\nu}`` (inner, 0 when ``\nu`` has an atom
at 0 or ``\int t^{-1} d\nu = \infty``), and ``\mu_a`` carries an
atom at 0 of mass exactly ``\nu(\{0\})``. Guionnet, Krishnapur &
Zeitouni prove that the eigenvalues of ``A = UTV`` (``U, V`` Haar
unitary, ``T`` deterministic diagonal with ``T^2``'s spectral law
converging to ``\nu``, ``\|T\|`` bounded, no singular-value mass
escaping to 0) converge to this measure (Ann. of Math. 174(2)
(2011), 1189–1217).

`brownMeasure` takes ``\nu`` in the same three forms as
`silversteinLaw` and returns a `BrownMeasure` — the radii, the
atom at 0, and the radial CDF and planar density
``\rho(z) = \Phi'(|z|)/(2\pi|z|)`` on a grid — since a real-line
`FreeMeasure` cannot represent a planar law. Read it with
`radialCDF` and `brownDensity`. Boundary cases are exact:
``\nu = \delta_c`` gives the uniform measure on the circle of
radius ``\sqrt c`` (singular: the density grid is empty), and an
atom of ``\nu`` at 0 reappears as the atom at 0.

```@example freeprob
bm = brownMeasure([1.0, 4.0], [0.5, 0.5])
(round(bm.inner, digits = 4), round(bm.outer, digits = 4), radialCDF(bm, sqrt(2.0)))
```

The radii are ``\sqrt{1.6}`` and ``\sqrt{2.5}``, and the median
radius is exactly ``\sqrt 2``. Haagerup & Larsen's Bernoulli
example, ``\nu = \tfrac12\delta_0 + \tfrac12\delta_1``, has radial
CDF ``\tfrac12/(1-r^2)`` and density ``\tfrac12/(\pi(1-r^2)^2)``
on the disk of radius ``\sqrt{1/2}``:

```@example freeprob
bb = brownMeasure([0.0, 1.0], [0.5, 0.5])
(bb.atom0, round(radialCDF(bb, 0.4), digits = 4), round(brownDensity(bb, 0.4), digits = 4))
```

The circular element (``\nu = ``MP``_1``, via
`brownMeasure(freePoissonLaw(1.0))`) is the uniform unit disk:
radial CDF ``r^2`` to about ``10^{-7}`` and density ``1/\pi`` on
the bulk of the disk. Formed-matrix simulation (``A = UP`` at
``n = 500``) matches the radial CDF with pooled KS distance about
``0.03``, the deviation sitting in the finite-``n`` edge layers.

```@docs
FreeMeasure
semicircleLaw
freePoissonLaw
diracLaw
discreteLaw
empiricalLaw
cauchyTransform
rTransform
sTransform
freeAdd
freeMul
freeCompress
silversteinLaw
BrownMeasure
brownMeasure
radialCDF
brownDensity
freeDensity
freeAtoms
freeMean
freeVar
```
