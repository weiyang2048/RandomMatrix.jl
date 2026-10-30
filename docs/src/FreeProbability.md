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
freeDensity
freeAtoms
freeMean
freeVar
```
