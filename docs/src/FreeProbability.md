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
