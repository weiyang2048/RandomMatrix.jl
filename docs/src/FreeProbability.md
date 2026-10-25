# Free Probability

Free additive (``\boxplus``) and multiplicative (``\boxtimes``)
convolution of probability measures, computed numerically. A
`FreeMeasure` is a discrete representation (nodes and weights) of a
probability measure; the Cauchy, R and S transforms are then exact
rational functions, and the convolutions are evaluated by the
subordination fixed point (additive case) and the S-transform product
rule (multiplicative case). Atoms of a convolution follow the exact
Bercovici–Voiculescu rules.

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
