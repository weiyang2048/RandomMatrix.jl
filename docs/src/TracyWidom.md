# Tracy–Widom Distributions

The Tracy–Widom law ``F_\beta`` is the limiting distribution of the
largest eigenvalue of a Gaussian ``\beta``-ensemble, and — after the
[`twScaling`](@ref) centering — of the largest eigenvalue of a Wishart
matrix. The distribution is computed from the Hastings–McLeod solution
of Painlevé II; published means are ``-1.20653`` (``\beta = 1``),
``-1.77109`` (``\beta = 2``) and ``-2.30688`` (``\beta = 4``).

```@docs
TracyWidom
twScaling
twTest
BetaHermite
BetaLaguerre
```

## Worked example

`TracyWidom` is a `Distributions` distribution, so `mean`, `var`,
`cdf` and `quantile` work as usual; the mean of ``F_2`` is the
published ``-1.77109`` quoted above, so its cdf there is close to
``1/2``.

```@example tw
using RandomMatrix, Distributions
d = TracyWidom(2)
(mean = round(mean(d), digits = 5), var = round(var(d), digits = 5),
 cdf_at_mean = round(cdf(d, mean(d)), digits = 3),
 q95 = round(quantile(d, 0.95), digits = 3))
```

For a Wishart matrix, `twScaling` gives the Johnstone centering and
scaling to apply to the largest eigenvalue before comparing with
``F_1`` (real data):

```@example tw
(mu, sigma) = twScaling(400, 100)
(mu = round(mu, digits = 3), sigma = round(sigma, digits = 4))
```

## Testing the largest eigenvalue

`twTest` wires the two ingredients together into Johnstone's test of
the largest sample eigenvalue against the pure-noise null (identity
population covariance): it returns the scaled statistic
``(\lambda - \mu)/\sigma`` and its upper-tail p-value under ``F_1``.
At ``n = 400``, ``p = 100`` an observed ``\lambda = 2.28`` is
borderline (p-value ``0.06``), while ``\lambda = 2.35`` is a clear
signal outlier:

```@example tw
r = twTest(2.28, 400, 100)
(statistic = round(r.statistic, digits = 3), pvalue = round(r.pvalue, digits = 3))
```

```@example tw
r = twTest(2.35, 400, 100)
(statistic = round(r.statistic, digits = 3), pvalue = round(r.pvalue, digits = 4))
```

## Sampling the β-ensembles

[`BetaHermite`](@ref) and [`BetaLaguerre`](@ref) sample the *full
spectrum* of a Gaussian / Wishart β-ensemble at any ``\beta > 0``,
from the Dumitriu–Edelman tridiagonal models — the same construction
behind the general-β `rand` of `TracyWidom`. On the ``1/\sqrt{n}``
scale the Hermite bulk follows the semicircle law on ``[-2, 2]``
(extremes near ``\pm 2``, scaled second moment near ``1``):

```@example tw
λ = rand(BetaHermite(2, 400))
(round(minimum(λ) / sqrt(400), digits = 2),
 round(maximum(λ) / sqrt(400), digits = 2),
 round(mean(λ .^ 2) / 400, digits = 2))
```

For the Laguerre ensemble the eigenvalues on the ``1/n`` scale
follow `MarchenkoPastur(p/n)` — here ``p/n = 1/4``, so the mean is
``1`` and the upper edge is ``(1 + 1/2)^2 = 2.25``:

```@example tw
μ = rand(BetaLaguerre(1, 400, 100)) ./ 400
(round(mean(μ), digits = 3), round(maximum(μ), digits = 2))
```
