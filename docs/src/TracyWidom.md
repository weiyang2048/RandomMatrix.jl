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
