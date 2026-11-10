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

## Testing the largest MANOVA root

For a double Wishart (MANOVA) problem, Roy's greatest-root test
asks the same question about the largest eigenvalue ``\theta`` of
``(A + B)^{-1} B``, with ``A \sim W_p(I, m)`` the error Wishart and
``B \sim W_p(I, n)`` the hypothesis Wishart. Johnstone (2008) showed
that the logit transform ``W = \log(\theta / (1 - \theta))`` is
Tracy–Widom after centering and scaling — the angles in the
constants carry ``-1/2`` / ``-1`` corrections, which make the
approximation second-order accurate:

```@docs
manovaTWScaling
manovaTWTest
```

At ``p = 50``, ``m = 100``, ``n = 50`` the logit-scale centering is
``\mu \approx 2.064`` with scale ``\sigma \approx 0.127``; an
observed root ``\theta = 0.88`` is consistent with noise
(p-value ``0.294``), while ``\theta = 0.90`` is a borderline signal
(p-value ``0.045``):

```@example tw
(mu, sigma) = manovaTWScaling(50, 100, 50)
(mu = round(mu, digits = 4), sigma = round(sigma, digits = 4))
```

```@example tw
r = manovaTWTest(0.88, 50, 100, 50)
(statistic = round(r.statistic, digits = 3), pvalue = round(r.pvalue, digits = 3))
```

```@example tw
r = manovaTWTest(0.90, 50, 100, 50)
(statistic = round(r.statistic, digits = 3), pvalue = round(r.pvalue, digits = 3))
```

One convention warning: the package's Wachter law and spiked
MANOVA functions describe ``(A + B)^{-1} A`` with ``A`` the
numerator Wishart on ``n_1`` degrees of freedom. The largest
eigenvalue ``x`` of that matrix is the greatest root with error
degrees of freedom ``n_2`` and hypothesis degrees of freedom
``n_1`` — test it as `manovaTWTest(x, p, n₂, n₁)`.

## The unbalanced Jacobi edge

For the Jacobi (MANOVA) matrix ``J = (A + B)^{-1} B`` of two
independent complex Wisharts, the largest eigenvalue still has
Tracy–Widom(2) fluctuations at the right edge — but in the
*unbalanced* regime ``n - p = b\, p``, ``m - p = a\, p^{1+\nu}``
(``\nu > 0``), where one Wishart is much fatter than the other, the
centering is not Johnstone's. Didier, Gkogkou, Lopez Santander &
McLaughlin (arXiv:2611.00577, 2026) give the finite-``p`` location
and scale explicitly, together with an explicit ``p^{-2/3}``
correction to the distribution function (their Theorem 3.1):

```@docs
jacobiTWScaling
jacobiTWCDF
jacobiTWThreshold
```

At ``p = 100``, ``a = b = 1``, ``\nu = 1/2`` the edge sits at
``\beta_p \approx 0.399`` with scale ``\sigma_p \approx 0.0071``:

```@example jacobi
using RandomMatrix, Distributions
ls = jacobiTWScaling(100, 1.0, 1.0, 0.5)
(location = round(ls.location, digits = 4), scale = round(ls.scale, digits = 5))
```

Theorem 3.1 refines ``P(\theta_{\max} \le \varphi_p(\chi_p^{-1}(s)))``
beyond the plain ``F_2(s)``: at ``s = -1`` the correction moves the
cdf from 0.8072 to 0.8086 (the term is genuinely of order
``p^{-2/3}``, and small), and the threshold map turns Tracy–Widom
quantiles into approximate quantiles of ``\theta_{\max}`` — the 95%
point is ``\approx 0.398``:

```@example jacobi
(plain = round(jacobiTWCDF(-1.0, 100, 1.0, 1.0, 0.5; corrected = false), digits = 4),
 corrected = round(jacobiTWCDF(-1.0, 100, 1.0, 1.0, 0.5), digits = 4),
 q95 = round(jacobiTWThreshold(quantile(TracyWidom(2), 0.95), 100, 1.0, 1.0, 0.5), digits = 4))
```

One warning, also spelled out in the docstring of
[`jacobiTWScaling`](@ref): the paper's Corollary 3.2 gives a
closed-form *limiting* scale as well, but it agrees with the
finite-``p`` scale only to ``1 + o(1)`` with a slowly vanishing
error — at ``\nu = 1/2`` it runs 1.6–2.4 times too wide at
``p = 50``–``200``. The package exposes only the finite-``p``
scale.

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
