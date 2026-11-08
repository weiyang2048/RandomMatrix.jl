# Distributions

## Marchenko-Pastur 

```@docs
MarchenkoPastur
```

## Wachter Law

```@docs
Wachter
fitWachter
```

### Fitting the aspect ratios

`fitWachter` inverts the law's mean and variance formulas on an
observed MANOVA spectrum (method of moments — the analogue of
`fitMarchenkoPastur` for the Jacobi case). With ``m`` and ``v`` the
spectrum's mean and population variance and ``s = \gamma_1 + \gamma_2``,
the variance factors as ``v = s\,(m (1 - m))^2``, so
``s = v / (m (1 - m))^2``, ``\gamma_2 = m s``, ``\gamma_1 = (1 - m) s``
in closed form. Fitting 20,000 draws from `Wachter(0.5, 0.25)` recovers
the aspect ratios to two decimals:

```@example
using RandomMatrix, Random, Distributions
Random.seed!(20261107)
xs = rand(Wachter(0.5, 0.25), 20_000)
f = fitWachter(xs)
(γ₁ = round(f.γ₁, digits = 3), γ₂ = round(f.γ₂, digits = 3))
```

### Worked example

The symmetric case `Wachter(0.5, 0.5)` has mean ``1/2`` by the
``\gamma_2 / (\gamma_1 + \gamma_2)`` formula, and its cdf at the
mean is exactly ``1/2`` by the swap symmetry ``1 - X \sim``
`Wachter(γ₂, γ₁)`. With a rank-deficient numerator Wishart
(``\gamma_1 = 1.5 > 1``) a third of the mass sits in the atom at 0,
while the continuous part still carries the rest of the mean:

```@example
using RandomMatrix, Distributions
d = Wachter(0.5, 0.5)
(mean(d), var(d), round(cdf(d, 0.5), digits = 6))
```

```@example
using RandomMatrix, Distributions
d = Wachter(1.5, 0.5)
(pdf(d, 0), mean(d), round(cdf(d, 0.999), digits = 6)) # atom 1/3 at 0; no atom at 1
```

## Circular Law

```@docs
Circular
```
## Elliptic Law
```@docs
Elliptic
```

## Stieltjes Transform of the Semicircle Law
```@docs
stieltjes
```