# Distributions

## Marchenko-Pastur 

```@docs
MarchenkoPastur
```

## Wachter Law

```@docs
Wachter
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