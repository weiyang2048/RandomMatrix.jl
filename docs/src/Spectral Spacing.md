# Under Construction


```@docs
wignersurmise
```

```@docs
unfoldedeigvals
```

```@docs
unfoldedSpacings
```

```@docs
numberVariance
```

## Adjacent gap ratios

The adjacent gap ratio $r_n = \min(s_n, s_{n+1}) / \max(s_n, s_{n+1})$
of consecutive spacings needs no unfolding — the local level density
cancels in the ratio — so its mean is a one-number diagnostic of the
local statistics: $\langle r \rangle \approx 0.3863$ for uncorrelated
(Poisson) levels, $0.5307$ (GOE), $0.5996$ (GUE) and $0.6744$ (GSE)
(Atas, Bogomolny, Giraud & Roux, Phys. Rev. Lett. 110, 084101 (2013)).

```@docs
gapRatios
```

```@docs
meanGapRatio
```

```@docs
fitGapRatioBeta
```

```@docs
fitGapRatioBetaCI
```

```@example
using RandomMatrix, Random
E = rand(MersenneTwister(42), BetaHermite(2, 200))
meanGapRatio(E[41:160])   # bulk of a GUE spectrum: near the 0.5996 anchor
```

## Spectral form factor

The spectral form factor is the dynamical companion of the spacing
statistics above: where `numberVariance` probes level correlations
in the spectrum domain, `spectralFormFactor` probes them in the
time domain, through the dip–ramp–plateau structure of the
ensemble-averaged form factor. Both functions take a `beta`
keyword for the finite-temperature form factor
$|Z(\beta + it)|^2 / Z(\beta)^2$ (the default `beta = 0` is the
zero-temperature case).

```@docs
spectralFormFactor
```

```@docs
ensembleSpectralFormFactor
```

```@example
using RandomMatrix, Random
specs = [rand(MersenneTwister(42), BetaHermite(2, 100)) for _ in 1:50]
parts = ensembleSpectralFormFactor(specs, [0.5, 2.0, 6.3]; unfold=true)
parts.total   # decays, ramps, and approaches the plateau 1/100
```
