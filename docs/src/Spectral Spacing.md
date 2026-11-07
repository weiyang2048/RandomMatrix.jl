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
