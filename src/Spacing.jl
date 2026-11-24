export wignersurmise, unfoldedeigvals, unfoldedSpacings, numberVariance, gapRatios, meanGapRatio

"""
```julia
wignersurmise(s ; beta)  
```
- `beta` : default `1`, 

    `1` for the spacing density corresponds to GOE(2) 

    ``p_1(s)=\\frac{\\pi s}{2} e^{-\\pi s^{2} / 4}``

    `2` for the spacing density corresponds to GUE(2) 

    ``p_2(s)=\\frac{32 s^{2}}{\\pi^{2}} e^{-4 s^{2} / \\pi}``

-  For more info see [Wigner surmise on Wikipedia](https://en.wikipedia.org/wiki/Wigner_surmise)


# Examples 
Evaluate Wigner surmise for GOE at `2`
```julia
wignersurmise(2)

0.1357605281502967
```

Evaluate Wigner surmise for GUE at `2`
```julia
wignersurmise(2,beta=2)

0.07962814352725955
```
"""
function wignersurmise(s::Real; beta=1::Int)
    if beta == 1
        return (pi / 2) * s * exp(-pi * s^2 / 4)
    elseif beta == 2
        return 32 * (s / pi)^2 * exp(-4 * s^2 / pi)
    end
end

"""
```julia
unfoldedeigvals(E; deg, goe)
```
- `E` : The array of eigenvalues
- `deg` : default `10`
- `goe` : default `false`
"""
function unfoldedeigvals(E::Array; deg=10::Int, goe=false::Bool)
    sort!(E)
    if goe
        return length(E) * cdf(Semicircle(2), E)
    end

    return polyfit(E, (x -> sum(E .< x)).(E), deg).(E)

end


"""
```julia
unfoldedSpacings(E; deg, goe)
```
- `E` : The array of eigenvalues
- `deg` : default `10`
- `goe` : default `false`
```
"""
function unfoldedSpacings(E::Array; deg=10::Int, goe=false::Bool)
    return diff(unfoldedeigvals(E, deg=deg, goe=goe))
end


"""
    numberVariance(E, l::Float64)

Number variance of the spectrum `E` at unfolded interval length `l`.
`E` is first unfolded (see [`unfoldedeigvals`](@ref)); for each unfolded
eigenvalue ``e`` let ``n_e(l)`` be the number of *other* unfolded
eigenvalues within distance ``l/2`` of ``e``. The return value is the
spectral average of ``(n_e(l) - l)^2`` — the empirical analogue of
``\\Sigma^2(l) = \\langle (n(l) - \\langle n(l)\\rangle)^2 \\rangle``,
which grows only logarithmically in ``l`` for rigid (random-matrix)
spectra and linearly for uncorrelated (Poisson) spectra.

# Examples
```julia
numberVariance(sort(randn(50)), 2.0)
```
"""
function numberVariance(E::Array, l::Float64)
    n = length(E)
    E = unfoldedeigvals(E)
    stat = 0
    for (i, e) in enumerate(E)
        n_e_l = 0
        temp1, temp2 = i - 1, i + 1
        while temp1 >= 1 && E[temp1] >= e - l / 2
            temp1 -= 1
            n_e_l += 1
        end
        while temp2 <= n && E[temp2] <= e + l / 2
            temp2 += 1
            n_e_l += 1
        end
        stat += (n_e_l - l)^2
    end

    return stat / n
end

"""
    gapRatios(E)

Adjacent gap ratios of the spectrum `E`: with the spacings
``s_n = E_{n+1} - E_n`` of the sorted eigenvalues,

```math
r_n = \\frac{\\min(s_n, s_{n+1})}{\\max(s_n, s_{n+1})} \\in [0, 1].
```

Unlike the spacing distribution or [`numberVariance`](@ref), the ratio
needs no unfolding: the local level density cancels between consecutive
spacings, so `gapRatios` applies directly to a raw spectrum. A ratio of
exactly `0` is returned where both spacings of a pair vanish (a triple
degeneracy). `E` is not mutated. See [`meanGapRatio`](@ref) for the
standard summary statistic and its exact reference values.

Reference: Atas, Bogomolny, Giraud & Roux, "Distribution of the ratio
of consecutive level spacings in random matrix ensembles", Phys. Rev.
Lett. 110, 084101 (2013).

# Examples
```julia
gapRatios([0.0, 1.0, 3.0, 4.0])   # spacings 1, 2, 1 -> [0.5, 0.5]
```
"""
function gapRatios(E::AbstractVector{<:Real})
    n = length(E)
    n >= 3 || throw(ArgumentError("gapRatios needs at least 3 eigenvalues, got $n"))
    all(isfinite, E) || throw(ArgumentError("gapRatios needs finite eigenvalues"))
    s = diff(sort!(collect(E)))
    return [iszero(max(s[i], s[i + 1])) ? 0.0 : min(s[i], s[i + 1]) / max(s[i], s[i + 1])
            for i in 1:(length(s) - 1)]
end

"""
    meanGapRatio(E)

Mean adjacent gap ratio ``\\langle r \\rangle`` of the spectrum `E`
(see [`gapRatios`](@ref)). Because the ratio needs no unfolding, this
single number is the standard first diagnostic for the local statistics
of a spectrum. Exact / large-``n`` reference values (Atas et al. 2013):

- uncorrelated (Poisson) levels: ``2 \\log 2 - 1 \\approx 0.3863``,
- GOE (``\\beta = 1``): ``\\approx 0.5307``,
- GUE (``\\beta = 2``): ``\\approx 0.5996``,
- GSE (``\\beta = 4``): ``\\approx 0.6744``.

Finite spectra sit slightly above these bulk values; compare bulk
(interior) eigenvalues for a sharp test.

# Examples
```julia
meanGapRatio([0.0, 1.0, 3.0, 4.0])   # 0.5
```
"""
meanGapRatio(E::AbstractVector{<:Real}) = sum(gapRatios(E)) / (length(E) - 2)
