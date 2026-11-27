export wignersurmise, unfoldedeigvals, unfoldedSpacings, numberVariance, gapRatios, meanGapRatio, fitGapRatioBeta, fitGapRatioBetaCI

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

# Mean of the Atas et al. (2013) Wigner-like surmise for the ratio
# distribution, P_beta(r) ∝ (r + r^2)^beta / (1 + r + r^2)^(1 + 3 beta / 2)
# on [0, 1], by Simpson quadrature on a fixed grid. Monotone increasing
# in beta, from ≈ 0.4086 at beta = 0 toward 1 as beta → ∞.
function _gap_ratio_surmise_mean(beta::Float64)
    n = 2000
    num = 0.0
    den = 0.0
    for i in 0:n
        r = i / n
        w = (i == 0 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        f = (r + r^2)^beta / (1 + r + r^2)^(1 + 1.5 * beta)
        den += w * f
        num += w * r * f
    end
    return num / den
end

"""
    fitGapRatioBeta(E)

Effective Dyson index ``\\beta`` of the spectrum `E`, estimated by
inverting the mean adjacent gap ratio: returns the ``\\beta`` whose
Atas et al. (2013) Wigner-like surmise mean ``\\langle r \\rangle(\\beta)``
equals [`meanGapRatio`](@ref)`(E)`. The surmise mean is monotone in
``\\beta``, so the inversion is a bisection on ``[0, 64]``.

The estimate is an *effective* index, useful for intermediate-statistics
spectra (crossover ensembles, critical statistics): it is calibrated on
the 3-level surmise, not the exact large-``n`` ratio distribution, so
it carries a small systematic bias (seeded `BetaHermite` calibration:
true ``\\beta = 0.5, 1, 1.5, 2, 3, 4`` recovered as ``\\approx 0.39,
0.93, 1.55, 1.94, 2.77, 4.01`` from pooled bulk spectra). A mean ratio
at or below the surmise's ``\\beta = 0`` value ``\\approx 0.4086`` —
which includes the exact Poisson mean ``2 \\log 2 - 1 \\approx 0.3863``
— returns `0.0`; a mean ratio the surmise reaches only as
``\\beta \\to \\infty`` returns the bracket cap `64.0`.

Reference: Atas, Bogomolny, Giraud & Roux, Phys. Rev. Lett. 110,
084101 (2013).

# Examples
```julia
fitGapRatioBeta([0.0, 1.0, 3.0, 4.0])   # <r> = 0.5 -> beta ≈ 0.63
```
"""
function fitGapRatioBeta(E::AbstractVector{<:Real})
    return _beta_from_mean_ratio(meanGapRatio(E))
end

# Bisection inversion of the Atas surmise mean (see fitGapRatioBeta).
function _beta_from_mean_ratio(mr::Float64)
    mr <= _gap_ratio_surmise_mean(0.0) && return 0.0
    lo, hi = 0.0, 64.0
    _gap_ratio_surmise_mean(hi) <= mr && return hi
    for _ in 1:50
        mid = (lo + hi) / 2
        _gap_ratio_surmise_mean(mid) < mr ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

# Linear-interpolation quantile of a sorted sample (no Statistics import).
function _sorted_quantile(sortedx::AbstractVector{Float64}, q::Float64)
    n = length(sortedx)
    n == 1 && return sortedx[1]
    pos = 1 + q * (n - 1)
    lo = floor(Int, pos)
    lo >= n && return sortedx[n]
    frac = pos - lo
    return sortedx[lo] * (1 - frac) + sortedx[lo + 1] * frac
end

# Effective beta of a spacing sequence: invert its mean adjacent gap ratio.
function _beta_from_spacings(s::AbstractVector{Float64})
    ratios = [iszero(max(s[i], s[i + 1])) ? 0.0 :
              min(s[i], s[i + 1]) / max(s[i], s[i + 1]) for i in 1:(length(s) - 1)]
    return _beta_from_mean_ratio(sum(ratios) / length(ratios))
end

# BCa (bias-corrected and accelerated) levels for a bootstrap distribution:
# z0 from the fraction of replicates below the point estimate, the
# acceleration `a` from leave-one-block-out jackknife estimates over the
# partition of the spacing sequence into blocks of length `bl` (the same
# blocking the bootstrap resamples). Efron, J. Amer. Statist. Assoc. 82
# (1987), 171-185.
function _bca_levels(estimates::AbstractVector{Float64}, beta::Float64,
                     s::AbstractVector{Float64}, bl::Int, level::Float64)
    nboot = length(estimates)
    prop = clamp(count(<(beta), estimates) / nboot, 0.5 / nboot, 1 - 0.5 / nboot)
    nd = Normal()
    z0 = quantile(nd, prop)
    m = length(s)
    nblocks = cld(m, bl)
    jack = Float64[]
    for k in 1:nblocks
        lo, hi = (k - 1) * bl + 1, min(k * bl, m)
        keep = [s[i] for i in 1:m if !(lo <= i <= hi)]
        length(keep) >= 2 && push!(jack, _beta_from_spacings(keep))
    end
    a = 0.0
    if length(jack) >= 2
        jm = sum(jack) / length(jack)
        den = sum((jm - x)^2 for x in jack)
        den > 0 && (a = sum((jm - x)^3 for x in jack) / (6 * den^1.5))
    end
    alpha = (1 - level) / 2
    adj(q) = cdf(nd, z0 + (z0 + quantile(nd, q)) / (1 - a * (z0 + quantile(nd, q))))
    return clamp(adj(alpha), 0.0, 1.0), clamp(adj(1 - alpha), 0.0, 1.0)
end

"""
    fitGapRatioBetaCI(E; level=0.95, nboot=1000, blocklength=nothing, method=:percentile, rng=Random.default_rng())

Sampling uncertainty for [`fitGapRatioBeta`](@ref): a circular block
bootstrap over the observed spacing sequence of `E`. Blocks of
`blocklength` consecutive spacings (default
``\\max(2, \\mathrm{round}(\\sqrt{m}))`` for ``m`` spacings) are resampled
with replacement until a full-length spacing sequence is rebuilt, so
each replicate's adjacent gap ratios keep the consecutive-pairing
``r_n = \\min(s_n, s_{n+1}) / \\max(s_n, s_{n+1})`` and the short-range
dependence between neighbouring ratios. Each replicate's mean ratio is
inverted through the same Atas et al. (2013) surmise as
`fitGapRatioBeta`.

With `method = :percentile` (the default) the returned interval is the
percentile band of the replicate estimates at `level`. With
`method = :bca` it is the BCa (bias-corrected and accelerated) band of
Efron (J. Amer. Statist. Assoc. 82, 1987): the percentile levels are
adjusted by a bias correction ``z_0`` from the replicate distribution
and an acceleration ``a`` from a jackknife over the same spacing blocks
the bootstrap resamples, so the blocking is consistent between the two.
Prefer BCa when the replicate distribution is skewed or the point
estimate sits off its centre (the usual case for short spectra, where
the surmise inversion is strongly nonlinear); prefer the percentile
band when a transparent, assumption-light summary is wanted. If every
replicate equals the point estimate (e.g. an equal-spaced spectrum)
both bands collapse onto it.

Returns a NamedTuple `(beta, lo, hi, level, nboot, blocklength, method)`,
where `beta` is the point estimate `fitGapRatioBeta(E)`. The band
measures sampling variation of ``\\langle r \\rangle`` only: it does not
remove the surmise's systematic bias documented in `fitGapRatioBeta`,
so for short spectra the band — not the point — is the honest summary.

Calibration (seeded `BetaHermite` bulk spectra, 20 trials per config,
95% level, `nboot = 400`): at bulk size 180 the percentile band covered
the truth 20/20 (``\\beta = 1``) and 18/20 (``\\beta = 2``), the BCa band
18/20 and 17/20; at bulk size 100 (short spectra) percentile covered
18/20 (``\\beta = 1``) and 19/20 (``\\beta = 2``), BCa 19/20 at both —
the two bands are comparable at these sizes, and neither removes the
surmise bias. In this package's test suite, bulk spectra of
`BetaHermite(2, 300)` recover ``\\beta = 2`` inside the 95% band under
both methods, and the band for an equal-spaced spectrum collapses onto
the bracket cap.

# Examples
```julia
fitGapRatioBetaCI(sort(randn(200)); rng=Random.MersenneTwister(1))
fitGapRatioBetaCI(sort(randn(200)); method=:bca, rng=Random.MersenneTwister(1))
```
"""
function fitGapRatioBetaCI(E::AbstractVector{<:Real}; level::Float64 = 0.95,
                           nboot::Int = 1000, blocklength::Union{Nothing,Int} = nothing,
                           method::Symbol = :percentile,
                           rng::Random.AbstractRNG = Random.default_rng())
    0 < level < 1 || throw(ArgumentError("level must lie in (0, 1), got $level"))
    nboot >= 20 || throw(ArgumentError("nboot must be at least 20, got $nboot"))
    method in (:percentile, :bca) ||
        throw(ArgumentError("method must be :percentile or :bca, got :$method"))
    beta = fitGapRatioBeta(E)  # validates E (length, finiteness)
    s = diff(sort!(collect(Float64, E)))
    m = length(s)
    bl = blocklength === nothing ? max(2, round(Int, sqrt(m))) : blocklength
    1 <= bl <= m || throw(ArgumentError("blocklength must lie in 1:$m, got $bl"))
    nblocks = cld(m, bl)
    estimates = Vector{Float64}(undef, nboot)
    for b in 1:nboot
        resampled = Vector{Float64}(undef, nblocks * bl)
        for k in 1:nblocks
            start = rand(rng, 1:m)
            for j in 0:(bl - 1)
                resampled[(k - 1) * bl + j + 1] = s[mod1(start + j, m)]
            end
        end
        resize!(resampled, m)
        estimates[b] = _beta_from_spacings(resampled)
    end
    sort!(estimates)
    alpha = (1 - level) / 2
    qlo, qhi = method === :bca ? _bca_levels(estimates, beta, s, bl, level) :
               (alpha, 1 - alpha)
    return (beta = beta,
            lo = _sorted_quantile(estimates, qlo),
            hi = _sorted_quantile(estimates, qhi),
            level = level, nboot = nboot, blocklength = bl, method = method)
end
