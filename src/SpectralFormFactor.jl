export spectralFormFactor, ensembleSpectralFormFactor

# Spectral form factor diagnostics, beside the spacing statistics in
# Spacing.jl. The single-spectrum quantity is computed on the levels
# exactly as given; the package's existing unfolding
# (unfoldedeigvals) is applied only on request via the `unfold`
# keyword, under the same keyword arguments numberVariance uses.

function _sff_levels(E::AbstractVector{<:Real}, unfold::Bool, deg::Int, goe::Bool)
    isempty(E) && throw(ArgumentError("spectralFormFactor requires a non-empty spectrum"))
    levels = float.(collect(E))
    if unfold
        levels = collect(unfoldedeigvals(levels, deg=deg, goe=goe))
    end
    return levels
end

function _sff_amplitude(levels::AbstractVector{<:Real}, t::Real)
    z = zero(ComplexF64)
    for e in levels
        z += cis(-t * e)
    end
    return z
end

"""
    spectralFormFactor(E, t; unfold=false, deg=10, goe=false)

Spectral form factor of the spectrum `E` at time `t`,

```math
g(t) = \\frac{1}{N^2} \\left| \\sum_{j=1}^{N} e^{-i t E_j} \\right|^2 ,
```

with `N = length(E)`. `t` may be a single real number (returns a
`Float64`) or a vector of times (returns a `Vector{Float64}`).

The levels are used *exactly as given*: no unfolding is applied
unless `unfold=true`, in which case the spectrum is first passed
through [`unfoldedeigvals`](@ref) (with its `deg` and `goe`
keywords, on a copy — `E` is never mutated). With this
``1/N^2`` normalization ``g(0) = 1`` and the late-time plateau of an
unfolded spectrum sits at ``1/N``. The dip–ramp–plateau structure of
random-matrix spectra — an early decay, a dip below the plateau, a
roughly linear ramp, and saturation at ``1/N`` beyond the
Heisenberg time — is a property of the *ensemble-averaged* form
factor of unfolded levels; see [`ensembleSpectralFormFactor`](@ref).
For uncorrelated (Poisson) levels the ensemble average instead
decays straight to the plateau ``1/N`` and stays there: no dip, no
ramp.

This is the zero-temperature (``\\beta = 0``) form factor of Cotler
et al., "Black holes and random matrices", J. High Energy Phys.
2017(5):118, arXiv:1611.04650. The finite-temperature form factor is
not implemented.

# Examples
```julia
spectralFormFactor([0.0, 1.0], π)   # the two phases cancel: 0
spectralFormFactor(rand(MersenneTwister(1), BetaHermite(2, 50)), 0.5)
```
"""
function spectralFormFactor(E::AbstractVector{<:Real}, t::Real;
        unfold::Bool=false, deg::Int=10, goe::Bool=false)
    levels = _sff_levels(E, unfold, deg, goe)
    return abs2(_sff_amplitude(levels, t)) / length(levels)^2
end

function spectralFormFactor(E::AbstractVector{<:Real}, ts::AbstractVector{<:Real};
        unfold::Bool=false, deg::Int=10, goe::Bool=false)
    levels = _sff_levels(E, unfold, deg, goe)
    n2 = length(levels)^2
    return [abs2(_sff_amplitude(levels, t)) / n2 for t in ts]
end

"""
    ensembleSpectralFormFactor(spectra, t; unfold=false, deg=10, goe=false)

Ensemble-averaged spectral form factor of a collection of spectra,
with the disconnected/connected decomposition of Cotler et al.
(J. High Energy Phys. 2017(5):118).

`spectra` is a vector of spectra, all of the same length `N` (for
example, repeated draws from [`BetaHermite`](@ref)). Writing
``Z_s(t) = \\sum_j e^{-i t E_{sj}}`` for the amplitude of spectrum
`s`, the return value is the `NamedTuple`

```julia
(total = ..., connected = ..., disconnected = ...)
```

where `total` is the ensemble mean of the single-spectrum form
factor [`spectralFormFactor`](@ref), ``total = \\langle |Z(t)|^2
\\rangle / N^2``; `disconnected` is the part carried by the average
density alone, ``|\\langle Z(t) \\rangle|^2 / N^2``; and `connected`
is the remainder, `total - disconnected`, which isolates the
two-level correlations — for unfolded random-matrix spectra it is
the ramp, rising from near zero to the plateau ``1/N`` at the
Heisenberg time, while for uncorrelated levels it sits at ``1/N``
at all times. With a single spectrum the decomposition is trivial:
`disconnected == total` and `connected == 0`.

As in [`spectralFormFactor`](@ref), the levels are used as given
unless `unfold=true`, which first applies
[`unfoldedeigvals`](@ref) to each spectrum.
"""
function ensembleSpectralFormFactor(spectra::AbstractVector{<:AbstractVector{<:Real}},
        t::Real; unfold::Bool=false, deg::Int=10, goe::Bool=false)
    parts = _ensemble_sff_parts(spectra, [t], unfold, deg, goe)
    return (total=parts[1][1], connected=parts[2][1], disconnected=parts[3][1])
end

function ensembleSpectralFormFactor(spectra::AbstractVector{<:AbstractVector{<:Real}},
        ts::AbstractVector{<:Real}; unfold::Bool=false, deg::Int=10, goe::Bool=false)
    parts = _ensemble_sff_parts(spectra, ts, unfold, deg, goe)
    return (total=parts[1], connected=parts[2], disconnected=parts[3])
end

function _ensemble_sff_parts(spectra, ts, unfold, deg, goe)
    isempty(spectra) &&
        throw(ArgumentError("ensembleSpectralFormFactor requires at least one spectrum"))
    levels = [_sff_levels(E, unfold, deg, goe) for E in spectra]
    n = length(first(levels))
    all(l -> length(l) == n, levels) || throw(ArgumentError(
        "ensembleSpectralFormFactor requires spectra of equal length, got lengths " *
        string(length.(levels))))
    n2 = n^2
    total = zeros(Float64, length(ts))
    disconnected = zeros(Float64, length(ts))
    for (k, t) in enumerate(ts)
        zsum = zero(ComplexF64)
        zsq = 0.0
        for l in levels
            z = _sff_amplitude(l, t)
            zsum += z
            zsq += abs2(z)
        end
        m = length(levels)
        total[k] = zsq / m / n2
        disconnected[k] = abs2(zsum / m) / n2
    end
    return total, total .- disconnected, disconnected
end
