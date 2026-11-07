export spectralFormFactor, ensembleSpectralFormFactor

# Spectral form factor diagnostics, beside the spacing statistics in
# Spacing.jl. The single-spectrum quantity is computed on the levels
# exactly as given; the package's existing unfolding
# (unfoldedeigvals) is applied only on request via the `unfold`
# keyword, under the same keyword arguments numberVariance uses.
#
# Finite temperature (the `beta` keyword): the amplitude of a spectrum
# is the normalized analytically continued partition function
#     Z(beta + i t) / Z(beta),   Z(beta + i t) = sum_j exp(-(beta + i t) E_j),
# so the form factor is the thermal weight sum_j w_j cis(-t E_j) with
# w_j = exp(-beta E_j), normalized by sum_j w_j. Energies are shifted
# by a constant before exponentiating (the shift cancels exactly in
# the normalized ratio) so that large |beta * E| cannot over/underflow.

function _sff_levels(E::AbstractVector{<:Real}, unfold::Bool, deg::Int, goe::Bool)
    isempty(E) && throw(ArgumentError("spectralFormFactor requires a non-empty spectrum"))
    levels = float.(collect(E))
    if unfold
        levels = collect(unfoldedeigvals(levels, deg=deg, goe=goe))
    end
    return levels
end

function _sff_check_beta(beta::Real)
    (isfinite(beta) && beta >= 0) || throw(ArgumentError(
        "spectral form factor requires a finite inverse temperature beta >= 0, got beta = $beta"))
    return float(beta)
end

# Thermal weights and phases of `levels` at time `t`, with the energies
# shifted by `shift` (a constant that cancels in every normalized
# ratio below; for an ensemble it must be the SAME shift for every
# spectrum, or the disconnected part's phases would not line up).
# Returns the normalized amplitude sum_j w_j cis(-t (E_j - shift)) /
# sum_j w_j, with w_j = exp(-beta (E_j - shift)).
function _sff_amplitude(levels::AbstractVector{<:Real}, t::Real, beta::Float64, shift::Real)
    z = zero(ComplexF64)
    wsum = 0.0
    for e in levels
        w = exp(-beta * (e - shift))
        wsum += w
        z += w * cis(-t * (e - shift))
    end
    return z / wsum
end

"""
    spectralFormFactor(E, t; beta=0.0, unfold=false, deg=10, goe=false)

Spectral form factor of the spectrum `E` at time `t` and inverse
temperature `beta`,

```math
g(t, \\beta) = \\frac{|Z(\\beta + i t)|^2}{Z(\\beta)^2}, \\qquad
Z(\\beta + i t) = \\sum_{j=1}^{N} e^{-(\\beta + i t) E_j},
```

with `N = length(E)`. `t` may be a single real number (returns a
`Float64`) or a vector of times (returns a `Vector{Float64}`).

At the default `beta = 0` this is the zero-temperature form factor
``|\\sum_j e^{-i t E_j}|^2 / N^2``. At `beta > 0` the levels carry
thermal weights ``w_j = e^{-\\beta E_j}``: ``g(0, \\beta) = 1`` still
holds, but the late-time plateau — the time average of `g`, for
distinct levels — rises from ``1/N`` to the participation ratio
``\\sum_j w_j^2 / (\\sum_j w_j)^2`` of the weights, and the dip below
the plateau becomes shallower as fewer levels effectively
contribute. `beta` is in inverse units of the levels as given (after
unfolding, if requested — unfolded levels run to ``O(N)``, so a
comparable thermal effect there needs `beta` of order ``1/N``).
`beta` must be finite and nonnegative.

The levels are used *exactly as given*: no unfolding is applied
unless `unfold=true`, in which case the spectrum is first passed
through [`unfoldedeigvals`](@ref) (with its `deg` and `goe`
keywords, on a copy — `E` is never mutated). The dip–ramp–plateau
structure of random-matrix spectra — an early decay, a dip below the
plateau, a roughly linear ramp, and saturation at the plateau beyond
the Heisenberg time — is a property of the *ensemble-averaged* form
factor of unfolded levels; see [`ensembleSpectralFormFactor`](@ref).
For uncorrelated (Poisson) levels the ensemble average instead
decays straight to the plateau and stays there: no dip, no ramp.

This is the spectral form factor of Cotler et al., "Black holes and
random matrices", J. High Energy Phys. 2017(5):118,
arXiv:1611.04650, whose ``\\beta = 0`` case the original version of
this function computed.

# Examples
```julia
spectralFormFactor([0.0, 1.0], π)              # the two phases cancel: 0
spectralFormFactor([0.0, 1.0], π; beta=1.0)    # ((1 - e^-1)/(1 + e^-1))^2 ≈ 0.2136
spectralFormFactor(rand(MersenneTwister(1), BetaHermite(2, 50)), 0.5)
```
"""
function spectralFormFactor(E::AbstractVector{<:Real}, t::Real;
        beta::Real=0.0, unfold::Bool=false, deg::Int=10, goe::Bool=false)
    β = _sff_check_beta(beta)
    levels = _sff_levels(E, unfold, deg, goe)
    return abs2(_sff_amplitude(levels, t, β, minimum(levels)))
end

function spectralFormFactor(E::AbstractVector{<:Real}, ts::AbstractVector{<:Real};
        beta::Real=0.0, unfold::Bool=false, deg::Int=10, goe::Bool=false)
    β = _sff_check_beta(beta)
    levels = _sff_levels(E, unfold, deg, goe)
    shift = minimum(levels)
    return [abs2(_sff_amplitude(levels, t, β, shift)) for t in ts]
end

"""
    ensembleSpectralFormFactor(spectra, t; beta=0.0, unfold=false, deg=10, goe=false)

Ensemble-averaged spectral form factor of a collection of spectra at
inverse temperature `beta` (see [`spectralFormFactor`](@ref) for the
definition and the role of `beta`), with the
disconnected/connected decomposition of Cotler et al. (J. High
Energy Phys. 2017(5):118).

`spectra` is a vector of spectra, all of the same length `N` (for
example, repeated draws from [`BetaHermite`](@ref)). Writing
``A_s(t) = Z_s(\\beta + i t) / Z_s(\\beta)`` for the normalized
amplitude of spectrum `s` — each spectrum is normalized by its own
``Z_s(\\beta)``, so at `beta = 0` this is exactly the original
zero-temperature decomposition — the return value is the
`NamedTuple`

```julia
(total = ..., connected = ..., disconnected = ...)
```

where `total` is the ensemble mean of the single-spectrum form
factor [`spectralFormFactor`](@ref), ``total = \\langle |A(t)|^2
\\rangle``; `disconnected` is the part carried by the average
density alone, ``|\\langle A(t) \\rangle|^2``; and `connected` is
the remainder, `total - disconnected`, which isolates the two-level
correlations — for unfolded random-matrix spectra at `beta = 0` it
is the ramp, rising from near zero to the plateau ``1/N`` at the
Heisenberg time, while for uncorrelated levels it sits at ``1/N``
at all times. At `t = 0` every amplitude is 1, so `total ==
disconnected == 1` and `connected == 0`, at any `beta`. With a
single spectrum the decomposition is trivial: `disconnected ==
total` and `connected == 0`.

As in [`spectralFormFactor`](@ref), the levels are used as given
unless `unfold=true`, which first applies
[`unfoldedeigvals`](@ref) to each spectrum.
"""
function ensembleSpectralFormFactor(spectra::AbstractVector{<:AbstractVector{<:Real}},
        t::Real; beta::Real=0.0, unfold::Bool=false, deg::Int=10, goe::Bool=false)
    parts = _ensemble_sff_parts(spectra, [t], _sff_check_beta(beta), unfold, deg, goe)
    return (total=parts[1][1], connected=parts[2][1], disconnected=parts[3][1])
end

function ensembleSpectralFormFactor(spectra::AbstractVector{<:AbstractVector{<:Real}},
        ts::AbstractVector{<:Real}; beta::Real=0.0, unfold::Bool=false, deg::Int=10, goe::Bool=false)
    parts = _ensemble_sff_parts(spectra, ts, _sff_check_beta(beta), unfold, deg, goe)
    return (total=parts[1], connected=parts[2], disconnected=parts[3])
end

function _ensemble_sff_parts(spectra, ts, beta, unfold, deg, goe)
    isempty(spectra) &&
        throw(ArgumentError("ensembleSpectralFormFactor requires at least one spectrum"))
    levels = [_sff_levels(E, unfold, deg, goe) for E in spectra]
    n = length(first(levels))
    all(l -> length(l) == n, levels) || throw(ArgumentError(
        "ensembleSpectralFormFactor requires spectra of equal length, got lengths " *
        string(length.(levels))))
    # One common energy shift for the whole ensemble: it cancels in
    # each normalized amplitude's magnitude and, being common, also in
    # the ensemble mean amplitude behind the disconnected part.
    shift = minimum(minimum(l) for l in levels)
    total = zeros(Float64, length(ts))
    disconnected = zeros(Float64, length(ts))
    for (k, t) in enumerate(ts)
        asum = zero(ComplexF64)
        asq = 0.0
        for l in levels
            a = _sff_amplitude(l, t, beta, shift)
            asum += a
            asq += abs2(a)
        end
        m = length(levels)
        total[k] = asq / m
        disconnected[k] = abs2(asum / m)
    end
    return total, total .- disconnected, disconnected
end
