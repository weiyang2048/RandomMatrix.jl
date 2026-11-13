# Generalized spiked covariance: an arbitrary population spectrum H.
#
# The layer in SpikedCovariance.jl treats a population covariance whose
# bulk is a single point mass sigma^2. Here the population eigenvalues
# have a general limiting spectral distribution H, and the spike layer
# is rebuilt on the Silverstein companion transform of FreeProbability.jl
# (this file is included after it, since the input forms dispatch on
# FreeMeasure). Theory: Baik & Silverstein, "Eigenvalues of large
# sample covariance matrices of spiked population models",
# J. Multivariate Anal. 97 (2006), 1382-1408; computational formulation
# in Couillet & Liao, "Random Matrix Methods for Machine Learning",
# Cambridge Univ. Press, 2022.
#
# The one identity everything follows from: for a sample eigenvalue
# lambda outside the limiting bulk F of the sample covariance, with
# munderline the companion Stieltjes transform of
# silversteinLaw's equation,
#
#     lambda is an outlier for the population spike l
#         iff  munderline(lambda) = -1 / l,
#
# and inverting the companion map z(m) = gamma * I(m) - 1/m of
# FreeProbability.jl at m = -1/l gives the outlier map in closed form,
#
#     lambda = psi(l) = l * (1 + gamma * int t / (l - t) dH(t)),
#     psi'(l) = 1 - gamma * int t^2 / (l - t)^2 dH(t).
#
# psi is convex on (sup supp H, Inf) with psi' increasing from -Inf to
# 1 (the top point of supp H always carries mass in the discrete
# representations used here), so psi has a unique minimizer lc: spikes
# l > lc produce outliers at psi(l), spikes l <= lc are absorbed and
# stick to the bulk edge psi(lc) -- which is exactly the upper support
# edge of silversteinLaw(H, gamma), since psi(lc) is the critical
# value z(m_c) of the companion inverse map. The squared eigenvector
# overlap is l psi'(l) / psi(l). At H = delta_{sigma^2} every formula
# reduces algebraically to the scalar layer of SpikedCovariance.jl
# (psi(l) = l (1 + gamma sigma^2 / (l - sigma^2)), lc = sigma^2
# (1 + sqrt(gamma))), and the test suite pins that reduction to
# rounding error.
#
# Scope: spikes are read against the TOP of supp H (the supercritical
# side, as in the scalar layer). spikeVariance has no general-H method
# here: the Baik-Silverstein fluctuation variance for a general H is a
# separate formula from Paul's identity-bulk one and is not folded in.

export dTransform

# Validate (points, masses, gamma) exactly as _silverstein_discrete
# does, normalize the masses, and drop zero-mass points (they are not
# part of the support; the top support point must carry mass for the
# bracketing below). Returns (t, w, tmax) with tmax = max supp H.
function _spikeH_setup(points, masses, γ)
    (isfinite(γ) && γ > 0) ||
        throw(ArgumentError("spiked covariance with general H requires a finite gamma > 0, got gamma = $γ"))
    isempty(points) && throw(ArgumentError("spike layer: empty population spectrum"))
    length(points) == length(masses) ||
        throw(ArgumentError("spike layer: points and masses must have the same length"))
    all(isfinite, points) ||
        throw(ArgumentError("spike layer: population eigenvalues must be finite"))
    all(x -> x >= 0, points) ||
        throw(ArgumentError("spike layer: population eigenvalues must be >= 0 (T is positive semidefinite)"))
    all(isfinite, masses) || throw(ArgumentError("spike layer: masses must be finite"))
    all(x -> x >= 0, masses) || throw(ArgumentError("spike layer: masses must be >= 0"))
    s = sum(masses)
    s > 0 || throw(ArgumentError("spike layer: masses must have positive total"))
    w = collect(float.(masses)) ./ s
    t = collect(float.(points))
    keep = w .> 0
    t, w = t[keep], w[keep]
    tmax = maximum(t)
    tmax > 0 || throw(ArgumentError(
        "spike layer: population spectrum is a point mass at 0 (T = 0), no spike structure"))
    return t, w, tmax
end

_spikeH_A(t, w, l) = sum(wi * ti / (l - ti) for (ti, wi) in zip(t, w))
_spikeH_B(t, w, l) = sum(wi * ti^2 / (l - ti)^2 for (ti, wi) in zip(t, w))
_spikeH_psi(t, w, γ, l) = l * (1 + γ * _spikeH_A(t, w, l))
_spikeH_dpsi(t, w, γ, l) = 1 - γ * _spikeH_B(t, w, l)

# The BBP threshold lc and the bulk edge psi(lc): the unique root of
# psi' on (tmax, Inf), by bisection. The bracket opens at
# tmax + tmax * sqrt(gamma * w_top), where the top atom alone already
# contributes 1/gamma to the integral in psi' (so psi' <= 0 there,
# with equality only for H a single point mass); halving the offset
# until psi' < 0 handles that boundary case.
function _spikeH_threshold(t, w, γ, tmax)
    wtop = sum(wi for (ti, wi) in zip(t, w) if ti == tmax)
    δ = tmax * sqrt(γ * wtop)
    lo = tmax + δ
    while _spikeH_dpsi(t, w, γ, lo) >= 0 && δ > 1e-13 * tmax
        δ /= 2
        lo = tmax + δ
    end
    hi = lo + max(δ, 0.1 * tmax)
    while _spikeH_dpsi(t, w, γ, hi) <= 0
        hi = lo + 2 * (hi - lo)
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        _spikeH_dpsi(t, w, γ, mid) < 0 ? (lo = mid) : (hi = mid)
    end
    lc = (lo + hi) / 2
    return lc, _spikeH_psi(t, w, γ, lc)
end

function _spikeH_quantiles(d::UnivariateDistribution, hpoints)
    qs = [(i - 0.5) / hpoints for i in 1:hpoints]
    return quantile.(Ref(d), qs), fill(1.0, hpoints)
end

"""
    dTransform(l, points, masses, gamma)
    dTransform(l, mu::FreeMeasure, gamma)
    dTransform(l, d::UnivariateDistribution, gamma; hpoints=401)

The D-transform of the population spectral distribution `H` at the
spike `l`, in the normalization of Baik & Silverstein (2006) and
Couillet & Liao (2022): for `l` above the top of the support of `H`,

```math
\\mathcal D(l) = l \\left(1 + \\gamma \\int \\frac{t}{l - t} \\, dH(t)\\right),
\\qquad \\gamma = p/n .
```

Its role is the outlier map of the generalized spiked covariance
model: a population spike at `l` produces a sample outlier at
`dTransform(l, ...)` exactly when `l` exceeds
[`bbpThreshold`](@ref) for this `H` (the unique minimizer of
`dTransform` on `(sup supp H, Inf)`), and the outlier's squared
eigenvector overlap is `l * D'(l) / D(l)` with
`D'(l) = 1 - gamma * int t^2 / (l - t)^2 dH(t)`. Equivalently
`dTransform(l, ...) = z(-1/l)` for the companion inverse map
`z(m) = gamma * int t / (1 + t m) dH(t) - 1/m` of
[`silversteinLaw`](@ref), i.e. the outlier `lambda` is characterized
by `munderline(lambda) = -1/l`. At `H = delta_{sigma^2}` it reduces
to `l * (1 + gamma * sigma^2 / (l - sigma^2))`, the map inside the
scalar [`spikeLocation`](@ref).

`H` is given in the three input forms of [`silversteinLaw`](@ref):
discrete `points`/`masses` (the integral is then an exact finite
sum), a [`FreeMeasure`](@ref) (its nodes and weights), or a
`Distributions` law supported on `[0, Inf)`, discretized on an
equal-weight quantile grid of `hpoints` points first. Throws an
`ArgumentError` for `l` at or below the top of the support of `H`,
where the integral has its pole.
"""
function dTransform(l::Real, points::AbstractVector{<:Real},
                    masses::AbstractVector{<:Real}, γ::Real)
    t, w, tmax = _spikeH_setup(points, masses, γ)
    l > tmax || throw(ArgumentError(
        "dTransform: spike l = $l must exceed the top of the population spectrum ($tmax)"))
    return _spikeH_psi(t, w, γ, float(l))
end

dTransform(l::Real, μ::FreeMeasure, γ::Real) = dTransform(l, μ.nodes, μ.weights, γ)

function dTransform(l::Real, d::UnivariateDistribution, γ::Real; hpoints::Integer=401)
    pts, ms = _spikeH_quantiles(d, hpoints)
    return dTransform(l, pts, ms, γ)
end

"""
    bbpThreshold(points, masses, gamma)
    bbpThreshold(mu::FreeMeasure, gamma)
    bbpThreshold(d::UnivariateDistribution, gamma; hpoints=401)

Critical population eigenvalue for the BBP phase transition when the
population covariance has limiting spectral distribution `H` (aspect
ratio `gamma = p/n`): a spike `l` yields an outlier sample eigenvalue
only if `l > bbpThreshold(...)`. The threshold is the unique
minimizer `lc` of the [`dTransform`](@ref) on `(sup supp H, Inf)`,
i.e. the unique root there of
`1 = gamma * int t^2 / (lc - t)^2 dH(t)`, found by bisection; the
bulk edge the outlier separates from is `dTransform(lc, ...)`, the
upper support edge of [`silversteinLaw`](@ref)`(H, gamma)`. At
`H = delta_{sigma^2}` this is `sigma^2 (1 + sqrt(gamma))`, the scalar
[`bbpThreshold`](@ref)`(gamma; sigma)`. `H` is given in the three
input forms of [`silversteinLaw`](@ref).
"""
function bbpThreshold(points::AbstractVector{<:Real}, masses::AbstractVector{<:Real},
                      γ::Real)
    t, w, tmax = _spikeH_setup(points, masses, γ)
    return _spikeH_threshold(t, w, γ, tmax)[1]
end

bbpThreshold(μ::FreeMeasure, γ::Real) = bbpThreshold(μ.nodes, μ.weights, γ)

function bbpThreshold(d::UnivariateDistribution, γ::Real; hpoints::Integer=401)
    pts, ms = _spikeH_quantiles(d, hpoints)
    return bbpThreshold(pts, ms, γ)
end

"""
    spikeLocation(l, points, masses, gamma)
    spikeLocation(l, mu::FreeMeasure, gamma)
    spikeLocation(l, d::UnivariateDistribution, gamma; hpoints=401)

Almost-sure limit of the sample eigenvalue generated by a population
spike `l` when the population covariance has limiting spectral
distribution `H` and aspect ratio `gamma = p/n`:

    dTransform(l, H, gamma)     if l > bbpThreshold(H, gamma)

Below the generalized BBP threshold the spike sticks to the upper
bulk edge `dTransform(lc, H, gamma)` -- the upper support edge of
[`silversteinLaw`](@ref)`(H, gamma)` -- which is what this returns.
This extends the scalar [`spikeLocation`](@ref) (the case
`H = delta_{sigma^2}`) to a general population, with `H` in the
three input forms of [`silversteinLaw`](@ref).
"""
function spikeLocation(l::Real, points::AbstractVector{<:Real},
                       masses::AbstractVector{<:Real}, γ::Real)
    t, w, tmax = _spikeH_setup(points, masses, γ)
    lc, bp = _spikeH_threshold(t, w, γ, tmax)
    return l <= lc ? bp : _spikeH_psi(t, w, γ, float(l))
end

spikeLocation(l::Real, μ::FreeMeasure, γ::Real) =
    spikeLocation(l, μ.nodes, μ.weights, γ)

function spikeLocation(l::Real, d::UnivariateDistribution, γ::Real; hpoints::Integer=401)
    pts, ms = _spikeH_quantiles(d, hpoints)
    return spikeLocation(l, pts, ms, γ)
end

"""
    spikeOverlap(l, points, masses, gamma)
    spikeOverlap(l, mu::FreeMeasure, gamma)
    spikeOverlap(l, d::UnivariateDistribution, gamma; hpoints=401)

Almost-sure limit of the squared overlap `|dot(u_hat, u)|^2` between
the sample eigenvector of the outlier and its population counterpart,
for a population spike `l` against a population spectral distribution
`H` with aspect ratio `gamma = p/n`:

    l * D'(l) / D(l),   D = dTransform(., H, gamma),
    D'(l) = 1 - gamma * int t^2 / (l - t)^2 dH(t)

Zero below the generalized BBP threshold: the sample eigenvector is
asymptotically orthogonal to the population direction. This extends
the scalar [`spikeOverlap`](@ref) (the case `H = delta_{sigma^2}`),
with `H` in the three input forms of [`silversteinLaw`](@ref).
"""
function spikeOverlap(l::Real, points::AbstractVector{<:Real},
                      masses::AbstractVector{<:Real}, γ::Real)
    t, w, tmax = _spikeH_setup(points, masses, γ)
    lc, _ = _spikeH_threshold(t, w, γ, tmax)
    l <= lc && return 0.0
    return _spikeH_dpsi(t, w, γ, float(l)) / (1 + γ * _spikeH_A(t, w, float(l)))
end

spikeOverlap(l::Real, μ::FreeMeasure, γ::Real) =
    spikeOverlap(l, μ.nodes, μ.weights, γ)

function spikeOverlap(l::Real, d::UnivariateDistribution, γ::Real; hpoints::Integer=401)
    pts, ms = _spikeH_quantiles(d, hpoints)
    return spikeOverlap(l, pts, ms, γ)
end

"""
    estimateSpike(lambda, points, masses, gamma)
    estimateSpike(lambda, mu::FreeMeasure, gamma)
    estimateSpike(lambda, d::UnivariateDistribution, gamma; hpoints=401)

Estimate the population spike `l` from an observed outlier sample
eigenvalue `lambda`, by inverting [`spikeLocation`](@ref) for a
general population spectral distribution `H`: the unique `l` above
[`bbpThreshold`](@ref)`(H, gamma)` with
`dTransform(l, H, gamma) = lambda`, found by bisection on the
increasing branch of the D-transform. For `lambda` at or below the
upper bulk edge of [`silversteinLaw`](@ref)`(H, gamma)` the spike is
not identifiable and the threshold value is returned, as in the
scalar [`estimateSpike`](@ref). `H` is given in the three input
forms of [`silversteinLaw`](@ref).
"""
function estimateSpike(λ::Real, points::AbstractVector{<:Real},
                       masses::AbstractVector{<:Real}, γ::Real)
    t, w, tmax = _spikeH_setup(points, masses, γ)
    lc, bp = _spikeH_threshold(t, w, γ, tmax)
    λ <= bp && return lc
    lo, hi = lc, max(float(λ), lc + 1.0)
    while _spikeH_psi(t, w, γ, hi) < λ
        hi = lc + 2 * (hi - lc)
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        _spikeH_psi(t, w, γ, mid) < λ ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

estimateSpike(λ::Real, μ::FreeMeasure, γ::Real) =
    estimateSpike(λ, μ.nodes, μ.weights, γ)

function estimateSpike(λ::Real, d::UnivariateDistribution, γ::Real; hpoints::Integer=401)
    pts, ms = _spikeH_quantiles(d, hpoints)
    return estimateSpike(λ, pts, ms, γ)
end
