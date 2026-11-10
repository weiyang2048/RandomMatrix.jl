# The Tracy-Widom distributions F_beta (beta = 1, 2, 4), as a
# Distributions.jl continuous univariate distribution.
#
# Method: the Hastings-McLeod solution q of Painleve II,
#     q'' = x q + 2 q^3,   q(x) ~ Ai(x) as x -> +infty,
# is computed once (lazily) by Newton iteration on a finite-difference grid.
# With R(s) = int_s^infty q and V(s) = int_s^infty (t-s) q(t)^2 dt,
#     F_2(s) = exp(-V(s)),
#     F_1(s) = sqrt(F_2(s)) * exp(-R(s)/2),
#     F_4(s) = sqrt(F_2(c s)) * (exp(-R(c s)/2) + exp(R(c s)/2)) / 2,
# with c = sqrt(2) (Dieng's form F_4(s/sqrt(2))^2 = F_2(s) cosh^2(R(s)/2)),
# which is the convention whose mean is -2.306884893241; the symplectic
# scaling differs between sources by root-2 / 2^(1/6) factors, so the
# constant was also verified numerically against published TW_4 quantiles.
# Densities follow by analytic differentiation:
# with Q2(s) = int_s^infty q^2,  F_2' = F_2 * Q2 and (log of the F_1
# factor)' contributes (Q2 + q)/2.
#
# For general beta > 0 no closed-form cdf exists; `rand` then uses the
# Dumitriu-Edelman tridiagonal model, and cdf/pdf/quantile throw.

export TracyWidom, twScaling, twTest, manovaTWScaling, manovaTWTest

const _TW_MEAN = Dict(1 => -1.2065335745820, 2 => -1.771086807411, 4 => -2.306884893241)
const _TW_VAR = Dict(1 => 1.607781034581, 2 => 0.8131947928329, 4 => 0.5177237207726)

struct _TWTable
    s::Vector{Float64}      # grid, increasing
    q::Vector{Float64}      # Hastings-McLeod q on the grid
    F::Dict{Int,Vector{Float64}}  # cdfs for beta = 1, 2, 4
    f::Dict{Int,Vector{Float64}}  # pdfs for beta = 1, 2, 4
end

const _TW_TABLE = Ref{Union{Nothing,_TWTable}}(nothing)

function _tw_solve_q()
    lo, hi, h = -10.0, 4.5, 0.004
    xs = collect(lo:h:hi)
    n = length(xs)
    # Boundary values from the asymptotics of q:
    #   x -> -inf: q ~ sqrt(-x/2) (1 + 1/(8 x^3));  x -> +inf: q ~ Ai(x).
    qlo = sqrt(-lo / 2) * (1 + 1 / (8 * lo^3))
    ζ = 2 / 3 * hi^1.5
    qhi = exp(-ζ) / (2 * sqrt(pi) * hi^0.25) * (1 - 5 / (72 * ζ))
    q = [x <= 0 ? sqrt(-x / 2 + 0.05) : qhi * exp(-(x - hi)) for x in xs]
    q[1] = qlo
    q[end] = qhi
    invh2 = 1 / h^2
    for _ in 1:100
        # Residual of q'' - x q - 2 q^3 = 0 on interior points.
        F = zeros(n)
        for i in 2:n-1
            F[i] = (q[i-1] - 2q[i] + q[i+1]) * invh2 - xs[i] * q[i] - 2 * q[i]^3
        end
        # Tridiagonal Jacobian: diag -2/h^2 - x - 6q^2, off-diag 1/h^2.
        dl = fill(invh2, n - 3)
        d = [-2 * invh2 - xs[i] - 6 * q[i]^2 for i in 2:n-1]
        du = fill(invh2, n - 3)
        δ = Tridiagonal(dl, d, du) \ (-F[2:n-1])
        step = maximum(abs, δ)
        q[2:n-1] .+= δ
        step < 1e-13 && break
    end
    return xs, q
end

function _tw_build_table()
    xs, q = _tw_solve_q()
    n = length(xs)
    h = xs[2] - xs[1]
    # Cumulative integrals from the right (trapezoid):
    #   R(s) = int_s^inf q,  Q2(s) = int_s^inf q^2,  W(s) = int_s^inf t q^2.
    R = zeros(n); Q2 = zeros(n); W = zeros(n)
    for i in n-1:-1:1
        R[i] = R[i+1] + h / 2 * (q[i] + q[i+1])
        Q2[i] = Q2[i+1] + h / 2 * (q[i]^2 + q[i+1]^2)
        W[i] = W[i+1] + h / 2 * (xs[i] * q[i]^2 + xs[i+1] * q[i+1]^2)
    end
    V = W .- xs .* Q2                      # int_s^inf (t-s) q^2 dt
    F2 = exp.(-V)
    F1 = exp.(-V ./ 2 .- R ./ 2)
    f2 = F2 .* Q2
    f1 = F1 .* (Q2 .+ q) ./ 2
    # F_4 with the sqrt(2) argument rescaling, evaluated on the s-grid by
    # interpolation of the F_2 / R arrays at c*s.
    c = sqrt(2)
    interp = function (arr, x)
        x <= xs[1] && return arr[1]
        x >= xs[end] && return arr[end]
        j = searchsortedlast(xs, x)
        t = (x - xs[j]) / h
        return arr[j] * (1 - t) + arr[j+1] * t
    end
    F4 = similar(xs); f4 = similar(xs)
    for (i, s) in enumerate(xs)
        y = c * s
        Fy = interp(exp.(-V ./ 2), y)       # sqrt(F_2) at y
        Ey = exp(-interp(R, y) / 2)
        F4[i] = Fy * (Ey + 1 / Ey) / 2
        Q2y = interp(Q2, y); qy = interp(q, y)
        # d/ds [F(y) (E + E^-1)/2], y = c s:
        dF = Fy * Q2y / 2 * (Ey + 1 / Ey) / 2 + Fy * (qy / 4) * (Ey - 1 / Ey)
        f4[i] = c * dF
    end
    F = Dict(1 => F1, 2 => F2, 4 => F4)
    f = Dict(1 => f1, 2 => f2, 4 => f4)
    # Enforce exact monotonicity of the cdfs against quadrature noise.
    for β in (1, 2, 4)
        F[β] .= clamp.(F[β], 0.0, 1.0)
        for i in 2:n
            F[β][i] = max(F[β][i], F[β][i-1])
        end
    end
    return _TWTable(xs, q, F, f)
end

function _tw_table()
    t = _TW_TABLE[]
    if t === nothing
        t = _tw_build_table()
        _TW_TABLE[] = t
    end
    return t
end

function _tw_lookup(arr::Vector{Float64}, xs::Vector{Float64}, x::Real)
    x <= xs[1] && return arr[1]
    x >= xs[end] && return arr[end]
    h = xs[2] - xs[1]
    j = clamp(floor(Int, (x - xs[1]) / h) + 1, 1, length(xs) - 1)
    t = (x - xs[j]) / h
    return arr[j] * (1 - t) + arr[j+1] * t
end

"""
    TracyWidom(beta)

The Tracy-Widom distribution `F_beta`: the limiting law of the largest
eigenvalue of a Gaussian `beta`-ensemble, and (with the
[`twScaling`](@ref) centering) of the largest eigenvalue of a Wishart
matrix. `beta` is 1 (real), 2 (complex) or 4 (symplectic) for
`cdf`/`pdf`/`quantile`; those are computed from the Hastings-McLeod
Painleve II function. For general `beta > 0`, `rand` samples the
Dumitriu-Edelman tridiagonal model (Monte-Carlo accuracy) and
`cdf`/`pdf`/`quantile` are unavailable.

Mean and variance (beta = 1, 2, 4): -1.20653/1.60778, -1.77109/0.81319,
-2.30688/0.51772.
"""
struct TracyWidom <: ContinuousUnivariateDistribution
    β::Float64
    function TracyWidom(β::Real)
        β > 0 || throw(ArgumentError("beta must be positive"))
        new(float(β))
    end
end

_tw_beta_int(d::TracyWidom) =
    (d.β == 1.0 || d.β == 2.0 || d.β == 4.0) ? Int(d.β) : nothing

function Distributions.cdf(d::TracyWidom, x::Real)
    β = _tw_beta_int(d)
    β === nothing && throw(ArgumentError("cdf is available only for beta = 1, 2, 4"))
    t = _tw_table()
    x <= t.s[1] && return 0.0
    x >= t.s[end] && return 1.0
    return _tw_lookup(t.F[β], t.s, x)
end

function Distributions.pdf(d::TracyWidom, x::Real)
    β = _tw_beta_int(d)
    β === nothing && throw(ArgumentError("pdf is available only for beta = 1, 2, 4"))
    t = _tw_table()
    (x <= t.s[1] || x >= t.s[end]) && return 0.0
    return max(_tw_lookup(t.f[β], t.s, x), 0.0)
end

function Distributions.quantile(d::TracyWidom, p::Real)
    β = _tw_beta_int(d)
    β === nothing && throw(ArgumentError("quantile is available only for beta = 1, 2, 4"))
    0 <= p <= 1 || throw(ArgumentError("p must lie in [0, 1]"))
    t = _tw_table()
    F = t.F[β]
    j = searchsortedfirst(F, p)
    j <= 1 && return t.s[1]
    j > length(F) && return t.s[end]
    # Linear inversion on the bracketing segment.
    F0, F1 = F[j-1], F[j]
    return F1 == F0 ? t.s[j] : t.s[j-1] + (p - F0) / (F1 - F0) * (t.s[j] - t.s[j-1])
end

function Distributions.mean(d::TracyWidom)
    β = _tw_beta_int(d)
    return β === nothing ? NaN : _TW_MEAN[β]
end

function Distributions.var(d::TracyWidom)
    β = _tw_beta_int(d)
    return β === nothing ? NaN : _TW_VAR[β]
end

function Distributions.rand(rng::Random.AbstractRNG, d::TracyWidom)
    β = _tw_beta_int(d)
    if β !== nothing
        return Distributions.quantile(d, rand(rng))
    end
    return _tw_tridiagonal_sample(rng, d.β)
end

# Dumitriu-Edelman: the top eigenvalue of the beta-Hermite tridiagonal
# matrix (built by `_beta_hermite_tridiagonal` in BetaEnsembles.jl),
# rescaled as N^(1/6) (lam - 2 sqrt(N)). That rescaled variable is the
# stochastic-Airy (Ramirez-Rider-Virag) Tracy-Widom variable, which at
# beta = 1, 2 coincides with the classic TracyWidom(beta) but at
# beta = 4 differs from it by a factor 2^(1/6): the SAO limit has
# mean -2.05520 / var 0.41092, and multiplication by 2^(1/6) gives the
# classic TracyWidom(4) variable (mean -2.30688 / var 0.51772).
# (Sutton, "The Fourfold Way": "our normalization at beta = 4 differs
# from that of Tracy and Widom by a factor of 2^(1/6)".)
function _tw_tridiagonal_sample(rng::Random.AbstractRNG, β::Real; N::Integer=250)
    lam = eigmax(_beta_hermite_tridiagonal(rng, β, N))
    w = N^(1 / 6) * (lam - 2 * sqrt(N))
    return β == 4 ? w * 2^(1 / 6) : w
end

"""
    twScaling(n, p)

Johnstone's centering and scaling `(mu, sigma)` for the largest eigenvalue
of the normalized sample covariance `(1/n) X'X` of an `n x p` standard
Gaussian data matrix: `(lambda_max - mu) / sigma` converges to
`TracyWidom(1)` (real data; `TracyWidom(2)` for complex data), with

    mu    = (sqrt(n-1) + sqrt(p))^2 / n
    sigma = (sqrt(n-1) + sqrt(p)) (1/sqrt(n-1) + 1/sqrt(p))^(1/3) / n
"""
function twScaling(n::Integer, p::Integer)
    a, b = sqrt(n - 1), sqrt(p)
    μ = (a + b)^2 / n
    σ = (a + b) * (1 / a + 1 / b)^(1 / 3) / n
    return (μ, σ)
end

"""
    twTest(lambda, n, p; beta=1)

Johnstone's Tracy-Widom test of the largest sample eigenvalue: is the
largest eigenvalue `lambda` of the normalized sample covariance
`(1/n) X'X` of an `n x p` Gaussian data matrix consistent with the null
of identity population covariance (pure noise), or is it a signal
outlier?

Returns a NamedTuple `(statistic, pvalue)` with the scaled statistic

    statistic = (lambda - mu) / sigma,     (mu, sigma) = twScaling(n, p)

and the upper-tail p-value `pvalue = 1 - F_beta(statistic)` under
[`TracyWidom`](@ref): a small p-value is evidence that `lambda` sits
above the noise bulk. `beta = 1` (default) is for real data, `beta = 2`
for complex data; Johnstone's scaling covers these two cases.

The null has noise variance 1; for data with noise variance `sigma2`,
apply the test to `lambda / sigma2`.

Reference: Johnstone, "On the distribution of the largest eigenvalue
in principal components analysis", Ann. Statist. 29(2), 2001,
DOI 10.1214/aos/1009210544.
"""
function twTest(λ::Real, n::Integer, p::Integer; beta::Real=1)
    (beta == 1 || beta == 2) || throw(ArgumentError(
        "beta must be 1 (real data) or 2 (complex data)"))
    n >= 2 || throw(ArgumentError("twTest requires n >= 2 observations"))
    p >= 1 || throw(ArgumentError("twTest requires p >= 1 variables"))
    isfinite(λ) || throw(ArgumentError("lambda must be finite"))
    μ, σ = twScaling(n, p)
    z = (float(λ) - μ) / σ
    return (statistic=z, pvalue=1 - cdf(TracyWidom(beta), z))
end

# Johnstone (2008) centering and scaling for the greatest root of a
# double Wishart (MANOVA / Jacobi) problem, on the logit scale.
# Real case: eqs. (5)-(6) of the paper. Complex case: Theorem 2
# (Sec. 1.4) — the constants are the omega^{-1}-weighted averages of
# the (w, omega) pairs built from the angle parameters at
# k = N and k = N - 1, where N = min(n, p), alpha = m - p and
# beta_par = |n - p|. Both cases were validated against MANOVA
# simulation before the test bands were set (see manovaTWTest).

function _manova_tw_check(p::Integer, m::Integer, n::Integer, beta::Real)
    (beta == 1 || beta == 2) || throw(ArgumentError(
        "beta must be 1 (real data) or 2 (complex data)"))
    p >= 2 || throw(ArgumentError("manovaTWTest requires p >= 2 variables"))
    m >= 2 || throw(ArgumentError("manovaTWTest requires error degrees of freedom m >= 2"))
    n >= 2 || throw(ArgumentError("manovaTWTest requires hypothesis degrees of freedom n >= 2"))
    return nothing
end

# Angle pair (gamma, phi) of Johnstone's eq. (6) from the two
# sin^2(./2) fractions, with the fractions validated into (0, 1) —
# outside that range the greatest-root approximation has no angles
# (it encodes max(p, n) < m + n - 1/2).
function _manova_tw_angles(sγ::Real, sφ::Real)
    (0 < sγ < 1 && 0 < sφ < 1) || throw(ArgumentError(
        "Johnstone's angle parameters are undefined for these (p, m, n): " *
        "the sin^2 fractions must lie in (0, 1), got $sγ and $sφ"))
    return (2 * asin(sqrt(sγ)), 2 * asin(sqrt(sφ)))
end

function _manova_tw_scaling_real(p::Integer, m::Integer, n::Integer)
    γ, φ = _manova_tw_angles((min(p, n) - 1 / 2) / (m + n - 1),
                             (max(p, n) - 1 / 2) / (m + n - 1))
    μ = 2 * log(tan((φ + γ) / 2))
    σ = (16 / (m + n - 1)^2 / (sin(φ + γ)^2 * sin(φ) * sin(γ)))^(1 / 3)
    return (μ, σ)
end

# The (w_k, omega_k) pair of Johnstone's Theorem 2 at level k.
function _manova_tw_w_omega(k::Integer, α::Integer, βpar::Integer)
    den = 2k + α + βpar + 1
    γ, φ = _manova_tw_angles((k + 1 / 2) / den, (k + βpar + 1 / 2) / den)
    w = 2 * log(tan((φ + γ) / 2))
    ω = (16 / den^2 / (sin(φ + γ)^2 * sin(φ) * sin(γ)))^(1 / 3)
    return (w, ω)
end

function _manova_tw_scaling_complex(p::Integer, m::Integer, n::Integer)
    N = min(n, p)
    α = m - p
    βpar = abs(n - p)
    wN, ωN = _manova_tw_w_omega(N, α, βpar)
    wM, ωM = _manova_tw_w_omega(N - 1, α, βpar)
    μ = (wN / ωN + wM / ωM) / (1 / ωN + 1 / ωM)
    σ = 2 / (1 / ωN + 1 / ωM)
    return (μ, σ)
end

"""
    manovaTWScaling(p, m, n; beta=1)

Johnstone's centering and scaling `(mu, sigma)` for the greatest root
test of [`manovaTWTest`](@ref): with `theta` the largest eigenvalue
of `(A + B)⁻¹ B` for independent `A ~ W_p(I, m)` (error) and
`B ~ W_p(I, n)` (hypothesis) Wisharts, the logit transform
`W = log(theta / (1 - theta))` satisfies

    (W - mu) / sigma  ->  TracyWidom(beta)

in distribution as `p, m, n` grow in proportion. For real data
(`beta = 1`), `mu = 2 log tan((phi + gamma) / 2)` and
`sigma^3 = 16 (m + n - 1)^{-2} / (sin^2(phi + gamma) sin(phi) sin(gamma))`
with angles `sin^2(gamma/2) = (min(p, n) - 1/2) / (m + n - 1)` and
`sin^2(phi/2) = (max(p, n) - 1/2) / (m + n - 1)` — eqs. (5)–(6) of
Johnstone (2008), whose `-1/2` and `-1` corrections give the
approximation its second-order, `O(p^{-2/3})`, accuracy. For complex
data (`beta = 2`) the constants are the weighted averages of
Theorem 2 of the same paper.

Reference: Johnstone, "Multivariate analysis and Jacobi ensembles:
largest eigenvalue, Tracy–Widom limits and rates of convergence",
Ann. Statist. 36(6), 2008, DOI 10.1214/08-AOS605.
"""
function manovaTWScaling(p::Integer, m::Integer, n::Integer; beta::Real=1)
    _manova_tw_check(p, m, n, beta)
    return beta == 1 ? _manova_tw_scaling_real(p, m, n) :
           _manova_tw_scaling_complex(p, m, n)
end

"""
    manovaTWTest(theta, p, m, n; beta=1)

Johnstone's Tracy–Widom greatest-root test for a MANOVA (double
Wishart) problem: is the largest eigenvalue `theta` of
`(A + B)⁻¹ B` — with `A ~ W_p(I, m)` the error Wishart on `m`
degrees of freedom and `B ~ W_p(I, n)` the hypothesis Wishart on
`n` degrees of freedom, independent with a common covariance under
the null — consistent with pure noise, or evidence of signal (Roy's
largest-root test with a Tracy–Widom p-value)?

Returns a NamedTuple `(statistic, pvalue)` with

    statistic = (log(theta / (1 - theta)) - mu) / sigma,
    (mu, sigma) = manovaTWScaling(p, m, n; beta)

and the upper-tail p-value `pvalue = 1 - F_beta(statistic)` under
[`TracyWidom`](@ref): a small p-value is evidence that the top
MANOVA root sits above the noise bulk. `beta = 1` (default) is for
real data, `beta = 2` for complex data.

Mind the package's other MANOVA convention: the [`Wachter`](@ref)
law and the spiked-MANOVA functions describe `(A + B)⁻¹ A` with
`A ~ W_p(Σ, n₁)` the numerator Wishart. The largest eigenvalue `x`
of that matrix is the greatest root with error degrees of freedom
`n₂` and hypothesis degrees of freedom `n₁`, so test it as
`manovaTWTest(x, p, n₂, n₁)`.

The approximation is second-order (`O(p^{-2/3})`) in the regime of
Johnstone's condition (3): `p, m, n` grow in proportion with
`min(p, n) / (m + n)` bounded away from 0 and `p < m`. It was
calibrated against simulation before the test bands were set: at
`(p, m, n) = (20, 160, 40)`, `(50, 100, 50)`, `(40, 120, 20)` and
`(30, 60, 90)` the scaled statistic's mean and sd match the
Tracy–Widom values within Monte Carlo error and the 5% rejection
rate falls in `0.040`–`0.054` (real) and `0.043`–`0.047` (complex).

This is the balanced regime: for the *unbalanced complex* regime
see [`jacobiTWScaling`](@ref).

Reference: Johnstone, "Multivariate analysis and Jacobi ensembles:
largest eigenvalue, Tracy–Widom limits and rates of convergence",
Ann. Statist. 36(6), 2008, DOI 10.1214/08-AOS605.
"""
function manovaTWTest(θ::Real, p::Integer, m::Integer, n::Integer; beta::Real=1)
    _manova_tw_check(p, m, n, beta)
    (isfinite(θ) && 0 < θ < 1) || throw(ArgumentError(
        "theta must lie in the open interval (0, 1), got theta = $θ"))
    μ, σ = manovaTWScaling(p, m, n; beta=beta)
    z = (log(float(θ) / (1 - float(θ))) - μ) / σ
    return (statistic=z, pvalue=1 - cdf(TracyWidom(beta), z))
end
