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

export TracyWidom, twScaling

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
# matrix with diagonal sqrt(2/beta) N(0,1) and off-diagonal
# chi_{(N-i) beta} / sqrt(beta), rescaled as N^(1/6) (lam - 2 sqrt(N)).
function _tw_tridiagonal_sample(rng::Random.AbstractRNG, β::Real; N::Integer=250)
    dv = sqrt(2 / β) .* randn(rng, N)
    ev = [sqrt(rand(rng, Chisq((N - i) * β))) / sqrt(β) for i in 1:N-1]
    lam = eigmax(SymTridiagonal(dv, ev))
    w = N^(1 / 6) * (lam - 2 * sqrt(N))
    return β == 4 ? w / 2^(1 / 6) : w
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
