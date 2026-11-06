# Eigenvalue samplers for the Gaussian and Wishart beta ensembles at
# general beta > 0, via the Dumitriu-Edelman tridiagonal models
# (Dumitriu & Edelman, J. Math. Phys. 43, 2002).
#
# Hermite: the symmetric tridiagonal with diagonal sqrt(2/beta) N(0,1)
# and off-diagonal chi_{(N-i) beta} / sqrt(beta) has eigenvalue density
#     prod_{i<j} |lam_i - lam_j|^beta * exp(-beta/4 * sum_i lam_i^2),
# the Gaussian beta ensemble. (The trace identity E[sum lam^2] =
# 2N/beta + N(N-1) holds exactly and is used as a calibration anchor.)
# The same construction drives the general-beta `rand` of `TracyWidom`
# (see `_tw_tridiagonal_sample` in TracyWidom.jl, which shares the
# builder below).
#
# Laguerre: the lower bidiagonal B with diagonal
# chi_{beta(n-i+1)} / sqrt(beta) and subdiagonal
# chi_{beta(p-i+1)} / sqrt(beta) gives B'B the eigenvalue density
#     prod_i lam_i^{beta(n-p+1)/2 - 1} * exp(-beta/2 * sum_i lam_i)
#         * prod_{i<j} |lam_i - lam_j|^beta,
# the Wishart beta ensemble: at beta = 1 and beta = 2 these are exactly
# the eigenvalue laws of real and complex Wishart Gram matrices, and
# E[sum lam] = n p exactly, so the eigenvalues divided by n follow the
# Marchenko-Pastur bulk with aspect p/n at every beta.

export BetaHermite, BetaLaguerre

function _beta_hermite_tridiagonal(rng::Random.AbstractRNG, β::Real, N::Integer)
    dv = sqrt(2 / β) .* randn(rng, N)
    ev = [sqrt(rand(rng, Chisq((N - i) * β))) / sqrt(β) for i in 1:N-1]
    return SymTridiagonal(dv, ev)
end

function _beta_laguerre_tridiagonal(rng::Random.AbstractRNG, β::Real,
                                     n::Integer, p::Integer)
    d = [sqrt(rand(rng, Chisq(β * (n - i + 1)))) / sqrt(β) for i in 1:p]
    e = [sqrt(rand(rng, Chisq(β * (p - i + 1)))) / sqrt(β) for i in 2:p]
    # T = B'B for the lower bidiagonal B with diagonal d, subdiagonal e
    # (e[i] is B[i+1, i]): T_ii = d_i^2 + e_{i-1}^2, T_{i,i+1} = e_i d_i+1.
    td = d .^ 2 .+ vcat(0.0, e .^ 2)
    te = e .* d[2:end]
    return SymTridiagonal(td, te)
end

"""
    BetaHermite(beta, n)

The Gaussian `beta`-ensemble: the eigenvalue distribution of an
`n x n` Dumitriu–Edelman tridiagonal matrix, with joint density

    prod_{i<j} |λᵢ - λⱼ|^beta * exp(-beta/4 * sumᵢ λᵢ^2).

`rand` returns the `n` eigenvalues as a sorted vector; it is a
`Distributions` multivariate distribution, so `rand(rng, d, k)` draws
`k` spectra as the columns of an `n x k` matrix. `beta = 1, 2, 4` are
the classical orthogonal / unitary / symplectic cases (the eigenvalue
law of a GOE / GUE / GSE matrix in this scaling), and any `beta > 0`
is allowed.

The empirical distribution of `λ / sqrt(n)` converges to the
semicircle law on `[-2, 2]`, and the rescaled largest eigenvalue
`n^(1/6) (λmax - 2 sqrt(n))` converges to [`TracyWidom`](@ref)`(beta)`
(Ramírez–Rider–Virág) — with one exception: at `beta = 4` that
rescaled variable converges to the stochastic-Airy normalization of
TW_4 (mean -2.05520, var 0.41092), and it converges to the classic
`TracyWidom(4)` (mean -2.30688, var 0.51772) only after
multiplication by `2^(1/6)`. The exact identity
`E[sum λᵢ^2] = 2n/beta + n(n-1)` holds at every `n`, and
`mean(d) = zeros(n)` by the symmetry `λ -> -λ` of the density.

```julia
rand(BetaHermite(2, 100))        # one spectrum of 100 eigenvalues
rand(BetaHermite(0.5, 50), 200)  # 200 spectra as a 50 x 200 matrix
```

Reference: Dumitriu & Edelman, "Matrix models for beta ensembles",
J. Math. Phys. 43(12), 2002, DOI 10.1063/1.1507823.
"""
struct BetaHermite <: ContinuousMultivariateDistribution
    β::Float64
    n::Int
    function BetaHermite(β::Real, n::Integer)
        (isfinite(β) && β > 0) ||
            throw(ArgumentError("beta must be positive and finite, got β = $β"))
        n >= 1 || throw(ArgumentError("BetaHermite requires n >= 1, got n = $n"))
        new(float(β), Int(n))
    end
end

Base.length(d::BetaHermite) = d.n

Distributions.insupport(d::BetaHermite, x::AbstractVector) =
    length(x) == d.n && issorted(x)

Distributions.mean(d::BetaHermite) = zeros(d.n)

function Distributions._rand!(rng::Random.AbstractRNG, d::BetaHermite,
                              x::AbstractVector{<:Real})
    x .= eigvals(_beta_hermite_tridiagonal(rng, d.β, d.n))
    return x
end

"""
    BetaLaguerre(beta, n, p)

The Wishart `beta`-ensemble: the eigenvalue distribution of an
unnormalized `p x p` Gram matrix with `n` observations, sampled from
the Dumitriu–Edelman bidiagonal model. The joint density is

    prodᵢ λᵢ^(beta(n-p+1)/2 - 1) * exp(-beta/2 * sumᵢ λᵢ)
        * prod_{i<j} |λᵢ - λⱼ|^beta,     λᵢ >= 0,

so at `beta = 1` and `beta = 2` these are exactly the eigenvalues of
a real and a complex Wishart matrix `X'X` with `X` an `n x p`
standard Gaussian matrix. `rand` returns the `p` eigenvalues as a
sorted vector; `rand(rng, d, k)` draws `k` spectra as the columns of
a `p x k` matrix. Requires `n >= p >= 1` (for `n < p` the Gram matrix
is singular, a case this sampler does not cover).

The exact identity `E[sum λᵢ] = n p` holds at every `beta`, so the
eigenvalues divided by `n` have mean 1 and their empirical
distribution converges to [`MarchenkoPastur`](@ref)`(p/n)`; the
largest eigenvalue, centered with [`twScaling`](@ref)`(n, p)`,
follows `TracyWidom(beta)` in the same sense as for
[`BetaHermite`](@ref) (including its `beta = 4` `2^(1/6)` caveat).

```julia
rand(BetaLaguerre(1, 400, 100)) ./ 400   # one Wishart spectrum, MP scale
```

Reference: Dumitriu & Edelman, "Matrix models for beta ensembles",
J. Math. Phys. 43(12), 2002, DOI 10.1063/1.1507823.
"""
struct BetaLaguerre <: ContinuousMultivariateDistribution
    β::Float64
    n::Int
    p::Int
    function BetaLaguerre(β::Real, n::Integer, p::Integer)
        (isfinite(β) && β > 0) ||
            throw(ArgumentError("beta must be positive and finite, got β = $β"))
        p >= 1 || throw(ArgumentError("BetaLaguerre requires p >= 1, got p = $p"))
        n >= p || throw(ArgumentError(
            "BetaLaguerre requires n >= p (the Gram matrix is singular for n < p), got n = $n, p = $p"))
        new(float(β), Int(n), Int(p))
    end
end

Base.length(d::BetaLaguerre) = d.p

Distributions.insupport(d::BetaLaguerre, x::AbstractVector) =
    length(x) == d.p && issorted(x) && all(>=(0), x)

function Distributions._rand!(rng::Random.AbstractRNG, d::BetaLaguerre,
                              x::AbstractVector{<:Real})
    x .= eigvals(_beta_laguerre_tridiagonal(rng, d.β, d.n, d.p))
    return x
end
