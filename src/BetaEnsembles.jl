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
# For n < p the Gram matrix is singular of rank n: its spectrum is
# p - n exact zeros plus n positive eigenvalues, and by the Gram swap
# (the nonzero spectra of X'X and XX' agree) those positive
# eigenvalues are the nonsingular ensemble with (n, p) exchanged, so
# the singular case routes through the same bidiagonal builder with
# the arguments swapped. E[sum lam] = n p holds in both cases.
#
# Jacobi: the Killip-Nenciu model (Killip & Nenciu, Int. Math. Res.
# Not. 2004(50)) gives a lower bidiagonal M, built from independent
# square roots of Beta variables (the cosine/sine factors below),
# whose MM' has eigenvalue density
#     prod_i lam_i^a (1 - lam_i)^b * prod_{i<j} |lam_i - lam_j|^beta,
#     a = beta/2 (n1 - p + 1) - 1,  b = beta/2 (n2 - p + 1) - 1,
# the beta-Jacobi (MANOVA) ensemble: at beta = 1 and beta = 2 these
# are exactly the eigenvalue laws of (A + B)^{-1} A for real and
# complex Wisharts A ~ W_p(I, n1), B ~ W_p(I, n2). The exact identity
# E[sum lam] = p n1 / (n1 + n2) holds at every beta (the matrix-Beta
# mean at beta = 1, 2; the Aomoto integral for the joint density in
# general), and the empirical law tends to Wachter(p/n1, p/n2).
# For n_i < p the corresponding Wishart is singular: the spectrum
# has p - n1 exact zeros and p - n2 exact ones, and the eigenvalues
# in (0, 1) form a nonsingular ensemble of dual dimensions — the
# Gram swap again, with the parameter exchange spelled out in the
# BetaJacobi docstring. E[sum lam] = p n1 / (n1 + n2) survives in
# every regime with n1 + n2 >= p.

export BetaHermite, BetaLaguerre, BetaJacobi

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
    # (e[i] is B[i+1, i]): T_ii = d_i^2 + e_i^2 (with e_p := 0) and
    # T_{i,i+1} = e_i d_{i+1}. Both pieces must come from the SAME Gram
    # matrix: pairing this diagonal with the BB' diagonal
    # d_i^2 + e_{i-1}^2 instead assembles a matrix that is neither B'B
    # nor BB' and corrupts the hard edge (negative eigenvalues at
    # beta = 1, E[n lambda_min] inflated ~30x at beta = 2).
    td = d .^ 2 .+ vcat(e .^ 2, 0.0)
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
a `p x k` matrix. Requires `n >= 1` and `p >= 1`. For `n < p` the
Gram matrix is singular of rank `n`, and `rand` returns its spectrum
in the same form: `p - n` exact zeros plus `n` positive eigenvalues
which, by the Gram swap between `X'X` and `XX'`, follow the joint
density above with `n` and `p` exchanged.

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
        n >= 1 || throw(ArgumentError("BetaLaguerre requires n >= 1, got n = $n"))
        new(float(β), Int(n), Int(p))
    end
end

Base.length(d::BetaLaguerre) = d.p

Distributions.insupport(d::BetaLaguerre, x::AbstractVector) =
    length(x) == d.p && issorted(x) && all(>=(0), x)

function Distributions._rand!(rng::Random.AbstractRNG, d::BetaLaguerre,
                              x::AbstractVector{<:Real})
    if d.n >= d.p
        x .= eigvals(_beta_laguerre_tridiagonal(rng, d.β, d.n, d.p))
    else
        # Singular case n < p: p - n eigenvalues are exactly 0, and
        # the n positive ones are the spectrum of the dual Gram
        # matrix, i.e. the nonsingular ensemble with (n, p) swapped.
        x[1:(d.p - d.n)] .= 0.0
        x[(d.p - d.n + 1):end] .=
            eigvals(_beta_laguerre_tridiagonal(rng, d.β, d.p, d.n))
    end
    return x
end

# The Killip-Nenciu factors of the beta-Jacobi bidiagonal M: `d` is
# its diagonal (length p) and `s` its subdiagonal (length p - 1,
# s[k] = M[k + 1, k]). With C_k, Ct_k independent and
# S_k = sqrt(1 - C_k^2), St_k = sqrt(1 - Ct_k^2):
#   d[k] = C_k * St_k (k < p),  d[p] = C_p;   s[k] = S_{k + 1} * Ct_k,
#   C_k  ~ sqrt(Beta(beta/2 (n1 - p + k), beta/2 (n2 - p + k))),
#   Ct_k ~ sqrt(Beta(beta/2 k, beta/2 (n1 + n2 - 2p + k + 1))).
# (Statement as in Holcomb & Moreno Flores, arXiv:1203.4170, Thm. 2.)
# Draw order is C_1..C_p then Ct_1..Ct_{p-1}; tests rely on it.
function _beta_jacobi_factors(rng::Random.AbstractRNG, β::Real,
                              p::Integer, n1::Integer, n2::Integer)
    C = [sqrt(rand(rng, Beta(β / 2 * (n1 - p + k), β / 2 * (n2 - p + k))))
         for k in 1:p]
    Ct = [sqrt(rand(rng, Beta(β / 2 * k, β / 2 * (n1 + n2 - 2p + k + 1))))
          for k in 1:(p - 1)]
    S = sqrt.(1 .- C .^ 2)
    St = sqrt.(1 .- Ct .^ 2)
    d = vcat(C[1:(p - 1)] .* St, C[p])
    s = S[2:p] .* Ct
    return d, s
end

function _beta_jacobi_tridiagonal(rng::Random.AbstractRNG, β::Real,
                                  p::Integer, n1::Integer, n2::Integer)
    d, s = _beta_jacobi_factors(rng, β, p, n1, n2)
    # T = M M' for the lower bidiagonal M with diagonal d and
    # subdiagonal s: T_ii = d_i^2 + s_{i-1}^2 (with s_0 := 0) and
    # T_{i,i+1} = d_i s_i. Both pieces come from the SAME M: pairing
    # this diagonal with the M'M off-diagonal instead assembles a
    # matrix that is neither Gram matrix (the BetaLaguerre hard-edge
    # bug of 2026-11-14 was exactly that pairing mistake).
    td = d .^ 2 .+ vcat(0.0, s .^ 2)
    te = d[1:(p - 1)] .* s
    return SymTridiagonal(td, te)
end

"""
    BetaJacobi(beta, p, n1, n2)

The Jacobi (MANOVA) `beta`-ensemble: the eigenvalue distribution of
``(A + B)^{-1} A`` at general `beta > 0`, sampled from the
Killip–Nenciu bidiagonal model. The joint density of the `p`
eigenvalues on `[0, 1]` is

    prodᵢ λᵢ^a (1 - λᵢ)^b * prod_{i<j} |λᵢ - λⱼ|^beta,
    a = beta/2 * (n1 - p + 1) - 1,  b = beta/2 * (n2 - p + 1) - 1,

so at `beta = 1` and `beta = 2` these are exactly the eigenvalues of
``(A + B)^{-1} A`` for real and complex Wishart matrices
``A \\sim W_p(I, n_1)``, ``B \\sim W_p(I, n_2)``. `rand` returns the
`p` eigenvalues as a sorted vector; `rand(rng, d, k)` draws `k`
spectra as the columns of a `p x k` matrix. Requires `n1 >= 1`,
`n2 >= 1` and `n1 + n2 >= p`.

For `n_i < p` the corresponding Wishart is singular and the
spectrum carries atoms: `p - n1` eigenvalues are exactly `0` and
`p - n2` exactly `1` — the masses the [`Wachter`](@ref) law places
at the endpoints when `p / n_i > 1`. The eigenvalues in `(0, 1)`
follow a nonsingular `BetaJacobi` ensemble of dual dimensions, by
the same Gram swap as in [`BetaLaguerre`](@ref): for
`n1 < p <= n2` they are `BetaJacobi(beta, n1, p, n1 + n2 - p)`, for
`n2 < p <= n1` the reflection of
`BetaJacobi(beta, n2, p, n1 + n2 - p)`, and for `n1, n2 < p` they
are `BetaJacobi(beta, n1 + n2 - p, n2, n1)`. For `n1 + n2 < p` the
sum `A + B` is itself singular and ``(A + B)^{-1} A`` does not
exist, so the constructor throws.

The exact identity `E[sum λᵢ] = p * n1 / (n1 + n2)` holds at every
`beta` (the atoms and the dual ensemble keep it exact in the
singular regimes too), so `mean(d) = fill(n1 / (n1 + n2), p)`; the
empirical distribution converges to
[`Wachter`](@ref)`(p/n1, p/n2)`. Swapping
`n1` and `n2` reflects the spectrum: `1 .- reverse(λ)` for
`BetaJacobi(beta, p, n2, n1)` has the law of
`BetaJacobi(beta, p, n1, n2)`. At `p = 1` the single eigenvalue is
`Beta(beta * n1 / 2, beta * n2 / 2)` distributed. The largest
eigenvalue, on the logit scale, is centered by
[`manovaTWScaling`](@ref) at `beta = 1, 2`.

```julia
rand(BetaJacobi(1, 100, 200, 300))   # one real MANOVA spectrum
rand(BetaJacobi(4, 50, 80, 80))      # a beta = 4 spectrum, no matrix exists
```

References: Killip & Nenciu, "Matrix models for circular
ensembles", Int. Math. Res. Not. 2004(50), 2004, 2665–2701;
Edelman & Sutton, "The beta-Jacobi matrix model, the CS
decomposition, and generalized singular value problems", Found.
Comput. Math. 8, 2008, 259–285; the bidiagonal statement used here
is Theorem 2 of Holcomb & Moreno Flores, "Edge scaling of the
β-Jacobi ensemble", arXiv:1203.4170.
"""
struct BetaJacobi <: ContinuousMultivariateDistribution
    β::Float64
    p::Int
    n1::Int
    n2::Int
    function BetaJacobi(β::Real, p::Integer, n1::Integer, n2::Integer)
        (isfinite(β) && β > 0) ||
            throw(ArgumentError("beta must be positive and finite, got β = $β"))
        p >= 1 || throw(ArgumentError("BetaJacobi requires p >= 1, got p = $p"))
        n1 >= 1 || throw(ArgumentError("BetaJacobi requires n1 >= 1, got n1 = $n1"))
        n2 >= 1 || throw(ArgumentError("BetaJacobi requires n2 >= 1, got n2 = $n2"))
        n1 + n2 >= p || throw(ArgumentError(
            "BetaJacobi requires n1 + n2 >= p (otherwise A + B is singular and (A + B)^{-1} A is undefined), got n1 + n2 = $(n1 + n2), p = $p"))
        new(float(β), Int(p), Int(n1), Int(n2))
    end
end

Base.length(d::BetaJacobi) = d.p

Distributions.insupport(d::BetaJacobi, x::AbstractVector) =
    length(x) == d.p && issorted(x) && all(λ -> 0 <= λ <= 1, x)

Distributions.mean(d::BetaJacobi) = fill(d.n1 / (d.n1 + d.n2), d.p)

function Distributions._rand!(rng::Random.AbstractRNG, d::BetaJacobi,
                              x::AbstractVector{<:Real})
    β, p, n1, n2 = d.β, d.p, d.n1, d.n2
    if n1 >= p && n2 >= p
        x .= eigvals(_beta_jacobi_tridiagonal(rng, β, p, n1, n2))
    elseif n1 < p && n2 >= p
        # Singular numerator: p - n1 eigenvalues are exactly 0, and
        # the n1 eigenvalues in (0, 1) are the nonsingular dual
        # ensemble BetaJacobi(beta, n1, p, n1 + n2 - p) (Gram swap:
        # the nonzero pencil eigenvalues agree with the dual
        # problem's, exactly as for BetaLaguerre).
        x[1:(p - n1)] .= 0.0
        x[(p - n1 + 1):end] .=
            eigvals(_beta_jacobi_tridiagonal(rng, β, n1, p, n1 + n2 - p))
    elseif n2 < p && n1 >= p
        # Singular denominator: p - n2 eigenvalues are exactly 1,
        # and the n2 eigenvalues in (0, 1) are the reflection of the
        # dual ensemble BetaJacobi(beta, n2, p, n1 + n2 - p).
        x[1:n2] .= 1 .- reverse(
            eigvals(_beta_jacobi_tridiagonal(rng, β, n2, p, n1 + n2 - p)))
        x[(n2 + 1):end] .= 1.0
    else
        # Both singular (n1 + n2 >= p by construction): p - n1
        # zeros, p - n2 ones, and the n1 + n2 - p eigenvalues in
        # (0, 1) are the nonsingular ensemble
        # BetaJacobi(beta, n1 + n2 - p, n2, n1).
        q = n1 + n2 - p
        x[1:(p - n1)] .= 0.0
        if q > 0
            x[(p - n1 + 1):n2] .=
                eigvals(_beta_jacobi_tridiagonal(rng, β, q, n2, n1))
        end
        x[(n2 + 1):end] .= 1.0
    end
    return x
end
