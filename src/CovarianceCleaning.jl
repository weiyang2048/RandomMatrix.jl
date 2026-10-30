# Cleaning of empirical covariance matrices.
#
# Spectrum-based tools (Marcenko-Pastur): estimate the noise level
# sigma^2 by fitting the Marcenko-Pastur law to the bulk of an empirical
# spectrum; eigenvalue clipping (Laloux, Cizeau, Bouchaud & Potters
# 1999); and the rotationally invariant estimator (RIE) of Bun,
# Bouchaud & Potters (2017), whose shrinkage function coincides with
# Ledoit-Wolf (2020) analytic nonlinear shrinkage (the two differ only
# in how the sample Stieltjes transform is estimated; here the empirical
# Stieltjes transform with Lorentzian broadening eta = p^(-1/2),
# following Bun et al.). That broadening leaves a finite-size bias of
# order p^(-1/2) in the mean cleaned eigenvalue at moderate p (about
# +0.09 at p = 200, gamma = 0.5); see the rieEigenvalues docstring for
# measured values and the bias/loss trade-off in eta.
#
# Data-based tools: the linear shrinkage estimators of Ledoit & Wolf
# (2004) and of Chen, Wiesel, Eldar & Hero (2010; OAS), which take the
# data matrix itself and shrink the sample covariance linearly toward a
# scaled identity, plus Ledoit & Wolf's (2003) constant-correlation
# target for the same estimator. They are the linear counterparts of
# the RIE above (whose shrinkage is nonlinear, eigenvalue by
# eigenvalue); the
# Ledoit-Wolf (2004) estimator here is unrelated to the Ledoit-Wolf
# (2020) nonlinear shrinkage the RIE coincides with.

export fitMarchenkoPastur, clipEigenvalues, rieEigenvalues,
    cleanCovariance, cleanCorrelation,
    ledoitWolf, ledoitWolfShrinkage, oas, oasShrinkage

_mp_edges(γ::Real, σ2::Real) = (σ2 * (1 - sqrt(γ))^2, σ2 * (1 + sqrt(γ))^2)

# Median of the Marcenko-Pastur law with sigma^2 = 1.
#
# For gamma > 1 the law has an atom at 0 of mass 1 - 1/gamma plus a
# continuous part of mass 1/gamma on [(1-sqrt(gamma))^2, (1+sqrt(gamma))^2]
# (the density in Distributions.jl integrates to min(1, 1/gamma)). The
# median is therefore 0 once the atom alone reaches mass 1/2 (gamma >= 2);
# for 1 < gamma < 2 it is the point of the continuous part where the
# accumulated mass reaches 1/2 - (1 - 1/gamma), and for gamma <= 1 it is
# the ordinary median of the continuous part. The quadrature runs in the
# Chebyshev variable x = mid - half*cos(theta), which removes the
# square-root edge singularities of the density (and the x^(-1/2)
# singularity at 0 when gamma = 1), so a midpoint rule on a uniform
# theta grid is accurate over the whole range.
function _mp_median(γ::Real)
    atom = γ > 1 ? 1 - 1 / γ : 0.0
    atom >= 0.5 && return 0.0
    a, b = _mp_edges(γ, 1.0)
    mid, half = (a + b) / 2, (b - a) / 2
    d = MarchenkoPastur(γ, 1.0)
    target = 0.5 - atom
    n = 4096
    Δθ = π / n
    cum = 0.0
    for i in 1:n
        θ = (i - 0.5) * Δθ
        x = mid - half * cos(θ)
        w = pdf(d, x) * half * sin(θ) * Δθ
        if cum + w >= target
            frac = w > 0 ? (target - cum) / w : 0.0
            xlo = mid - half * cos((i - 1) * Δθ)
            xhi = mid - half * cos(i * Δθ)
            return xlo + frac * (xhi - xlo)
        end
        cum += w
    end
    return b
end

"""
    fitMarchenkoPastur(evals, gamma; method=:trim)

Estimate the noise variance `sigma^2` of a covariance spectrum by fitting
the Marcenko-Pastur law with aspect ratio `gamma = p/n` to the eigenvalues
`evals` (the eigenvalues of a sample covariance matrix).

Methods:
- `:trim` (default): iterate `sigma^2 <- mean{lambda_i <= sigma^2 (1+sqrt(gamma))^2}`
  to a fixed point - eigenvalues above the fitted upper edge are treated as
  signal and excluded from the noise estimate.
- `:median`: match medians, `sigma^2 = median(evals) / median(MP_gamma,1)`.
  Requires `gamma < 2`: for `gamma >= 2` the Marcenko-Pastur median is 0
  (the atom at 0 has mass `1 - 1/gamma >= 1/2`), so median matching is
  degenerate and an `ArgumentError` is thrown; use `:trim` there.
"""
function fitMarchenkoPastur(evals::AbstractVector{<:Real}, γ::Real; method::Symbol=:trim)
    λ = collect(float.(evals))
    if method === :trim
        σ2 = mean(λ)
        for _ in 1:200
            edge = σ2 * (1 + sqrt(γ))^2
            bulk = filter(<=(edge), λ)
            isempty(bulk) && break
            new = mean(bulk)
            abs(new - σ2) <= 1e-12 * σ2 && (σ2 = new; break)
            σ2 = new
        end
        return σ2
    elseif method === :median
        m = _mp_median(γ)
        m > 0 || throw(ArgumentError(
            "method=:median is degenerate for gamma >= 2: the " *
            "Marcenko-Pastur median is 0 (atom at 0 of mass " *
            "1 - 1/gamma >= 1/2); use method=:trim"))
        return median(λ) / m
    else
        throw(ArgumentError("method must be :trim or :median"))
    end
end

"""
    clipEigenvalues(evals, gamma; sigma2=fitMarchenkoPastur(evals, gamma))

Eigenvalue clipping (Laloux et al. 1999): eigenvalues inside the fitted
Marcenko-Pastur bulk (`lambda <= sigma^2 (1+sqrt(gamma))^2`) are replaced by
their average; eigenvalues above the edge (signal) are kept unchanged. The
replacement preserves the trace exactly. Returns a vector aligned with the
input order.
"""
function clipEigenvalues(evals::AbstractVector{<:Real}, γ::Real;
                         σ2::Real=fitMarchenkoPastur(evals, γ))
    λ = collect(float.(evals))
    edge = σ2 * (1 + sqrt(γ))^2
    bulk = λ .<= edge
    out = copy(λ)
    if any(bulk)
        out[bulk] .= mean(λ[bulk])
    end
    return out
end

"""
    rieEigenvalues(evals, gamma; eta=length(evals)^(-1/2))

Cleaned eigenvalues of the rotationally invariant estimator (Bun, Bouchaud
& Potters 2017; the Ledoit-Peche formula):

    xi_i = lambda_i / |1 - gamma + gamma * lambda_i * g(z_i)|^2,
    z_i = lambda_i - i*eta,
    g(z) = (1/p) * sum_j 1 / (z - lambda_j)

with `gamma = p/n` and `g` the empirical Stieltjes transform of the sample
spectrum, broadened by `eta` (a Lorentzian local average over neighbouring
eigenvalues). The keyword is also accepted under its Greek spelling `η`
(`eta` takes precedence when both are given). Properties: for a pure-noise
spectrum (`Sigma = sigma^2 I`) the result is flat at `sigma^2`
asymptotically; a supercritical spike `l` observed at `lambda > l` is
shrunk to `xi < l < lambda`.

Finite-size bias: at moderate `p` the default broadening `eta = p^(-1/2)`
leaves a positive bias in the mean cleaned eigenvalue under the null
(`Sigma = I`). Simulation means of `mean(xi)` (24 trials) at
`gamma = 0.5` are about 1.13, 1.09 and 1.07 at `p = 100, 200, 400`
(1.07, 1.05, 1.04 at `gamma = 0.25`) — a bias of roughly
`c(gamma) * p^(-1/2)` with `c(0.25) ≈ 0.73` and `c(0.5) ≈ 1.3`, i.e. of
the order of `eta` itself. The mechanism is the broadening: `g` is
evaluated a distance `eta` off the real axis, which smooths the spectrum
over a window containing `O(sqrt(p))` eigenvalues. Shrinking `eta`
reduces the bias — at `eta = p^(-2/3)` the null means above drop to
about 1.05, 1.03, 1.02 — but `eta` cannot be taken to zero at fixed `p`:
the self-term `1/(z_i - lambda_i) = 1/(-i*eta)` is what regularizes `g`
at its own pole, and once `eta` reaches the mean eigenvalue spacing the
estimator deteriorates (on a spiked population its Frobenius loss, in
oracle units, grows from about 1.15 at the default to about 1.33 at
`eta = p^(-3/4)`, `p = 200`, `gamma = 0.5`). The default is kept because
it is near loss-optimal in those spiked simulations; pass a smaller
`eta` (e.g. `eta = length(evals)^(-2/3)`) when an accurate trace matters
more than minimum loss — at `p >= 200` its spiked loss is tied with or
better than the default, while at `p = 100` it costs about 8%.

Requires `gamma < 1` for the zero-eigenvalue block when `gamma >= 1`:
eigenvalues equal to 0 are mapped to 0 (see `cleanCovariance`).
"""
function rieEigenvalues(evals::AbstractVector{<:Real}, γ::Real;
                        η::Real=length(evals)^(-1 / 2),
                        eta::Union{Real,Nothing}=nothing)
    ηeff = eta === nothing ? float(η) : float(eta)
    λ = collect(float.(evals))
    p = length(λ)
    ξ = similar(λ)
    for i in 1:p
        z = λ[i] - im * ηeff
        g = sum(1 / (z - λ[j]) for j in 1:p) / p
        ξ[i] = λ[i] / abs2(1 - γ + γ * λ[i] * g)
    end
    return ξ
end

"""
    cleanCovariance(S, n; method=:rie)

Clean the sample covariance matrix `S` (formed from `n` observations) by
eigenvalue shrinkage in its own eigenbasis: with `gamma = size(S,1)/n`,

- `method=:rie` - the rotationally invariant estimator ([`rieEigenvalues`](@ref));
  requires `n >= size(S, 1)`.
- `method=:clip` - eigenvalue clipping ([`clipEigenvalues`](@ref)) with the
  noise level fitted by [`fitMarchenkoPastur`](@ref).

Returns a `Symmetric` matrix with the sample eigenvectors and the cleaned
eigenvalues.
"""
function cleanCovariance(S::AbstractMatrix{<:Real}, n::Integer; method::Symbol=:rie)
    p = size(S, 1)
    γ = p / n
    F = eigen(Symmetric(S))
    ξ = if method === :rie
        γ < 1 || throw(ArgumentError("method=:rie requires n >= p (gamma < 1)"))
        rieEigenvalues(F.values, γ)
    elseif method === :clip
        clipEigenvalues(F.values, γ)
    else
        throw(ArgumentError("method must be :rie or :clip"))
    end
    return Symmetric(F.vectors * Diagonal(ξ) * F.vectors')
end

"""
    cleanCorrelation(C, n; method=:rie)

Clean the sample correlation matrix `C` formed from `n` observations.
Identical machinery to [`cleanCovariance`](@ref); provided because
correlation spectra (unit diagonal, trace `= p`) are the standard input in
applications.
"""
cleanCorrelation(C::AbstractMatrix{<:Real}, n::Integer; method::Symbol=:rie) =
    cleanCovariance(C, n; method)

# ---------------------------------------------------------------------------
# Linear shrinkage from the data matrix.
#
# The estimators below share the form
#
#     Sigma_hat = (1 - rho) * S + rho * F,
#
# with S the sample covariance (1/n divisor) of the centered data and F
# a structured target: the scaled identity m * I, m = tr(S) / p, for
# Ledoit-Wolf (2004) and OAS (which differ only in the estimated
# intensity rho), or the constant-correlation matrix of Ledoit & Wolf
# (2003) for `ledoitWolf(X; target = :constant_correlation)`. Every
# target here has the same trace and diagonal as S in the identity and
# target even shares S's whole diagonal), so the shrinkage preserves
# tr(S) exactly, and Sigma_hat is positive definite whenever rho > 0
# and the target is, even for p > n where S is singular.

# Validate the data matrix X (p x n: variables in rows, observations in
# columns) and return a row-centered Float64 copy together with (p, n).
function _centeredData(X::AbstractMatrix{<:Real})
    p, n = size(X)
    p >= 1 || throw(ArgumentError("data matrix X has no rows (variables)"))
    n >= 2 || throw(ArgumentError(
        "need at least 2 observations (columns of X) for a covariance estimate, got n = $n"))
    all(isfinite, X) || throw(ArgumentError("data matrix X contains NaN or Inf entries"))
    Xf = Float64.(X)
    Xf .-= vec(sum(Xf; dims=2)) ./ n
    return Xf, p, n
end

# Assemble (1 - rho) * S + rho * m * I as a Symmetric matrix.
function _shrunkToIdentity(S::Matrix{Float64}, m::Float64, ρ::Float64)
    A = (1 - ρ) .* S
    for i in diagind(A)
        A[i] += ρ * m
    end
    return Symmetric(A)
end

# Sample covariance (1/n divisor), scale m = tr(S)/p, intensity rho.
function _ledoitWolf(X::AbstractMatrix{<:Real})
    Xc, p, n = _centeredData(X)
    S = (Xc * Xc') / n
    m = tr(S) / p
    s2 = sum(abs2, S) # ||S||_F^2
    d2 = (s2 - p * m^2) / p # ||S - mI||_F^2 / p
    ρ = if d2 > 0
        # bbar^2 = (1/(n^2 p)) sum_k ||x_k x_k' - S||_F^2, with the cross
        # terms summed in closed form: sum_k x_k' S x_k = n * ||S||_F^2.
        colnorm2 = vec(sum(abs2, Xc; dims=1)) # ||x_k||^2 per observation
        bbar2 = (sum(abs2, colnorm2) / n - s2) / (n * p)
        clamp(min(bbar2, d2) / d2, 0.0, 1.0)
    else
        # S is exactly mI: every intensity gives the same estimator, and
        # 1 is the limit of rho from non-degenerate data.
        1.0
    end
    return S, m, ρ
end

# Sample covariance (1/n divisor), scale m = tr(S)/p, intensity rho.
function _oas(X::AbstractMatrix{<:Real})
    Xc, p, n = _centeredData(X)
    S = (Xc * Xc') / n
    t = tr(S)
    m = t / p
    s2 = sum(abs2, S) # tr(S^2)
    num = (1 - 2 / p) * s2 + t^2
    den = (n + 1 - 2 / p) * (s2 - t^2 / p)
    # den == 0 iff S is exactly a multiple of the identity; as there,
    # rho = 1 is the continuous limit (and the estimator is S either way).
    ρ = den > 0 ? clamp(num / den, 0.0, 1.0) : 1.0
    return S, m, ρ
end

# Ledoit-Wolf (2003) shrinkage toward the constant-correlation target:
# sample covariance, target F, intensity rho. The target keeps the
# sample variances on the diagonal and sets every off-diagonal entry
# to rbar * sqrt(s_ii * s_jj), with rbar the average sample
# correlation. The intensity is the papers' asymptotic formula
# rho = clamp((pihat - rhohat) / (gamma * n), 0, 1) derived for this
# target (see the ledoitWolfShrinkage docstring).
function _ledoitWolfCC(X::AbstractMatrix{<:Real})
    Xc, p, n = _centeredData(X)
    S = (Xc * Xc') / n
    v = diag(S) # sample variances s_ii
    sd = sqrt.(v)
    # Average sample correlation over the i < j pairs. Pairs touching
    # a zero-variance variable contribute 0 to the average; their
    # target entries vanish anyway because sd_i * sd_j = 0. (The
    # papers assume s_ii > 0 throughout.)
    rbar = 0.0
    if p > 1
        acc = 0.0
        for j in 1:p, i in 1:p
            if i != j && v[i] > 0 && v[j] > 0
                acc += S[i, j] / (sd[i] * sd[j])
            end
        end
        rbar = acc / (p * (p - 1))
    end
    F = rbar .* (sd * sd')
    for i in 1:p
        F[i, i] = v[i]
    end
    # pihat = sum_ij (1/n) sum_k (x_ik x_jk - s_ij)^2
    #       = (1/n) sum_k ||x_k||^4 - ||S||_F^2, in closed form as in
    # _ledoitWolf (the cross terms use sum_k x_k' S x_k = n ||S||_F^2).
    colnorm2 = vec(sum(abs2, Xc; dims=1)) # ||x_k||^2 per observation
    pihat = sum(abs2, colnorm2) / n - sum(abs2, S)
    # With A[i, k] = x_ik^2 - s_ii: pihat_ii = (1/n) sum_k A[i, k]^2,
    # and theta_{ii,ij} = (1/n) sum_k A[i, k] x_ik x_jk (the -s_ij
    # part of the second factor drops out because sum_k A[i, k] = 0),
    # so Theta = ((A .* Xc) * Xc') / n has entries theta_{ii,ij}, and
    # theta_{jj,ij} = Theta[j, i].
    A = Xc .^ 2
    for i in 1:p
        A[i, :] .-= v[i]
    end
    Theta = ((A .* Xc) * Xc') ./ n
    rhohat = sum(vec(sum(abs2, A; dims=2))) / n # sum_i pihat_ii
    for j in 1:p, i in 1:p
        if i != j && v[i] > 0 && v[j] > 0
            rhohat += (rbar / 2) * (sqrt(v[j] / v[i]) * Theta[i, j]
                                    + sqrt(v[i] / v[j]) * Theta[j, i])
        end
    end
    gamma = sum(abs2, F .- S) # ||F - S||_F^2
    # gamma == 0 iff S already equals its constant-correlation target
    # (always so for p == 1, and for constant data, where S = F = 0):
    # every intensity gives the same estimator, and 1 is the limit of
    # rho from non-degenerate data, as in _ledoitWolf.
    ρ = gamma > 0 ? clamp((pihat - rhohat) / gamma / n, 0.0, 1.0) : 1.0
    return S, F, ρ
end

"""
    ledoitWolf(X; target = :identity)

Ledoit-Wolf linear shrinkage estimate of the covariance matrix from
the data matrix `X` (`p x n`: variables in rows, observations in
columns; the data are centered internally):

    Sigma_hat = (1 - rho) * S + rho * F,

where `S` is the sample covariance (`1/n` divisor), `F` is the
shrinkage target selected by `target`, and `rho` is the
asymptotically optimal shrinkage intensity estimated from the data
([`ledoitWolfShrinkage`](@ref)). Two targets are available:

- `target = :identity` (the default; Ledoit & Wolf 2004): the scaled
  identity `F = m * I` with `m = tr(S) / p`.
- `target = :constant_correlation` (Ledoit & Wolf 2003): the
  constant-correlation matrix, which keeps the sample variances on
  the diagonal and sets every off-diagonal entry to
  `rbar * sqrt(s_ii * s_jj)`, where `rbar` is the average of the
  sample correlations. The estimate then preserves the diagonal of
  `S` (hence its trace) exactly.

Shrinking toward a structured target trades a little bias for a
large variance reduction; the estimate is positive definite whenever
`rho > 0` and the target is, even for `p > n`. Unlike
[`cleanCovariance`](@ref), which works from a covariance matrix and
shrinks its eigenvalues nonlinearly, this estimator needs the data
themselves, because the intensity estimate involves fourth moments
of the observations.

Note this is the *linear* shrinkage estimator, not the Ledoit-Wolf
(2020) analytic *nonlinear* shrinkage whose shrinkage function
[`rieEigenvalues`](@ref) implements.

References: Ledoit & Wolf, "Improved estimation of the covariance
matrix of stock returns with an application to portfolio selection",
Journal of Empirical Finance 10 (2003), 603-621; Ledoit & Wolf,
"A well-conditioned estimator for large-dimensional covariance
matrices", Journal of Multivariate Analysis 88 (2004), 365-411.
"""
function ledoitWolf(X::AbstractMatrix{<:Real}; target::Symbol=:identity)
    if target === :identity
        S, m, ρ = _ledoitWolf(X)
        return _shrunkToIdentity(S, m, ρ)
    elseif target === :constant_correlation
        S, F, ρ = _ledoitWolfCC(X)
        return Symmetric((1 - ρ) .* S .+ ρ .* F)
    end
    throw(ArgumentError("unknown shrinkage target $(repr(target)); " *
                        "expected :identity or :constant_correlation"))
end

"""
    ledoitWolfShrinkage(X; target = :identity)

The Ledoit-Wolf shrinkage intensity `rho in [0, 1]` used by
[`ledoitWolf`](@ref) with the same `target`, for the data matrix `X`
(`p x n`, centered internally).

For `target = :identity`, with `S` the sample covariance,
`m = tr(S)/p` and the centered observations `x_k`,

    rho = min(bbar^2, d^2) / d^2,
    d^2    = ||S - mI||_F^2 / p,
    bbar^2 = ( (1/n) * sum_k ||x_k||^4 - ||S||_F^2 ) / (n * p),

where `bbar^2` is `(1/(n^2 p)) * sum_k ||x_k x_k' - S||_F^2` with the
cross terms summed in closed form. The values agree with scikit-learn's
`ledoit_wolf` (cross-checked to 1e-15 on fixed datasets). If `S` is
exactly a multiple of the identity, every intensity yields the same
estimator and `rho = 1` (the limit from non-degenerate data) is
returned by convention; scikit-learn returns 0 in that corner, with the
same shrunk matrix.

For `target = :constant_correlation`, the intensity is the formula
Ledoit & Wolf (2003) derive for that target. With `F` the
constant-correlation target, `rbar` the average sample correlation,
and the entrywise asymptotic variances and covariances

    pi_ij      = (1/n) * sum_k (x_ik x_jk - s_ij)^2,
    theta_ii,ij = (1/n) * sum_k (x_ik^2 - s_ii) (x_ik x_jk - s_ij),

the estimator is

    rho = clamp( (pi - rhohat) / (gamma * n), 0, 1 ),
    pi    = sum_ij pi_ij,
    rhohat = sum_i pi_ii + sum_{i != j} (rbar / 2) *
             ( sqrt(s_jj / s_ii) * theta_ii,ij
               + sqrt(s_ii / s_jj) * theta_jj,ij ),
    gamma = ||S - F||_F^2.

Here `pi` sums the asymptotic variances of the entries of `S`,
`rhohat` sums their asymptotic covariances with the entries of the
(random) target `F` — the diagonal contributes `pi_ii` because
`f_ii = s_ii`, and each off-diagonal covariance follows from the
delta method applied to `f_ij = rbar * sqrt(s_ii s_jj)` — and
`gamma` measures the target's misspecification. Unlike the identity
formula this is not scikit-learn's estimator (which only implements
the identity target); the values are cross-checked against an
independent implementation of the papers' formulas. Pairs touching
a zero-variance variable contribute 0 to `rbar` and to the `rhohat`
double sum (their `theta` terms vanish identically). If `gamma = 0`
— `S` already equals its constant-correlation target, always the
case for `p = 1` — every intensity gives the same estimator and
`rho = 1` by the same limit convention as above.
"""
function ledoitWolfShrinkage(X::AbstractMatrix{<:Real}; target::Symbol=:identity)
    if target === :identity
        return _ledoitWolf(X)[3]
    elseif target === :constant_correlation
        return _ledoitWolfCC(X)[3]
    end
    throw(ArgumentError("unknown shrinkage target $(repr(target)); " *
                        "expected :identity or :constant_correlation"))
end

"""
    oas(X)

Oracle approximating shrinkage (OAS) estimate of the covariance matrix
from the data matrix `X` (`p x n`: variables in rows, observations in
columns; the data are centered internally):

    Sigma_hat = (1 - rho) * S + rho * m * I,   m = tr(S) / p,

with `S` the sample covariance (`1/n` divisor) and `rho` the OAS
intensity ([`oasShrinkage`](@ref)). OAS refines the Ledoit-Wolf linear
shrinkage under Gaussian assumptions by approximating the oracle
(MMSE-optimal) intensity with the population traces replaced by
iterated estimates; in small samples it typically shrinks noticeably
more than [`ledoitWolf`](@ref). Like it, the estimate preserves the
trace of `S` exactly and is positive definite whenever `rho > 0`, even
for `p > n`.

Reference: Chen, Wiesel, Eldar & Hero, "Shrinkage algorithms for MMSE
covariance estimation", IEEE Transactions on Signal Processing 58(10)
(2010), 5016-5029.
"""
function oas(X::AbstractMatrix{<:Real})
    S, m, ρ = _oas(X)
    return _shrunkToIdentity(S, m, ρ)
end

"""
    oasShrinkage(X)

The oracle approximating shrinkage (OAS) intensity `rho in [0, 1]` used
by [`oas`](@ref), for the data matrix `X` (`p x n`, centered
internally). With `S` the sample covariance (`1/n` divisor),

    rho = ( (1 - 2/p) * tr(S^2) + tr(S)^2 ) /
          ( (n + 1 - 2/p) * (tr(S^2) - tr(S)^2 / p) ),

clipped to `[0, 1]` — Eq. (23) of Chen, Wiesel, Eldar & Hero (2010),
with the `2/p` terms retained. (scikit-learn's `oas` omits those terms,
as its documentation notes; at small `p` that changes `rho` by several
percent, e.g. 0.88 instead of 0.79 on a `p = 4, n = 6` test dataset.)
If `S` is exactly a multiple of the identity the denominator vanishes;
every intensity then gives the same estimator and `rho = 1`, the
continuous limit, is returned.
"""
oasShrinkage(X::AbstractMatrix{<:Real}) = _oas(X)[3]
