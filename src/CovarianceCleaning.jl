# Marcenko-Pastur based cleaning of empirical covariance matrices.
#
# Three tools: estimate the noise level sigma^2 by fitting the
# Marcenko-Pastur law to the bulk of an empirical spectrum; eigenvalue
# clipping (Laloux, Cizeau, Bouchaud & Potters 1999); and the rotationally
# invariant estimator (RIE) of Bun, Bouchaud & Potters (2017), whose
# shrinkage function coincides with Ledoit-Wolf (2020) analytic nonlinear
# shrinkage (the two differ only in how the sample Stieltjes transform is
# estimated; here the empirical Stieltjes transform with Lorentzian
# broadening eta = p^(-1/2), following Bun et al.). That broadening
# leaves a finite-size bias of order p^(-1/2) in the mean cleaned
# eigenvalue at moderate p (about +0.09 at p = 200, gamma = 0.5); see
# the rieEigenvalues docstring for measured values and the bias/loss
# trade-off in eta.

export fitMarchenkoPastur, clipEigenvalues, rieEigenvalues,
    cleanCovariance, cleanCorrelation

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
