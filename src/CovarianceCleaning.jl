# Marcenko-Pastur based cleaning of empirical covariance matrices.
#
# Three tools: estimate the noise level sigma^2 by fitting the
# Marcenko-Pastur law to the bulk of an empirical spectrum; eigenvalue
# clipping (Laloux, Cizeau, Bouchaud & Potters 1999); and the rotationally
# invariant estimator (RIE) of Bun, Bouchaud & Potters (2017), whose
# shrinkage function coincides with Ledoit-Wolf (2020) analytic nonlinear
# shrinkage (the two differ only in how the sample Stieltjes transform is
# estimated; here the empirical Stieltjes transform with Lorentzian
# broadening eta = p^(-1/2), following Bun et al.).

export fitMarchenkoPastur, clipEigenvalues, rieEigenvalues,
    cleanCovariance, cleanCorrelation

_mp_edges(γ::Real, σ2::Real) = (σ2 * (1 - sqrt(γ))^2, σ2 * (1 + sqrt(γ))^2)

# Median of the Marcenko-Pastur law with sigma^2 = 1, by quadrature of the
# density on its support.
function _mp_median(γ::Real)
    a, b = _mp_edges(γ, 1.0)
    γ >= 1 && return a  # atom at 0 of mass 1 - 1/gamma makes the median 0-ish;
    # for gamma >= 1 the lower edge is 0 and the median is found by the same
    # quadrature below on [0, b].
    lo = γ >= 1 ? 0.0 : a
    n = 4096
    xs = range(lo, b; length=n + 1)
    h = (b - lo) / n
    dens = [MarchenkoPastur(γ, 1.0) |> d -> pdf(d, x) for x in xs]
    cdf = cumsum(dens) .* h
    cdf ./= cdf[end]
    i = searchsortedfirst(cdf, 0.5)
    return xs[clamp(i, 1, n + 1)]
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
        return median(λ) / _mp_median(γ)
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
eigenvalues). Properties: for a pure-noise spectrum (`Sigma = sigma^2 I`)
the result is flat at `sigma^2`; a supercritical spike `l` observed at
`lambda > l` is shrunk to `xi < l < lambda`.

Requires `gamma < 1` for the zero-eigenvalue block when `gamma >= 1`:
eigenvalues equal to 0 are mapped to 0 (see `cleanCovariance`).
"""
function rieEigenvalues(evals::AbstractVector{<:Real}, γ::Real;
                        η::Real=length(evals)^(-1 / 2))
    λ = collect(float.(evals))
    p = length(λ)
    ξ = similar(λ)
    for i in 1:p
        z = λ[i] - im * η
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
