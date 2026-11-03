# Spiked MANOVA (Jacobi) models and their BBP phase transition.
#
# Model: A ~ W_p(Sigma, n1) and B ~ W_p(I, n2) independent Wisharts, with
# Sigma a spiked identity (bulk at 1, a few eigenvalues theta != 1). The
# MANOVA matrix (A + B)^{-1} A has eigenvalues in (0, 1) whose bulk follows
# the Wachter law with aspect ratios gamma1 = p / n1 and gamma2 = p / n2
# (see Wachter in Distributions.jl). A spike theta produces an outlier
# sample eigenvalue only outside a critical window around 1; inside the
# window the spike is absorbed by the Wachter bulk and the associated
# sample eigenvalue sticks to the nearer support edge.
#
# References: Wang & Yao, "Extreme eigenvalues of large-dimensional
# spiked Fisher matrices with application", Ann. Statist. 45(1), 2017,
# 415-460, Theorem 3.1 (proved there for the Fisher matrix
# F = S2^{-1} S1; the map below converts it to the MANOVA eigenvalues);
# Dharmawansa, Johnstone & Onatski (2014) for the Gaussian case.

export manovaThreshold, manovaSpikeLocation

function _manova_check(γ₁::Real, γ₂::Real)
    (0 < γ₁ < 1 && 0 < γ₂ < 1) || throw(ArgumentError(
        "spiked MANOVA aspect ratios must lie in (0, 1), got γ₁ = $γ₁, γ₂ = $γ₂"))
    return nothing
end

# Wang & Yao's Fisher displacement phi, mapped to a MANOVA eigenvalue:
# phi(theta) is the limit of the outlier of S2^{-1} S1 (sample
# covariances, df n1 and n2); the eigenvalues t of B^{-1} A satisfy
# t = (n1 / n2) phi = (gamma2 / gamma1) phi, and the eigenvalues of
# (A + B)^{-1} A are x = t / (1 + t).
function _manova_map(θ::Real, γ₁::Real, γ₂::Real)
    λ = θ * (θ - 1 + γ₁) / ((1 - γ₂) * θ - 1)
    t = (γ₂ / γ₁) * λ
    return t / (1 + t)
end

"""
    manovaThreshold(γ₁, γ₂)

Critical population spike for the BBP phase transition in a spiked
MANOVA model: for `A ~ W_p(Σ, n₁)`, `B ~ W_p(I, n₂)` with aspect
ratios `γ₁ = p / n₁` and `γ₂ = p / n₂`, a population spike `θ` (an
eigenvalue of `Σ`, bulk at 1) yields an outlier eigenvalue of
`(A + B)⁻¹ A` only if

```math
\\theta > \\frac{1 + \\rho}{1 - \\gamma_2},
\\qquad \\rho = \\sqrt{\\gamma_1 + \\gamma_2 - \\gamma_1 \\gamma_2}
```

(the same `ρ` as in the [`Wachter`](@ref) support). Spikes below 1
separate only under the lower critical value `(1 - ρ) / (1 - γ₂)`;
see [`manovaSpikeLocation`](@ref).
"""
function manovaThreshold(γ₁::Real, γ₂::Real)
    _manova_check(γ₁, γ₂)
    return (1 + _wachter_rho(γ₁, γ₂)) / (1 - γ₂)
end

"""
    manovaSpikeLocation(θ, γ₁, γ₂)

Almost-sure limit of the sample eigenvalue of `(A + B)⁻¹ A` generated
by a population spike `θ` (an eigenvalue of `Σ` in
`A ~ W_p(Σ, n₁)`, bulk at 1), with aspect ratios `γ₁ = p / n₁`,
`γ₂ = p / n₂`. With `ρ = √(γ₁ + γ₂ - γ₁γ₂)` the critical spikes are
`θ₊ = (1 + ρ) / (1 - γ₂)` (see [`manovaThreshold`](@ref)) and
`θ₋ = (1 - ρ) / (1 - γ₂)`:

- `θ ≥ θ₊`: the outlier sits above the Wachter support, at
  `x = t / (1 + t)` with `t = (γ₂ / γ₁) φ(θ)` and Wang & Yao's Fisher
  displacement `φ(θ) = θ (θ - 1 + γ₁) / ((1 - γ₂) θ - 1)`.
- `θ ≤ θ₋`: the outlier sits below the support, at the same map.
- `θ₋ < θ < θ₊`: the spike is absorbed by the bulk; the sample
  eigenvalue sticks to the nearer Wachter edge
  `x± = γ₂ (γ₁ ± ρ)² / (γ₁ + γ₂)²` (the upper edge for `θ ≥ 1`, the
  lower for `θ < 1`).

The map is continuous at the critical values: at `θ = θ₊` (resp.
`θ₋`) it returns exactly `x₊` (resp. `x₋`).
"""
function manovaSpikeLocation(θ::Real, γ₁::Real, γ₂::Real)
    _manova_check(γ₁, γ₂)
    θ > 0 || throw(ArgumentError("spike θ must be positive, got θ = $θ"))
    ρ = _wachter_rho(γ₁, γ₂)
    θ₊ = (1 + ρ) / (1 - γ₂)
    θ₋ = (1 - ρ) / (1 - γ₂)
    if θ >= θ₊ || θ <= θ₋
        return _manova_map(θ, γ₁, γ₂)
    end
    x₋, x₊ = _wachter_edges(γ₁, γ₂)
    return θ >= 1 ? x₊ : x₋
end
