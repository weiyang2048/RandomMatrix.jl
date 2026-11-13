# Random features: the limiting spectral law of the feature Gram /
# covariance matrix (Mei & Montanari), computed by composing the
# package's Marchenko-Pastur population with its Silverstein solver.
#
# Model: W (m x d) and X (d x n) with i.i.d. N(0,1) entries, features
# Phi = sigma.(W X / sqrt(d)) (m x n), feature covariance
# C = (1/n) Phi Phi'. Writing the Hermite coefficients of sigma for
# Z ~ N(0,1) as
#     mu0 = E sigma(Z),  mu1 = E[Z sigma(Z)] = E sigma'(Z),
#     mu*^2 = E sigma(Z)^2 - mu0^2 - mu1^2,
# the bulk limiting law of C depends on sigma only through
# (mu1, mu*): it is the Silverstein law with population
# H = law of mu1^2 * T + mu*^2, T ~ MP_{m/d}, at aspect gamma = m/n.
# (Condition on W: the columns of the linearized feature matrix are
# i.i.d. with population covariance mu1^2 W W'/d + mu*^2 I, whose
# spectral distribution is H; the sampling over the n columns is then
# Silverstein's. Substituting the Marchenko-Pastur quadratic into the
# Silverstein equation gives the quartic self-consistent equation of
# Pennington & Worah.) mu0 does not enter the bulk: it contributes a
# single outlier eigenvalue near mu0^2 * m.

export randomFeaturesLaw, hermiteCoefficients

# Gauss-Hermite nodes/weights for the weight exp(-x^2/2) on R, by the
# Golub-Welsch algorithm: the orthonormal probabilists' Hermite
# polynomials satisfy x p_k = sqrt(k+1) p_{k+1} + sqrt(k) p_{k-1}, so
# the Jacobi matrix is SymTridiagonal(0, sqrt.(1:K-1)) and the weights
# are sqrt(2 pi) * (first eigenvector component)^2.
function _gauss_hermite(K::Integer)
    F = eigen(SymTridiagonal(zeros(K), sqrt.(1.0:(K - 1))))
    return F.values, sqrt(2π) .* (F.vectors[1, :]) .^ 2
end

"""
    hermiteCoefficients(sigma; npoints=128) -> (mu0, mu1, muStar)

The first Hermite coefficients of an activation `sigma` for
`Z ~ N(0, 1)`:

```math
\\mu_0 = E\\,\\sigma(Z), \\qquad
\\mu_1 = E[Z \\sigma(Z)] = E\\,\\sigma'(Z), \\qquad
\\mu_\\star^2 = E\\,\\sigma(Z)^2 - \\mu_0^2 - \\mu_1^2 \\ge 0,
```

(the middle identity is Stein's lemma). These are the only features
of `sigma` that the limiting spectral law of a random-features Gram
matrix sees; see [`randomFeaturesLaw`](@ref). The expectations are
evaluated by Gauss-Hermite quadrature with `npoints` nodes
(Golub-Welsch on the probabilists' Hermite recurrence). Smooth
activations converge spectrally (machine precision well before 128
nodes for `tanh`, `erf`, polynomials); a kink converges algebraically
-- for ReLU the `mu0`/`muStar` error is about 1.3e-3 at the default
128 nodes and halves as `npoints` doubles, while `mu1` is exact to
machine precision. For ReLU prefer the `:relu` preset of
[`randomFeaturesLaw`](@ref), which uses the closed form.

Closed forms, for reference (all for `Z ~ N(0, 1)`):

  - linear, `sigma(x) = x`: `(0, 1, 0)`.
  - ReLU, `sigma(x) = max(x, 0)`: `mu0 = 1/sqrt(2 pi)` (the Gaussian
    density at 0), `mu1 = E[Z^2 1{Z>0}] = 1/2`,
    `E sigma^2 = 1/2`, hence `mu*^2 = 1/4 - 1/(2 pi)`.
  - `erf`: `mu0 = 0` (odd), `mu1 = E sigma'(Z) = (2/sqrt(pi)) E
    exp(-Z^2) = 2/sqrt(3 pi)`, and `E erf(Z)^2 = (2/pi) asin(2/3)`
    (from `E[erf(aZ) erf(bZ)] = (2/pi) asin(2ab / sqrt((1+2a^2)
    (1+2b^2)))`), hence `mu*^2 = (2/pi) asin(2/3) - 4/(3 pi)`.
  - quadratic, `sigma(x) = (x^2 - 1)/sqrt(2)`: `(0, 0, 1)` -- the
    normalized second Hermite polynomial itself.
"""
function hermiteCoefficients(σ::Function; npoints::Integer=128)
    npoints >= 2 || throw(ArgumentError(
        "hermiteCoefficients requires npoints >= 2, got $npoints"))
    x, w = _gauss_hermite(npoints)
    s = σ.(x)
    μ0 = sum(w .* s) / sqrt(2π)
    μ1 = sum(w .* x .* s) / sqrt(2π)
    e2 = sum(w .* s .^ 2) / sqrt(2π)
    return (μ0, μ1, sqrt(max(e2 - μ0^2 - μ1^2, 0.0)))
end

# Closed-form Hermite triples for the named presets (derivations in
# the hermiteCoefficients docstring).
const _RF_PRESETS = Dict{Symbol,NTuple{3,Float64}}(
    :linear => (0.0, 1.0, 0.0),
    :relu => (1 / sqrt(2π), 0.5, sqrt(0.25 - 1 / (2π))),
    :erf => (0.0, 2 / sqrt(3π), sqrt((2 / π) * asin(2 / 3) - 4 / (3π))),
    :quadratic => (0.0, 0.0, 1.0),
)

"""
    randomFeaturesLaw(mu0, mu1, muStar, psi, phi; npoints=2001, eps=nothing, hpoints=1201)
    randomFeaturesLaw(sigma::Function, psi, phi; quadpoints=128, kwargs...)
    randomFeaturesLaw(preset::Symbol, psi, phi; kwargs...)

The limiting spectral distribution of the random-features covariance
matrix, as a [`FreeMeasure`](@ref).

Model: let ``W`` (``m \\times d``) and ``X`` (``d \\times n``) have
i.i.d. ``N(0,1)`` entries and form the feature matrix ``\\Phi =
\\sigma(W X / \\sqrt d)`` (``m \\times n``; the preactivations are
marginally ``N(0,1)``). This is the Mei-Montanari random-features
model (equivalently: data ``x_j \\sim N(0, I_d/d)`` and weights
``\\theta_i \\sim N(0, I_d)`` with ``\\Phi_{ij} = \\sigma(\\theta_i
\\cdot x_j)``). The object returned is the limiting law of the
feature covariance ``C = \\Phi \\Phi' / n`` (``m \\times m``). The
sample Gram ``\\Phi' \\Phi / n`` (``n \\times n``) has the same
nonzero spectrum; its limiting law is the same continuous part
rescaled by ``m/n``, with an additional atom at 0 of mass ``(1 -
m/n)_+``.

With the Hermite coefficients of `sigma` for ``Z \\sim N(0,1)``
([`hermiteCoefficients`](@ref))

```math
\\mu_0 = E\\,\\sigma(Z), \\qquad \\mu_1 = E[Z\\sigma(Z)], \\qquad
\\mu_\\star^2 = E\\,\\sigma(Z)^2 - \\mu_0^2 - \\mu_1^2,
```

and aspect ratios ``\\psi = n/d`` (samples per dimension) and
``\\phi = m/d`` (features per dimension), the bulk law depends on
`sigma` only through `(mu1, muStar)`: it is the Silverstein law
([`silversteinLaw`](@ref)) with population

```math
H = \\mathrm{law}\\big(\\mu_1^2 T + \\mu_\\star^2\\big), \\qquad
T \\sim MP_{m/d},
```

at aspect ``\\gamma = m/n = \\phi/\\psi``. Derivation: linearizing
``\\sigma`` at the Hermite level, ``\\Phi \\approx \\mu_0 \\mathbf 1
\\mathbf 1' + \\mu_1 W X/\\sqrt d + \\tilde\\Phi`` with
``\\tilde\\Phi`` spectrally equivalent to an i.i.d. Gaussian matrix
of variance ``\\mu_\\star^2`` (the Gaussian equivalence of Mei &
Montanari); conditionally on ``W`` the columns are then i.i.d. with
population covariance ``\\mu_1^2 W W'/d + \\mu_\\star^2 I``, whose
spectral distribution is ``H``, and the column sampling is
Silverstein's. Substituting the Marchenko-Pastur quadratic for
``H``'s transform into the Silverstein equation yields the quartic
self-consistent equation for the Stieltjes transform found by
Pennington & Worah (2017) (see also Benigni & Peche, 2021;
Schroeder & Piccolo, 2021, Thm. 2.2, whose ``\\theta_1 =
E\\sigma(Z)^2``, ``\\theta_2 = (E\\sigma'(Z))^2 = \\mu_1^2`` are the
same two parameters).

Exact reductions and moments:

  - ``\\mu_1 = 0``: ``H`` collapses to a point mass at
    ``\\mu_\\star^2`` and the law is exactly the Marchenko-Pastur law
    ``MP_{m/n}`` scaled by ``\\mu_\\star^2`` -- the
    Pennington-Worah isospectral case ``\\theta_2 = 0``.
  - linear `sigma` (``\\mu_1 = 1``, ``\\mu_\\star = 0``): the law of
    a product of two Wishart matrices, ``MP_{m/d} \\boxtimes
    MP_{m/n}`` (the quartic degenerates to a cubic).
  - the mean is ``\\mu_1^2 + \\mu_\\star^2`` and the second moment
    ``\\mu_1^4 (1 + m/d) + 2\\mu_1^2\\mu_\\star^2 + \\mu_\\star^4 +
    (m/n)(\\mu_1^2 + \\mu_\\star^2)^2`` (Silverstein's moment formulas
    applied to ``H``).
  - atom at 0 of mass ``\\max(H(\\{0\\}), 1 - n/m)``, with
    ``H(\\{0\\}) = (1 - d/m)_+`` when ``\\mu_\\star = 0`` and 0
    otherwise: the rank deficiency of ``C``, whichever of the two
    rectangles causes more of it.

`mu0` does not enter the bulk law. It produces a single outlier
eigenvalue of ``C`` at ``\\mu_0^2 m (1 + o(1))`` (a rank-one mean
component; measured at ``1.014``--``1.019`` times ``\\mu_0^2 m`` for
``m = 300``--``375`` in the calibration simulations), which the
limiting law does not see; simulations comparing against this law
should drop the top eigenvalue when `mu0 != 0`, exactly as for the
trivial eigenvalue of a regular graph in the Kesten-McKay law.

The `sigma::Function` method computes the Hermite coefficients by
Gauss-Hermite quadrature (`quadpoints` nodes). The `preset::Symbol`
method accepts `:linear`, `:relu`, `:erf` and `:quadratic`, whose
coefficients are the closed forms in [`hermiteCoefficients`](@ref).

Accuracy: inherited from [`silversteinLaw`](@ref) (density to about
1e-3 in the interior). At the default Stieltjes regularization the
first two moments of the composed law carry a positive bias of up to
about 1.5% (the population's own discretization enters through
`hpoints`); passing a smaller `eps` (e.g. `eps = 5e-5`) reduces the
moment bias to about 0.1%. The `mu1 = 0` reduction to
Marchenko-Pastur is accurate to the `silversteinLaw` level (moments
to ~0.1%, interior density to ~2e-3). Calibrated against
formed-matrix simulation at `(d, n, m) = (300, 450, 375)`: the
empirical spectral moments of ReLU features (top eigenvalue dropped)
and of erf features agree with the exact moment formulas to within
simulation error (rep-to-rep sd), the bulk CDF distance is about
0.01, and the ReLU outlier sits at 1.014--1.019 times `mu0^2 m`. As
with `silversteinLaw`, this law has no closed-form density, cdf or
sampler; the grid `FreeMeasure` is the working representation
(`freeMean`, `freeVar`, `freeDensity`, `freeAtoms`,
`cauchyTransform` all apply).

# Examples

```julia
# ReLU features, n/d = 1.5 samples per dimension, m/d = 1.25
# features per dimension: mean mu1^2 + mu*^2 = 0.3408.
F = randomFeaturesLaw(:relu, 1.5, 1.25)
freeMean(F)   # approx 0.343

# The same law from the Hermite triple (mu0 is inert for the bulk):
randomFeaturesLaw(1/sqrt(2*pi), 0.5, sqrt(0.25 - 1/(2*pi)), 1.5, 1.25)
```
"""
function randomFeaturesLaw(μ0::Real, μ1::Real, μstar::Real, ψ::Real, φ::Real;
                           npoints::Integer=2001, eps=nothing,
                           hpoints::Integer=1201)
    (isfinite(ψ) && ψ > 0) || throw(ArgumentError(
        "randomFeaturesLaw requires a finite psi = n/d > 0, got psi = $ψ"))
    (isfinite(φ) && φ > 0) || throw(ArgumentError(
        "randomFeaturesLaw requires a finite phi = m/d > 0, got phi = $φ"))
    isfinite(μ0) || throw(ArgumentError(
        "randomFeaturesLaw requires a finite mu0, got mu0 = $μ0"))
    isfinite(μ1) || throw(ArgumentError(
        "randomFeaturesLaw requires a finite mu1, got mu1 = $μ1"))
    (isfinite(μstar) && μstar >= 0) || throw(ArgumentError(
        "randomFeaturesLaw requires a finite muStar >= 0, got muStar = $μstar"))
    ν = float(μstar)^2
    γ = float(φ) / float(ψ)
    if μ1 == 0
        # Population collapses to a point mass at nu: exactly the
        # scaled Marchenko-Pastur law. (Also covers the constant
        # activation mu1 = muStar = 0, for which silversteinLaw
        # returns diracLaw(0).)
        return silversteinLaw([ν], [1.0], γ; npoints=npoints, eps=eps)
    end
    pop = freePoissonLaw(φ; scale=float(μ1)^2, npoints=hpoints)
    nodes = pop.nodes .+ ν
    atoms = [(a + ν) => w for (a, w) in pop.atoms]
    return silversteinLaw(FreeMeasure(nodes, pop.weights, atoms), γ;
                          npoints=npoints, eps=eps)
end

function randomFeaturesLaw(σ::Function, ψ::Real, φ::Real;
                           quadpoints::Integer=128, kwargs...)
    μ0, μ1, μstar = hermiteCoefficients(σ; npoints=quadpoints)
    return randomFeaturesLaw(μ0, μ1, μstar, ψ, φ; kwargs...)
end

function randomFeaturesLaw(preset::Symbol, ψ::Real, φ::Real; kwargs...)
    haskey(_RF_PRESETS, preset) || throw(ArgumentError(
        "randomFeaturesLaw: unknown preset $(repr(preset)); " *
        "expected one of $(sort(collect(keys(_RF_PRESETS))))"))
    μ0, μ1, μstar = _RF_PRESETS[preset]
    return randomFeaturesLaw(μ0, μ1, μstar, ψ, φ; kwargs...)
end
