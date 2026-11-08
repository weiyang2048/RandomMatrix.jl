export ComplexNormal, Gaussian, Circular, Elliptic, MarchenkoPastur,
    Wachter, fitWachter, stieltjes
# re-export    
export mean, var, pdf, rand

"""
    stieltjes(d::Semicircle)

Return the Stieltjes transform of the semicircle law `d` as a function of
one argument `z`:

```math
m(z) = \\frac{-z + \\sqrt{z^2 - r^2}}{2}, \\qquad r = d.r .
```

# Examples
```julia
stieltjes(Semicircle(2))(0)  # == im
```
"""
stieltjes(d::Semicircle) = z -> (-z + sqrt(complex(z^2 - d.r^2))) / 2


"""
```julia
MarchenkoPastur <:ContinuousUnivariateDistribution

MarchenkoPastur(λ,σ) 
pdf(d::MarchenkoPastur,x::Real)  
```
- `λ` : default `0.5`
- `σ` : default `1`

# Examples 

Generate a MP rv with ρ = 0.5, σ = 1
```julia 
rand(MarchenkoPastur())

1.2656480923753979
```

Compute the desity for the MP distribution with λ=1.6 at the point 0
```julia
pdf(MarchenkoPastur(1.6),0)

0.375
```

Generate 100 MP rvs with ρ=0.1 and σ = 2
```julia
rand(MarchenkoPastur(0.1,2),100)

100-element Vector{Float64}:
 5.31001367107032
 2.3707745658317116
 2.5378523986772343
 2.585256212138476
 ⋮
 3.66799182506567
 6.517865226831382
 3.7628212250212423
 4.582520400697299
```
"""
struct MarchenkoPastur <: ContinuousUnivariateDistribution
    λ::Float64
    σ::Float64
    function MarchenkoPastur(λ=0.5, σ=1)
        new(λ, σ)
    end
end

function Distributions.pdf(d::MarchenkoPastur, x::Real)

    if d.λ > 1
        if x < 0
            return 0
        end
        if x == 0
            return 1 - 1 / d.λ
        end
    end

    λ₋, λ₊ = d.σ^2 * (1 ± sqrt(d.λ)) .^ 2
    if x < λ₋ || x > λ₊
        return 0
    else
        return 1 / (2π * d.σ^2) * (sqrt((λ₊ - x) * (x - λ₋))) / (d.λ * x)
    end
end

function Base.rand(rng::AbstractRNG, d::MarchenkoPastur)
    if d.λ > 1
        if rand() <= 1 - 1 / d.λ
            return 0
        end
    end

    λ₋, λ₊ = d.σ^2 * ±(1, sqrt(d.λ)) .^ 2
    while true
        x = rand(Uniform(λ₋, λ₊))
        if rand() <= pdf(d, x)
            return x
        end
    end
end

"""
    Wachter(γ₁, γ₂)
    Wachter()

The Wachter law (the MANOVA / Jacobi limiting spectral distribution):
the limiting distribution of an eigenvalue of ``(A + B)^{-1} A``, where
``A \\sim W_p(I, n_1)`` and ``B \\sim W_p(I, n_2)`` are independent
Wishart matrices whose degrees of freedom grow proportionally with
``p``, with aspect ratios

```math
\\gamma_1 = p / n_1, \\qquad \\gamma_2 = p / n_2
```

(the same convention as [`MarchenkoPastur`](@ref), whose ``\\lambda``
is ``p / n``). With ``\\rho = \\sqrt{\\gamma_1 + \\gamma_2 -
\\gamma_1 \\gamma_2}`` the continuous part has density

```math
f(x) = \\frac{\\gamma_1 + \\gamma_2}{\\gamma_1 \\gamma_2} \\,
\\frac{\\sqrt{(x_+ - x)(x - x_-)}}{2\\pi x (1 - x)},
\\qquad x \\in [x_-, x_+],
```

on the support ``x_\\pm = \\gamma_2 (\\gamma_1 \\pm \\rho)^2 /
(\\gamma_1 + \\gamma_2)^2``. If ``\\gamma_1 > 1`` (the numerator
Wishart is rank-deficient) there is an atom at ``0`` of mass
``1 - 1/\\gamma_1``; if ``\\gamma_2 > 1`` there is an atom at ``1``
of mass ``1 - 1/\\gamma_2``. Following the convention of
[`MarchenkoPastur`](@ref), `pdf` at an atom location returns the
atom mass. The mean and variance are

```math
\\frac{\\gamma_2}{\\gamma_1 + \\gamma_2},
\\qquad
\\frac{(\\gamma_1 \\gamma_2)^2}{(\\gamma_1 + \\gamma_2)^3},
```

and ``1 - X`` for ``X \\sim`` `Wachter(γ₁, γ₂)` is
`Wachter(γ₂, γ₁)` (the roles of the two Wisharts swap). The
parameters must satisfy ``\\gamma_1, \\gamma_2 > 0`` and
``\\rho^2 = \\gamma_1 + \\gamma_2 - \\gamma_1\\gamma_2 > 0``,
equivalently ``n_1 + n_2 > p`` (so that ``A + B`` is invertible).
Defaults are `γ₁ = γ₂ = 0.5`, the symmetric case with mean `1/2`.

`rand` uses exact rejection sampling in the variable
``x = x_- + (x_+ - x_-) \\sin^2\\theta``, in which the density is
bounded; `cdf` integrates the density in the same variable
(Simpson's rule), in which it is smooth even when a support edge
touches `0` or `1`.

# Examples

```julia
d = Wachter(0.25, 0.5)   # eigenvalues of (A + B)^{-1} A, p/n1 = 1/4, p/n2 = 1/2
mean(d)                  # 0.666...
pdf(d, 0.5)              # 1.2732...
rand(d)                  # a draw in [0, 1]
```

References: Wachter, "The limiting empirical measure of multiple
discriminant ratios", Ann. Statist. 8(5), 1980, 937–957; Collins,
"Product of random projections, Jacobi ensembles and universality
problems arising from free probability", Probab. Theory Related
Fields 133, 2005, 315–349; Johnstone, "Multivariate analysis and
Jacobi ensembles: Largest eigenvalue, Tracy–Widom limits and rates
of convergence", Ann. Statist. 36(6), 2008, 2638–2716.
"""
struct Wachter <: ContinuousUnivariateDistribution
    γ₁::Float64
    γ₂::Float64
    function Wachter(γ₁=0.5, γ₂=0.5)
        (γ₁ > 0 && γ₂ > 0 && isfinite(γ₁) && isfinite(γ₂)) ||
            throw(ArgumentError("Wachter aspect ratios must be positive and finite, got γ₁ = $γ₁, γ₂ = $γ₂"))
        γ₁ + γ₂ - γ₁ * γ₂ > 0 || throw(ArgumentError(
            "Wachter requires γ₁ + γ₂ - γ₁γ₂ > 0 (equivalently n₁ + n₂ > p), got γ₁ = $γ₁, γ₂ = $γ₂"))
        new(γ₁, γ₂)
    end
end

# The Wachter discriminant ρ = sqrt(γ₁ + γ₂ − γ₁γ₂); the same ρ drives
# the spiked MANOVA threshold in SpikedManova.jl.
_wachter_rho(γ₁::Real, γ₂::Real) = sqrt(γ₁ + γ₂ - γ₁ * γ₂)

# Support edges (x₋, x₊) of the continuous part.
function _wachter_edges(γ₁::Real, γ₂::Real)
    ρ = _wachter_rho(γ₁, γ₂)
    s = γ₁ + γ₂
    return (γ₂ * (γ₁ - ρ)^2 / s^2, γ₂ * (γ₁ + ρ)^2 / s^2)
end
_wachter_edges(d::Wachter) = _wachter_edges(d.γ₁, d.γ₂)

# Atom masses (at 0, at 1) from rank-deficient Wisharts.
_wachter_atoms(d::Wachter) =
    (max(0.0, 1 - 1 / d.γ₁), max(0.0, 1 - 1 / d.γ₂))

mean(d::Wachter) = d.γ₂ / (d.γ₁ + d.γ₂)
var(d::Wachter) = (d.γ₁ * d.γ₂)^2 / (d.γ₁ + d.γ₂)^3

Distributions.minimum(d::Wachter) = 0.0
Distributions.maximum(d::Wachter) = 1.0

function Distributions.pdf(d::Wachter, x::Real)
    a₀, a₁ = _wachter_atoms(d)
    x₋, x₊ = _wachter_edges(d)
    if x == 0
        # Atom mass, the MarchenkoPastur convention; with no atom the
        # density either vanishes below x₋ or diverges like x^(-1/2)
        # at an edge touching 0 (γ₁ = 1).
        return a₀ > 0 ? a₀ : (x₋ == 0 ? Inf : 0.0)
    elseif x == 1
        return a₁ > 0 ? a₁ : (x₊ == 1 ? Inf : 0.0)
    end
    (x < x₋ || x > x₊) && return 0.0
    pref = (d.γ₁ + d.γ₂) / (d.γ₁ * d.γ₂)
    return pref * sqrt((x₊ - x) * (x - x₋)) / (2π * x * (1 - x))
end

# Integrand of the continuous part after x = x₋ + Δ sin²θ, which
# removes both square-root edge singularities: with s = sin²θ the
# transformed density is pref Δ² s(1 - s) / (π x(θ) (1 - x(θ))).
# At an endpoint where the edge touches 0 or 1 the direct expression
# is 0/0; its finite limit is pref Δ / π, used there instead.
function _wachter_theta_density(d::Wachter, θ::Float64, x₋::Float64, x₊::Float64)
    Δ = x₊ - x₋
    pref = (d.γ₁ + d.γ₂) / (d.γ₁ * d.γ₂)
    if θ == 0
        return x₋ == 0 ? pref * Δ / π : 0.0
    elseif θ == π / 2
        return x₊ == 1 ? pref * Δ / π : 0.0
    end
    s = sin(θ)^2
    x = x₋ + Δ * s
    return pref * Δ^2 * s * (1 - s) / (π * x * (1 - x))
end

function Distributions.cdf(d::Wachter, x::Real)
    a₀, a₁ = _wachter_atoms(d)
    x < 0 && return 0.0
    x == 0 && return a₀
    x >= 1 && return 1.0
    x₋, x₊ = _wachter_edges(d)
    x < x₋ && return a₀
    x >= x₊ && return 1 - a₁
    θx = asin(sqrt((x - x₋) / (x₊ - x₋)))
    n = 2000  # Simpson intervals
    h = θx / n
    s = 0.0
    for i in 0:n
        w = (i == 0 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        s += w * _wachter_theta_density(d, i * h, x₋, x₊)
    end
    return a₀ + s * h / 3
end

function Base.rand(rng::AbstractRNG, d::Wachter)
    a₀, a₁ = _wachter_atoms(d)
    if a₀ > 0 || a₁ > 0
        t = rand(rng)
        t < a₀ && return 0.0
        (a₁ > 0 && t >= 1 - a₁) && return 1.0
    end
    x₋, x₊ = _wachter_edges(d)
    Δ = x₊ - x₋
    # Rejection sampling in θ (x = x₋ + Δ sin²θ): the acceptance
    # probability Δ² s(1 - s) / (x (1 - x)) is at most 1 because
    # x >= Δ s and 1 - x >= Δ (1 - s).
    while true
        θ = (π / 2) * rand(rng)
        s = sin(θ)^2
        x = x₋ + Δ * s
        (x <= 0 || x >= 1) && return x
        rand(rng) * x * (1 - x) <= Δ^2 * s * (1 - s) && return x
    end
end

"""
    fitWachter(evals) -> (γ₁ = ..., γ₂ = ...)

Estimate the Wachter aspect ratios ``γ₁ = p/n₁`` and ``γ₂ = p/n₂`` from a
spectrum `evals` of eigenvalues of a MANOVA matrix ``(A + B)^{-1} A`` —
the analogue of `fitMarchenkoPastur` for the Jacobi case.

Method of moments: the [`Wachter`](@ref) law has mean
``m = γ₂ / (γ₁ + γ₂)`` and variance ``v = (γ₁ γ₂)^2 / (γ₁ + γ₂)^3``.
With ``s = γ₁ + γ₂`` the variance factors as ``v = s (m (1 - m))^2``, so
the empirical mean and population variance of `evals` invert in closed
form: ``s = v / (m (1 - m))^2``, ``γ₂ = m s``, ``γ₁ = (1 - m) s``. The
atoms at 0 and 1 (for ``γ₁ > 1`` / ``γ₂ > 1``) need no special handling —
they enter the empirical moments directly.

An `ArgumentError` is thrown when no Wachter law can match the spectrum:
empty input, eigenvalues outside ``[0, 1]``, a degenerate spectrum (zero
variance, or a single atom), or a variance at the Bernoulli bound
``v = m (1 - m)`` — a spectrum supported on ``{0, 1}`` alone, which is
also exactly where the Wachter domain ``γ₁ + γ₂ - γ₁ γ₂ > 0`` closes.
"""
function fitWachter(evals::AbstractVector{<:Real})
    λ = collect(float.(evals))
    isempty(λ) && throw(ArgumentError("fitWachter: no eigenvalues given"))
    all(x -> isfinite(x) && -1e-8 <= x <= 1 + 1e-8, λ) || throw(ArgumentError(
        "fitWachter: MANOVA eigenvalues must lie in [0, 1], got range " *
        "[$(minimum(λ)), $(maximum(λ))]"))
    m = mean(λ)
    v = var(λ; corrected = false)
    0 < m < 1 || throw(ArgumentError(
        "fitWachter: spectrum mean $m is not in (0, 1) — the spectrum is a single atom"))
    v > 0 || throw(ArgumentError(
        "fitWachter: spectrum has zero variance — no Wachter law fits a single atom"))
    v < m * (1 - m) || throw(ArgumentError(
        "fitWachter: variance $v reaches the Bernoulli bound m(1 - m) = " *
        "$(m * (1 - m)) for a [0, 1] spectrum — the spectrum is supported " *
        "on {0, 1} alone and no Wachter law matches it"))
    s = v / (m * (1 - m))^2
    return (γ₁ = (1 - m) * s, γ₂ = m * s)
end

"""
```julia
ComplexNormal(μ,σ)
```
- `μ` : mean, `μ = 0` by default
- `σ` : standard deviation, `σ  =1 ` by default

```julia 
# Examples

# Generates 10 iid standard Complex Gaussian r.v.s
rand(ComplexNormal(), 10) 

# Generates complex normal with mean 1+1im, variance 4
rand(ComplexNormal(1+1im,2)) 
```
"""
struct ComplexNormal <: ContinuousUnivariateDistribution
    μ::ComplexF64
    σ::Float64
    function ComplexNormal(μ=0, σ=1)
        @assert(σ >= 0, "σ must be non-negative")
        new(μ, σ)
    end
end

Base.eltype(::Type{ComplexNormal}) = ComplexF64
rand(rng::AbstractRNG, d::ComplexNormal) = randn(rng, ComplexF64) * d.σ + d.μ;

mean(d::ComplexNormal) = d.μ
var(d::ComplexNormal) = d.σ

"""
```julia
Gaussian(beta,μ,σ)
```
- `beta` : 1 (default) for Real Gaussian, 2 for Complex Gaussian 
- `μ` : mean, `μ = 0` by default
-  `σ` : standard deviation, `σ  =1 ` by default

```julia 
# Generates 10 iid standard normal r.v.s
rand(Gaussian(), 10) 

# Generates a complex normal with mean 1+1im, variance 4
rand(Gaussian(2,1+1im,2))
```
"""
struct Gaussian <: ContinuousUnivariateDistribution
    beta::Int8
    μ::Union{ComplexF64,Float64}
    σ::Float64
    function Gaussian(beta=1::Int, μ=0::Int, σ=1::Int)
        if σ >= 0
            (beta == 1 ? Normal(μ, σ) : ComplexNormal(μ, σ))
        else
            error("σ must be non-negative")
        end
    end
end

mean(d::Gaussian) = d.μ
var(d::Gaussian) = d.σ
"""
```julia
Circular(c=0,R=1)
```
- The uniform distribution on the complex disk with center `c` and radius `R`

```julia
rand(Circular(1,10),200)
```
> Generates 200 uniform r.v.s on the complex disk of center 1 radius 10.
"""
struct Circular <: ContinuousUnivariateDistribution
    c::ComplexF64
    R::Float64
    function Circular(c=0, R=1)
        new(c, R)
    end
end

Base.eltype(::Type{Circular}) = ComplexF64
function rand(rng::AbstractRNG, d::Circular)
    while true
        x_1 = rand(rng, Uniform(-1, 1))
        x_2 = rand(rng, Uniform(-1, 1))
        if x_1^2 + x_2^2 <= 1
            return x_1 * d.R + x_2 * d.R * im + d.c
        end
    end
end

mean(d::Circular) = d.c


"""
```julia
Elliptic <:ContinuousUnivariateDistribution

Elliptic(ρ;c,R)
```
- the uniform rv on an ellipse of width `2(1+ρ)R`, height `2(1-ρ)R` centered at `c`
"""
struct Elliptic <: ContinuousUnivariateDistribution
    ρ::Float64
    c::ComplexF64
    R::Float64

    function Elliptic(ρ=0.5; c=0, R=1)
        @assert(abs(ρ) < 1, "ρ must be less than 1 in absolute value")
        new(ρ, c, R)
    end
end

Base.eltype(::Type{Elliptic}) = ComplexF64

function rand(rng::AbstractRNG, d::Elliptic)
    while true
        x_1 = rand(rng, Uniform(-(1 + d.ρ), (1 + d.ρ)))
        x_2 = rand(rng, Uniform(-(1 - d.ρ), (1 - d.ρ)))
        if x_1^2 / (1 + d.ρ)^2 + x_2^2 / (1 - d.ρ)^2 <= 1
            return x_1 * d.R + x_2 * d.R * im + d.c
        end
    end
end

mean(d::Elliptic) = d.c

# Distributions ships a generic `_rand!` only for real-valued univariate
# distributions. The complex-valued distributions defined in this file
# (ComplexNormal, Circular, Elliptic) need their own so that array sampling
# such as `rand(d, n)` — used by `randDiagonal` — works.
function _fill_rand!(rng::AbstractRNG, d::Union{ComplexNormal,Circular,Elliptic}, A::AbstractArray)
    for i in eachindex(A)
        A[i] = rand(rng, d)
    end
    return A
end

Distributions._rand!(rng::AbstractRNG, d::Union{ComplexNormal,Circular,Elliptic}, A::AbstractArray) =
    _fill_rand!(rng, d, A)

# Disambiguation: Distributions also defines `_rand!` for a univariate
# sampleable with `A::AbstractArray{<:Real}`; the method above is ambiguous
# with it for these distributions, so the intersection case gets its own
# (more specific) method. Filling a real array with complex samples throws
# on assignment, exactly as the generic Distributions method would.
Distributions._rand!(rng::AbstractRNG, d::Union{ComplexNormal,Circular,Elliptic}, A::AbstractArray{<:Real}) =
    _fill_rand!(rng, d, A)
