export ComplexNormal, Gaussian, Circular, Elliptic, MarchenkoPastur,
    stieltjes
# re-export    
export mean, var, pdf, rand

stieltjes(d::Semicircle) = z -> (-z + sqrt(z^2 - d.r^2)) / 2


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
