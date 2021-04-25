export ComplexNormal, Gaussian,
        
        # re-export    
        mean, var

"""
```julia
ComplexNormal(μ=0,σ=1) 
```
"""
struct  ComplexNormal <:ContinuousUnivariateDistribution
    μ::ComplexF64
    σ::Float64
    function ComplexNormal(μ=0,σ=1) new(μ,σ) end
end

Base.eltype(::Type{ComplexNormal}) = ComplexF64
Base.rand(rng::AbstractRNG, d::ComplexNormal) = randn(rng,ComplexF64)*d.σ+d.μ;

mean(d::ComplexNormal)=d.μ
var(d::ComplexNormal)=d.σ

"""
```julia
Gaussian(beta=1,μ=0,σ=1)
````
- `beta` 1 for Real Gaussian, 2 for Complex Gaussian 
"""
struct Gaussian <: ContinuousUnivariateDistribution
    beta::Int8
    μ::Union{ComplexF64,Float64}
    σ::Float64
    function Gaussian(beta=1,μ=0,σ=1)  beta==1 ? Normal(μ,σ) : ComplexNormal(μ,σ) end
end
